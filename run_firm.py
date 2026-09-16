"""Evaluate FIRM on clean or corrupted Kinetics50/VGGSound streams."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import dataloader
import models
from firm import FIRMState


VIDEO_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
)
AUDIO_CORRUPTIONS = (
    "gaussian_noise", "traffic", "crowd", "rain", "thunder", "wind"
)
METHODS = ("source", "firm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", choices=("ks50", "vggsound"), required=True)
    parser.add_argument("--json-root", type=Path, required=True)
    parser.add_argument("--label-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prototype-temperature", type=float, default=0.07)
    parser.add_argument("--evidence-weight", type=float, default=0.25)
    parser.add_argument("--projection-iterations", type=int, default=20)
    parser.add_argument("--disable-drift-gate", action="store_true")
    parser.add_argument(
        "--corruption-modality",
        choices=("none", "video", "audio"),
        required=True,
    )
    parser.add_argument("--corruption", default="all")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--dataset-mean", type=float, default=-5.081)
    parser.add_argument("--dataset-std", type=float, default=4.4849)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def corruption_names(modality: str, requested: str) -> tuple[str, ...]:
    choices = {
        "none": ("clean",),
        "video": VIDEO_CORRUPTIONS,
        "audio": AUDIO_CORRUPTIONS,
    }[modality]
    if requested == "all":
        return choices
    if requested not in choices:
        raise ValueError(f"Invalid {modality} corruption: {requested}")
    return (requested,)


def stream_json(args: argparse.Namespace, corruption: str) -> Path:
    if args.corruption_modality == "none":
        return args.json_root / "clean" / "severity_0.json"
    return (
        args.json_root / args.corruption_modality / corruption
        / f"severity_{args.severity}.json"
    )


def build_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.DataParallel, int]:
    classes = 50 if args.dataset == "ks50" else 309
    model = models.CAVMAEFT(label_dim=classes, modality_specific_depth=11)
    model = torch.nn.DataParallel(model)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.to(device).eval().requires_grad_(False)
    return model, classes


def extract_features(
    model: torch.nn.DataParallel,
    audio: torch.Tensor,
    video: torch.Tensor,
) -> dict[str, torch.Tensor]:
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        result = model.module.forward_eval_with_features(a=audio, v=video)
    if not bool(result["mask"]["full"].all()):
        raise ValueError("FIRM expects paired audio-video inputs")
    return {
        "fusion": result["feat"].float(),
        "audio": result["ca"].float(),
        "video": result["cv"].float(),
        "logits": result["logits"].float(),
    }


def run_stream(
    args: argparse.Namespace,
    corruption: str,
    model: torch.nn.DataParallel,
    classes: int,
    device: torch.device,
) -> list[dict[str, object]]:
    seed_everything(args.seed)
    torch.cuda.empty_cache()
    classifier_weight = model.module.mlp_head[-1].weight.detach()
    state = FIRMState(
        num_classes=classes,
        feature_dim=classifier_weight.shape[1],
        device=device,
        classifier_weight=classifier_weight,
        prototype_temperature=args.prototype_temperature,
        evidence_weight=args.evidence_weight,
        projection_iterations=args.projection_iterations,
        drift_gate=not args.disable_drift_gate,
    )
    audio_conf = {
        "num_mel_bins": 128,
        "target_length": args.target_length,
        "freqm": 0,
        "timem": 0,
        "mixup": 0,
        "dataset": args.dataset,
        "mode": "eval",
        "mean": args.dataset_mean,
        "std": args.dataset_std,
        "noise": False,
        "im_res": 224,
    }
    json_file = stream_json(args, corruption)
    if not json_file.is_file():
        raise FileNotFoundError(json_file)
    loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(
            str(json_file), label_csv=str(args.label_csv), audio_conf=audio_conf
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    correct = {
        name: torch.zeros((), dtype=torch.long, device=device)
        for name in METHODS
    }
    samples = 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    cuda_intervals = []
    progress = tqdm(loader, desc=f"{args.dataset}/{args.corruption_modality}/{corruption}")
    for audio, video, labels in progress:
        audio = audio.to(device, non_blocking=True)
        video = video.to(device, non_blocking=True)
        targets = labels.argmax(dim=1).to(device, non_blocking=True)
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
        features = extract_features(model, audio, video)
        for index in range(len(targets)):
            outputs = state.process_sample(
                features["fusion"][index:index + 1],
                features["audio"][index:index + 1],
                features["video"][index:index + 1],
                features["logits"][index:index + 1],
            )
            target = targets[index:index + 1]
            for name in METHODS:
                probability = outputs[name]
                correct[name].add_(probability.argmax(dim=1).eq(target).sum())
            samples += 1
        cuda_end.record()
        cuda_intervals.append((cuda_start, cuda_end))

    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    cuda_seconds = sum(
        start.elapsed_time(end) for start, end in cuda_intervals
    ) / 1000.0
    final_correct = {name: int(value.item()) for name, value in correct.items()}
    source_accuracy = 100.0 * final_correct["source"] / max(samples, 1)
    common = {
        "dataset": args.dataset,
        "modality": args.corruption_modality,
        "corruption": corruption,
        "severity": 0 if args.corruption_modality == "none" else args.severity,
        "samples": samples,
        "wall_seconds": wall_seconds,
        "cuda_compute_seconds": cuda_seconds,
        "samples_per_second": samples / max(wall_seconds, 1e-12),
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        ),
        "updated_parameters": 0,
        "uses_backpropagation": False,
    }
    rows = []
    for method in METHODS:
        accuracy = 100.0 * final_correct[method] / max(samples, 1)
        row = {
            **common,
            "method": method,
            "accuracy": accuracy,
            "delta_source": accuracy - source_accuracy,
        }
        print("FIRM_RESULT", json.dumps(row))
        rows.append(row)
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for CAV-MAE feature extraction")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    model, classes = build_model(args, device)
    rows = []
    for corruption in corruption_names(
        args.corruption_modality, args.corruption
    ):
        rows.extend(run_stream(args, corruption, model, classes, device))

    with (args.output_dir / "result.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Completed:", args.output_dir / "result.csv")


if __name__ == "__main__":
    main()
