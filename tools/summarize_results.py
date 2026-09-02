"""Aggregate FIRM output files into clean/video/audio/corruption means."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped = defaultdict(list)
    for path in args.root.rglob("result.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                grouped[(row["dataset"], row["method"], row["modality"])].append(
                    float(row["accuracy"])
                )

    rows = []
    for dataset in ("ks50", "vggsound"):
        for method in ("source", "rebalance", "firm"):
            means = {}
            for modality in ("none", "video", "audio"):
                values = grouped[(dataset, method, modality)]
                means[modality] = (
                    sum(values) / len(values) if values else float("nan")
                )
            rows.append({
                "dataset": dataset,
                "method": method,
                "clean": means["none"],
                "video_15": means["video"],
                "audio_6": means["audio"],
                "corruption_21": (
                    15 * means["video"] + 6 * means["audio"]
                ) / 21,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output}")
    for row in rows:
        print(
            f"{row['dataset']:8s} {row['method']:9s} "
            f"clean={row['clean']:.3f} video={row['video_15']:.3f} "
            f"audio={row['audio_6']:.3f} corr21={row['corruption_21']:.3f}"
        )


if __name__ == "__main__":
    main()

