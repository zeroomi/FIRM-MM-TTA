# FIRM-MM-TTA

Code for **FIRM: Frozen Inference with Rebalanced Marginals for
Backpropagation-Free Multimodal Test-Time Adaptation**.

FIRM keeps the source audio-visual network fixed at test time. Adaptation is
performed on its output probabilities and features, without gradients or
optimizer updates. The current implementation uses CAV-MAE as the source
model and follows the Kinetics50-C and VGGSound-C protocol used by READ and
AdaPGC.

![FIRM method](assets/method_overview.png)

## Method

For each sample, the frozen model returns a source posterior and audio,
video, and fused features. FIRM maintains one causal decision state with two
parts:

- Marginal rebalancing projects the observed posterior prefix toward a
  reference class prior. A drift gate interpolates between the source and
  projected posteriors.
- Prototype refinement compares the current features with class prototypes
  accumulated from earlier samples. The prediction is produced before the
  current sample is written to the prototype state.

The final prediction is a product of the rebalanced posterior and the three
prototype posteriors. The network parameters are unchanged throughout the
stream.

## Results

The following numbers use the same locally generated corruption streams,
checkpoints, sample order, batch size, workers, and RTX 4090 for Source,
AdaPGC, and FIRM. Accuracy is averaged equally over 15 video corruptions or
6 audio corruptions at severity 5.

| Dataset | Corrupted modality | Source | AdaPGC | FIRM |
| --- | --- | ---: | ---: | ---: |
| Kinetics50-C | Video | 60.487 | **66.853** | 66.832 |
| Kinetics50-C | Audio | 69.248 | 73.054 | **73.432** |
| VGGSound-C | Video | 56.035 | 57.295 | **58.296** |
| VGGSound-C | Audio | 25.057 | 37.283 | **40.118** |
| Four-group mean |  | 52.707 | 58.621 | **59.670** |

End-to-end time includes data loading, model inference, and adaptation over
all streams in each group. Checkpoint construction is excluded. Peak memory
is measured with `torch.cuda.max_memory_allocated`.

| Dataset | Modality | Method | Time (min) | Peak memory (GiB) |
| --- | --- | --- | ---: | ---: |
| Kinetics50-C | Video | AdaPGC | 17.68 | 10.36 |
|  |  | FIRM | 4.46 | 0.99 |
| Kinetics50-C | Audio | AdaPGC | 7.01 | 8.60 |
|  |  | FIRM | 2.57 | 0.99 |
| VGGSound-C | Video | AdaPGC | 289.77 | 20.11 |
|  |  | FIRM | 49.81 | 1.01 |
| VGGSound-C | Audio | AdaPGC | 104.11 | 20.11 |
|  |  | FIRM | 16.88 | 1.01 |

## Installation

The reported runs used Python 3.12, PyTorch 2.5.1, CUDA 12.4,
torchvision 0.20.1, and torchaudio 2.5.1.

```bash
conda create -n firm-mmtta python=3.12 -y
conda activate firm-mmtta

python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

The same environment can also be created from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate firm-mmtta
```

The decision-state tests run on CPU and do not require a dataset or
checkpoint:

```bash
python -m pytest -q
```

## Data and checkpoints

Kinetics50, VGGSound, the corruption protocol, and the CAV-MAE checkpoints
are available through the
[READ repository](https://github.com/XLearning-SCU/2024-ICLR-READ).
Datasets and weights are not included here.

FIRM reads the same JSON format as READ/AdaPGC:

```json
{
  "data": [
    {
      "video_id": "sample_id",
      "wav": "/path/to/sample_id.wav",
      "video_path": "/path/to/sample_frames",
      "labels": "class_mid"
    }
  ]
}
```

One possible layout is:

```text
data/json/ks50/
  clean/severity_0.json
  video/<corruption>/severity_5.json
  audio/<corruption>/severity_5.json
data/json/vgg/
  clean/severity_0.json
  video/<corruption>/severity_5.json
  audio/<corruption>/severity_5.json
checkpoints/
  cav_mae_ks50.pth
  vgg_65.5.pth
```

Paths stored inside each JSON file must point to the local WAV files and
frame directories. The label mappings used by the loader are under
`configs/labels/`.

## Running FIRM

Example for Kinetics50-C video Gaussian noise:

```bash
python run_firm.py \
  --dataset ks50 \
  --json-root /path/to/json/ks50 \
  --label-csv configs/labels/class_labels_indices_ks50.csv \
  --checkpoint /path/to/cav_mae_ks50.pth \
  --corruption-modality video \
  --corruption gaussian_noise \
  --severity 5 \
  --gpu 0 \
  --batch-size 8 \
  --num-workers 8 \
  --output-dir outputs/ks50_video_gaussian_noise
```

`result.csv` contains Source and full-FIRM accuracy,
along with wall time, CUDA time, throughput, and peak allocated GPU memory.
Ground-truth labels are not passed to FIRM; they are used only to compute the
reported accuracy.

To evaluate both datasets on clean data and all 21 corruption streams:

```bash
JSON_KS50=/path/to/json/ks50 \
JSON_VGG=/path/to/json/vgg \
CHECKPOINT_KS50=/path/to/cav_mae_ks50.pth \
CHECKPOINT_VGG=/path/to/vgg_65.5.pth \
GPU=0 BATCH_SIZE=8 NUM_WORKERS=8 \
bash scripts/run_all21.sh
```

Set `DATASETS=ks50` or `DATASETS=vggsound` to run only one dataset. Existing
groups with a non-empty `result.csv` are skipped. The final summary is written
to `outputs/firm_all21/summary.csv`.

## Experimental protocol

- Samples are processed in their JSON order without shuffling.
- State is reset between clean/corruption streams.
- The main experiments use a uniform reference prior.
- Default parameters are 20 projection iterations, prototype temperature
  0.07, and prototype evidence weight 0.25.
- FIRM stores the posterior prefix directly. This implementation is intended
  for finite benchmark streams and is not constant-memory for an unbounded
  stream.

## Repository layout

```text
firm/method.py          decision-state implementation
models/                 CAV-MAE model and feature extraction
run_firm.py             evaluation entry point
scripts/run_all21.sh    clean and all-corruption launcher
tools/summarize_results.py
configs/labels/         class label mappings
tests/test_firm.py      method-level CPU tests
```

## Citation

The paper reference will be added after publication. For now, citation
metadata is provided in [`CITATION.cff`](CITATION.cff).

## Acknowledgements

This repository includes code derived from
[CAV-MAE](https://github.com/YuanGongND/cav-mae),
[READ](https://github.com/XLearning-SCU/2024-ICLR-READ), and
[AdaPGC](https://github.com/XLearning-SCU/AdaPGC).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
