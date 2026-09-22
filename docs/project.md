# Ingestion Weight Analyzer

## Project Definition

This project helps balance a CCTV image dataset by computing per-camera **ingestion weights** and using those weights to sample new images for a final annotated dataset. The pipeline is used in a person-detection / CCTV context where each NVR (network video recorder) and camera channel contributes a different number of images to the existing pool.

The system ingests a CSV that records how many images each `(NVR, Camera)` pair already has, calculates how under-represented each pair is relative to the total desired dataset size, and outputs a weighted allocation for additional images. It also includes tools to:

1. **Filter** raw CCTV frames using a YOLO model (keeping only frames with enough target objects, e.g. `person`).
2. **Sample** images from the raw pool according to the computed weights.

## Goals

- **Balance the dataset**: ensure each `(NVR, Camera)` pair is represented proportionally in the final combined dataset.
- **Compute ingestion weights**: derive a weight for every camera pair based on its current deficit compared to a target per-camera count.
- **Allocate new images**: distribute a fixed number of new images (`added_data`) across camera pairs using the weights, resolving remainders fairly.
- **Support quality pre-filtering**: run YOLO object detection to discard frames that do not meet minimum object counts requirements before sampling.
- **Provide reproducible sampling**: weighted random sampling with a fixed seed, producing manifests and reports.

## High-level Pipeline

```
Raw images on disk
        |
        v
  [filter.py]  --(YOLO)-->  Keep frames with >= minimum_count persons
        |
        v
  [main.py]  --(compute weights & allocation)-->  data/result/dataset_1.csv
        |
        v
  [sampler.py]  --(weighted sampling)-->  sampled images + manifest
```

## Key Components

### `src/main.py`

The core ingestion-weight calculator.

- Reads `data/dataset_1.csv`.
- Uses fixed dataset sizes:
  - `existing_data = 6000`
  - `added_data = 3000`
  - `total_data = 9000`
- `target_per_camera = total_data / camera_count`
- Computes `Deficit = max(target_per_camera - current_images, 0)`.
- Normalizes deficits into `Weight` and `Weight_Percentage`.
- Allocates `added_data` by taking the floor of `Weight * added_data`, then distributes any leftover images to the pairs with the largest fractional remainders.
- Outputs `data/result/dataset_1.csv` with columns including `Added_Images`, `Final_Images`, `Current_Percentage`, and `Final_Percentage`.

### `src/services/filter.py`

A YOLO-based pre-filter for raw CCTV frames.

- Configuration: `config/config.yaml`.
- Loads a YOLO model (`models/best.pt`).
- Scans the configured `input_dir` for images.
- Runs inference in batches on GPU/CPU.
- Keeps images where the requested class (default `person`) appears at least `minimum_count` times.
- Copies kept images to `output_dir` and writes a manifest CSV (`prefilter_manifest.csv`) with per-image results.

### `src/services/sampler.py`

Weighted random sampler that selects the actual images to add to the dataset.

- Reads computed weights from `data/result/dataset_1.csv`.
- Scans raw images grouped by `(NVR, Camera)`.
- Allocates `SAMPLE_SIZE = 3000` images proportionally to the weights.
- Samples with a fixed seed (`SEED = 42`).
- Copies selected images to `OUTPUT_DIR` and writes a report and manifest.

## Data Files

| File | Purpose |
|------|---------|
| `data/dataset_1.csv` | Input per-camera image counts and metadata. |
| `data/result/dataset_1.csv` | Output of `main.py`: weights and allocation. |
| `data/prefilter_manifest.csv` | Output of `filter.py`: which images passed/failed filtering. |
| `data/result/*_sample_report.csv` / `*_sample_manifest.csv` | Output of `sampler.py`. |

## Configuration

`config/config.yaml` controls the filter step:

```yaml
model:
  path: "models/best.pt"
  confidence: 0.5
  iou: 0.3
  class:
    - person
  minimum_count: 3

dataset:
  input_dir: "/mnt/secondary/dataset/person_cctv/GW"
  output_dir: "/mnt/secondary/dataset/person_cctv/GW_prefiltered"

inference:
  device: 0
  batch_size: 16

output:
  manifest: "data/prefilter_manifest.csv"
```

## Runtime Requirements

- Python 3
- `pandas`
- `ultralytics` (for `filter.py`)
- `pyyaml` (for `filter.py`)
- A YOLO model file at `models/best.pt` (for `filter.py`)
- Raw image directories named in the pattern `nvr<N>_ch<C>_...` (for `sampler.py`)

## How to Use

1. Ensure `data/dataset_1.csv` exists with the desired per-camera counts.
2. (Optional) Run the pre-filter:
   ```bash
   python -m src.services.filter
   ```
3. Compute weights and allocation:
   ```bash
   python src/main.py
   ```
4. Sample images:
   ```bash
   python -m src.services.sampler
   ```

## Notes for Other Agents

- `src/main.py` contains hard-coded dataset sizes (`existing_data`, `added_data`). These are the canonical project assumptions.
- The sampler relies on filenames matching the regex `^nvr(?P<nvr>\d+)_ch(?P<camera>\d+)_.*\.(jpg|jpeg|png)$`. If raw image naming ever changes, this is the first place to update.
- The filter step is optional for the allocation math but required for the final sampled dataset quality.
