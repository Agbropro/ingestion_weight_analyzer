import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


CONFIG_PATH = "config/config.yaml"

IMAGE_PATTERN = re.compile(
    r"^nvr(?P<nvr>\d+)_ch(?P<camera>\d+)_.*\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)


def load_config(path: str) -> dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_image(path: Path) -> tuple[str, str] | None:
    """Parse NVR and camera from image filename."""
    match = IMAGE_PATTERN.match(path.name)

    if not match:
        return None

    nvr = f"nvr{int(match.group('nvr'))}"
    camera = f"ch{int(match.group('camera'))}"

    return nvr, camera


def scan_images(
    base_dir: Path,
) -> dict[tuple[str, str], list[Path]]:
    """Group images by NVR and camera."""
    images: dict[
        tuple[str, str],
        list[Path],
    ] = defaultdict(list)

    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue

        parsed = parse_image(path)

        if parsed is None:
            continue

        images[parsed].append(path)

    return images


def load_weights(
    path: Path,
) -> dict[tuple[str, str], float]:
    """Load normalized camera weights."""
    data = pd.read_csv(path)

    if "Weight" in data.columns:
        weight_column = "Weight"

    elif "Weight_Percentage" in data.columns:
        weight_column = "Weight_Percentage"

        data[weight_column] = (
            data[weight_column] / 100
        )

    else:
        raise ValueError(
            "CSV must contain Weight "
            "or Weight_Percentage."
        )

    weights: dict[tuple[str, str], float] = {}

    for _, row in data.iterrows():
        nvr = str(row["NVR"]).lower().strip()
        camera = str(row["Camera"]).lower().strip()

        if not nvr.startswith("nvr"):
            nvr = f"nvr{nvr}"

        if not camera.startswith("ch"):
            camera = f"ch{camera}"

        weights[(nvr, camera)] = float(
            row[weight_column]
        )

    return weights


def allocate_samples(
    weights: dict[tuple[str, str], float],
    images: dict[tuple[str, str], list[Path]],
    sample_size: int,
) -> dict[tuple[str, str], int]:
    """Allocate exact sample count using weights."""
    allocation = {
        key: 0
        for key in weights
    }

    capacity = {
        key: len(images.get(key, []))
        for key in weights
    }

    available_total = sum(capacity.values())

    if available_total < sample_size:
        raise ValueError(
            f"Only {available_total} weighted images "
            f"are available, but {sample_size} requested."
        )

    remaining = sample_size

    while remaining > 0:
        active = [
            key
            for key, weight in weights.items()
            if weight > 0
            and allocation[key] < capacity[key]
        ]

        if not active:
            raise RuntimeError(
                f"Unable to allocate remaining "
                f"{remaining} images."
            )

        total_weight = sum(
            weights[key]
            for key in active
        )

        raw = {
            key: (
                remaining
                * weights[key]
                / total_weight
            )
            for key in active
        }

        added = 0

        # First allocate integer portions.
        for key in active:
            free_capacity = (
                capacity[key]
                - allocation[key]
            )

            amount = min(
                int(raw[key]),
                free_capacity,
            )

            allocation[key] += amount
            added += amount

        remaining -= added

        if remaining == 0:
            break

        # Largest remainder method.
        ranked = sorted(
            active,
            key=lambda key: raw[key] - int(raw[key]),
            reverse=True,
        )

        distributed = 0

        for key in ranked:
            if remaining == 0:
                break

            if allocation[key] >= capacity[key]:
                continue

            allocation[key] += 1

            remaining -= 1
            distributed += 1

        if added == 0 and distributed == 0:
            raise RuntimeError(
                "Unable to distribute remaining samples."
            )

    return allocation


def sample_images(
    images: dict[tuple[str, str], list[Path]],
    allocation: dict[tuple[str, str], int],
    seed: int,
) -> list[
    tuple[str, str, Path]
]:
    """Sample images without replacement."""
    rng = random.Random(seed)

    selected = []

    for key, amount in allocation.items():
        if amount <= 0:
            continue

        candidates = images.get(key, [])

        chosen = rng.sample(
            candidates,
            amount,
        )

        nvr, camera = key

        for path in chosen:
            selected.append(
                (
                    nvr,
                    camera,
                    path,
                )
            )

    rng.shuffle(selected)

    return selected


def copy_images(
    selected: list[tuple[str, str, Path]],
    base_dir: Path,
    output_dir: Path,
) -> None:
    """Copy sampled images preserving structure."""
    for _, _, source in selected:
        relative = source.relative_to(base_dir)

        destination = (
            output_dir
            / relative
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )


def make_report(
    images: dict[tuple[str, str], list[Path]],
    weights: dict[tuple[str, str], float],
    allocation: dict[tuple[str, str], int],
) -> pd.DataFrame:
    """Create sampling summary."""
    rows = []

    total_weight = sum(
        weight
        for key, weight in weights.items()
        if len(images.get(key, [])) > 0
    )

    for key, weight in weights.items():
        nvr, camera = key

        available = len(
            images.get(key, [])
        )

        sampled = allocation.get(
            key,
            0,
        )

        normalized_weight = (
            weight / total_weight
            if total_weight > 0
            else 0
        )

        rows.append(
            {
                "NVR": nvr,
                "Camera": camera,
                "Available": available,
                "Weight": weight,
                "Weight_Percentage": (
                    normalized_weight * 100
                ),
                "Sampled": sampled,
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        by="Sampled",
        ascending=False,
    )


def run_sampler(config: dict[str, Any]) -> None:
    """Run weighted camera sampling."""
    sampler_config = config["sampler"]

    base_dir = Path(sampler_config["input_dir"]).resolve()
    output_dir = Path(sampler_config["output_dir"]).resolve()
    weight_file = Path(sampler_config["weight_file"])
    output_report = Path(sampler_config["output_report"])
    sample_size = int(sampler_config.get("sample_size", 3000))
    seed = int(sampler_config.get("seed", 42))

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {base_dir}"
        )

    print("Scanning dataset...")

    images = scan_images(
        base_dir
    )

    print(
        f"Found {sum(map(len, images.values())):,} images"
    )

    print(
        f"Found {len(images)} NVR/camera groups"
    )

    weights = load_weights(
        weight_file
    )

    print(
        f"Loaded {len(weights)} camera weights"
    )

    allocation = allocate_samples(
        weights=weights,
        images=images,
        sample_size=sample_size,
    )

    selected = sample_images(
        images=images,
        allocation=allocation,
        seed=seed,
    )

    if len(selected) != sample_size:
        raise RuntimeError(
            f"Expected {sample_size} images, "
            f"got {len(selected)}."
        )

    copy_images(
        selected=selected,
        base_dir=base_dir,
        output_dir=output_dir,
    )

    report = make_report(
        images=images,
        weights=weights,
        allocation=allocation,
    )

    report_path = Path(f"{output_report}_sample_report.csv")
    manifest_path = Path(f"{output_report}_sample_manifest.csv")

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        report_path,
        index=False,
    )

    manifest = pd.DataFrame(
        [
            {
                "NVR": nvr,
                "Camera": camera,
                "Source": str(path),
                "Relative_Path": str(
                    path.relative_to(base_dir)
                ),
            }
            for nvr, camera, path in selected
        ]
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    print()
    print(report.to_string(index=False))

    print()
    print(
        f"Total sampled: "
        f"{report['Sampled'].sum():,}"
    )

    print(
        f"Output: {output_dir}"
    )


def main() -> None:
    """Load configuration and run sampling."""
    config = load_config(CONFIG_PATH)
    run_sampler(config)


if __name__ == "__main__":
    main()