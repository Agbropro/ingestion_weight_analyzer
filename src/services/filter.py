import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def load_config(path: str) -> dict[str, Any]:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logger() -> logging.Logger:
    """Create application logger."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "[%(asctime)s] "
            "[%(levelname)s] "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logging.getLogger("prefilter")


def get_images(root: Path) -> list[Path]:
    """Find all dataset images."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_names(model: YOLO) -> dict[int, str]:
    """Get model class names."""
    names = model.names

    if isinstance(names, dict):
        return {
            int(class_id): str(name)
            for class_id, name in names.items()
        }

    return {
        index: str(name)
        for index, name in enumerate(names)
    }


def get_classes(
    model: YOLO,
    requested: list[str],
) -> tuple[list[int], dict[int, str]]:
    """Resolve class names into model class IDs."""
    names = get_names(model)

    name_to_id = {
        name.lower(): class_id
        for class_id, name in names.items()
    }

    invalid = [
        name
        for name in requested
        if name.lower() not in name_to_id
    ]

    if invalid:
        available = ", ".join(names.values())

        raise ValueError(
            f"Unknown classes: {invalid}. "
            f"Available classes: {available}"
        )

    class_ids = [
        name_to_id[name.lower()]
        for name in requested
    ]

    return class_ids, names


def count_objects(
    result: Any,
    names: dict[int, str],
) -> Counter[str]:
    """Count detected objects by class."""
    counts: Counter[str] = Counter()

    if result.boxes is None:
        return counts

    if result.boxes.cls is None:
        return counts

    class_ids = (
        result.boxes.cls
        .detach()
        .cpu()
        .tolist()
    )

    for class_id in class_ids:
        name = names[int(class_id)]
        counts[name] += 1

    return counts


def copy_image(
    source: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """Copy image while preserving directory structure."""
    relative_path = source.relative_to(input_dir)

    destination = output_dir / relative_path

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    return destination


def make_batches(
    images: list[Path],
    batch_size: int,
) -> list[list[Path]]:
    """Split image paths into batches."""
    return [
        images[index:index + batch_size]
        for index in range(
            0,
            len(images),
            batch_size,
        )
    ]


def run_filter(config: dict[str, Any]) -> None:
    """Run YOLO dataset prefiltering."""
    logger = logging.getLogger("prefilter")

    model_config = config["model"]
    dataset_config = config["dataset"]
    inference_config = config["inference"]

    model_path = model_config["path"]

    confidence = float(
        model_config["confidence"]
    )

    iou = float(
        model_config["iou"]
    )

    minimum_count = int(
        model_config["minimum_count"]
    )

    requested_classes = [
        str(name)
        for name in model_config["class"]
    ]

    input_dir = Path(
        dataset_config["input_dir"]
    )

    output_dir = Path(
        dataset_config["output_dir"]
    )

    device = inference_config.get(
        "device",
        0,
    )

    batch_size = int(
        inference_config.get(
            "batch_size",
            16,
        )
    )

    manifest_path = Path(
        config["output"]["manifest"]
    )

    logger.info("Loading model: %s", model_path)

    model = YOLO(model_path)

    class_ids, names = get_classes(
        model=model,
        requested=requested_classes,
    )

    logger.info(
        "Filter classes: %s",
        ", ".join(requested_classes),
    )

    logger.info(
        "Minimum object count: %d",
        minimum_count,
    )

    logger.info(
        "Confidence: %.2f",
        confidence,
    )

    logger.info(
        "IoU: %.2f",
        iou,
    )

    images = get_images(input_dir)

    logger.info(
        "Found %,d images",
        len(images),
    )

    batches = make_batches(
        images=images,
        batch_size=batch_size,
    )

    records: list[dict[str, Any]] = []

    total_passed = 0
    total_failed = 0

    for batch_index, batch in enumerate(
        batches,
        start=1,
    ):
        sources = [
            str(path)
            for path in batch
        ]

        results = model.predict(
            source=sources,
            conf=confidence,
            iou=iou,
            classes=class_ids,
            device=device,
            verbose=False,
        )

        for image_path, result in zip(
            batch,
            results,
        ):
            counts = count_objects(
                result=result,
                names=names,
            )

            total_objects = sum(
                counts.values()
            )

            passed = (
                total_objects
                >= minimum_count
            )

            destination = None

            if passed:
                destination = copy_image(
                    source=image_path,
                    input_dir=input_dir,
                    output_dir=output_dir,
                )

                total_passed += 1

            else:
                total_failed += 1

            record = {
                "source": str(image_path),
                "passed": passed,
                "object_count": total_objects,
            }

            for class_name in requested_classes:
                record[
                    f"count_{class_name}"
                ] = counts.get(
                    class_name,
                    0,
                )

            if destination is not None:
                record["destination"] = str(
                    destination
                )
            else:
                record["destination"] = ""

            records.append(record)

        logger.info(
            "Batch %d/%d | processed=%d | passed=%d | rejected=%d",
            batch_index,
            len(batches),
            len(records),
            total_passed,
            total_failed,
        )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.DataFrame(records)

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    logger.info(
        "Completed | total=%d | passed=%d | rejected=%d",
        len(images),
        total_passed,
        total_failed,
    )

    logger.info(
        "Output directory: %s",
        output_dir,
    )

    logger.info(
        "Manifest: %s",
        manifest_path,
    )


def main() -> None:
    """Run application."""
    setup_logger()

    config = load_config(
        "config/config.yaml"
    )

    run_filter(config)


if __name__ == "__main__":
    main()