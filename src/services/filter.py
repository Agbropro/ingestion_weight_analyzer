import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, UnidentifiedImageError
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


def check_image(path: Path) -> tuple[bool, str]:
    """Check whether an image can be decoded."""
    try:
        with Image.open(path) as image:
            image.verify()

        return True, ""

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        return False, str(error)


def load_processed(
    manifest_path: Path,
) -> tuple[set[str], int, int, int]:
    """Load previously processed image paths."""
    if not manifest_path.exists():
        return set(), 0, 0, 0

    try:
        manifest = pd.read_csv(
            manifest_path,
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return set(), 0, 0, 0

    if "source" not in manifest.columns:
        return set(), 0, 0, 0

    processed = set(
        manifest["source"]
        .dropna()
        .astype(str)
    )

    if "status" not in manifest.columns:
        return processed, 0, 0, 0

    status = (
        manifest["status"]
        .fillna("")
        .astype(str)
    )

    passed = int(
        (status == "passed").sum()
    )

    rejected = int(
        (status == "rejected").sum()
    )

    invalid = int(
        (status == "invalid").sum()
    )

    return (
        processed,
        passed,
        rejected,
        invalid,
    )


def save_records(
    records: list[dict[str, Any]],
    manifest_path: Path,
) -> None:
    """Append batch records to manifest."""
    if not records:
        return

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    has_data = (
        manifest_path.exists()
        and manifest_path.stat().st_size > 0
    )

    pd.DataFrame(records).to_csv(
        manifest_path,
        mode="a",
        header=not has_data,
        index=False,
    )


def make_record(
    image_path: Path,
    requested_classes: list[str],
    counts: Counter[str] | None = None,
    passed: bool = False,
    destination: Path | None = None,
    status: str = "rejected",
    error: str = "",
) -> dict[str, Any]:
    """Create manifest record."""
    counts = counts or Counter()

    record: dict[str, Any] = {
        "source": str(image_path),
        "status": status,
        "passed": passed,
        "object_count": sum(counts.values()),
        "destination": (
            str(destination)
            if destination is not None
            else ""
        ),
        "error": error,
    }

    for class_name in requested_classes:
        record[f"count_{class_name}"] = counts.get(
            class_name,
            0,
        )

    return record


def process_result(
    image_path: Path,
    result: Any,
    names: dict[int, str],
    requested_classes: list[str],
    minimum_count: int,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Process one inference result."""
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
    status = "rejected"

    if passed:
        destination = copy_image(
            source=image_path,
            input_dir=input_dir,
            output_dir=output_dir,
        )

        status = "passed"

    return make_record(
        image_path=image_path,
        requested_classes=requested_classes,
        counts=counts,
        passed=passed,
        destination=destination,
        status=status,
    )


def predict_single(
    model: YOLO,
    image_path: Path,
    confidence: float,
    iou: float,
    class_ids: list[int],
    device: Any,
) -> Any:
    """Run YOLO inference on one image."""
    results = model.predict(
        source=str(image_path),
        conf=confidence,
        iou=iou,
        classes=class_ids,
        device=device,
        verbose=False,
    )

    if not results:
        raise RuntimeError(
            f"No YOLO result returned for {image_path}"
        )

    return results[0]


def process_batch(
    model: YOLO,
    batch: list[Path],
    confidence: float,
    iou: float,
    class_ids: list[int],
    device: Any,
    names: dict[int, str],
    requested_classes: list[str],
    minimum_count: int,
    input_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """Process batch with unreadable-image fallback."""
    sources = [
        str(path)
        for path in batch
    ]

    try:
        results = model.predict(
            source=sources,
            conf=confidence,
            iou=iou,
            classes=class_ids,
            device=device,
            verbose=False,
        )

        if len(results) != len(batch):
            raise RuntimeError(
                "YOLO returned a different result count "
                f"than input count: {len(results)} "
                f"!= {len(batch)}"
            )

        return [
            process_result(
                image_path=image_path,
                result=result,
                names=names,
                requested_classes=requested_classes,
                minimum_count=minimum_count,
                input_dir=input_dir,
                output_dir=output_dir,
            )
            for image_path, result in zip(
                batch,
                results,
            )
        ]

    except (
        UnidentifiedImageError,
        OSError,
    ) as error:
        logger.warning(
            "Batch contains unreadable image. "
            "Falling back to individual processing | %s",
            error,
        )

    records: list[dict[str, Any]] = []

    for image_path in batch:
        valid, error = check_image(
            image_path
        )

        if not valid:
            logger.warning(
                "Skipping invalid image: %s | %s",
                image_path,
                error,
            )

            records.append(
                make_record(
                    image_path=image_path,
                    requested_classes=requested_classes,
                    status="invalid",
                    error=error,
                )
            )

            continue

        try:
            result = predict_single(
                model=model,
                image_path=image_path,
                confidence=confidence,
                iou=iou,
                class_ids=class_ids,
                device=device,
            )

        except (
            UnidentifiedImageError,
            OSError,
        ) as error:
            logger.warning(
                "Skipping unreadable image: %s | %s",
                image_path,
                error,
            )

            records.append(
                make_record(
                    image_path=image_path,
                    requested_classes=requested_classes,
                    status="invalid",
                    error=str(error),
                )
            )

            continue

        except Exception:
            logger.exception(
                "YOLO inference failed on valid image: %s",
                image_path,
            )

            raise

        record = process_result(
            image_path=image_path,
            result=result,
            names=names,
            requested_classes=requested_classes,
            minimum_count=minimum_count,
            input_dir=input_dir,
            output_dir=output_dir,
        )

        records.append(record)

    return records


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
    ).resolve()

    output_dir = Path(
        dataset_config["output_dir"]
    ).resolve()

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

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    logger.info(
        "Loading model: %s",
        model_path,
    )

    model = YOLO(
        model_path
    )

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

    logger.info(
        "Device: %s",
        device,
    )

    logger.info(
        "Batch size: %d",
        batch_size,
    )

    images = get_images(
        input_dir
    )

    logger.info(
        "Found %s images",
        f"{len(images):,}",
    )

    (
        processed_paths,
        total_passed,
        total_rejected,
        total_invalid,
    ) = load_processed(
        manifest_path
    )

    if processed_paths:
        logger.info(
            "Resume manifest found | "
            "processed=%s | "
            "passed=%s | "
            "rejected=%s | "
            "invalid=%s",
            f"{len(processed_paths):,}",
            f"{total_passed:,}",
            f"{total_rejected:,}",
            f"{total_invalid:,}",
        )

    remaining_images = [
        path
        for path in images
        if str(path) not in processed_paths
    ]

    skipped = (
        len(images)
        - len(remaining_images)
    )

    logger.info(
        "Resume status | "
        "total=%s | "
        "skipped=%s | "
        "remaining=%s",
        f"{len(images):,}",
        f"{skipped:,}",
        f"{len(remaining_images):,}",
    )

    if not remaining_images:
        logger.info(
            "Nothing to process. Dataset already completed."
        )
        return

    batches = make_batches(
        images=remaining_images,
        batch_size=batch_size,
    )

    session_processed = 0

    for batch_index, batch in enumerate(
        batches,
        start=1,
    ):
        batch_records = process_batch(
            model=model,
            batch=batch,
            confidence=confidence,
            iou=iou,
            class_ids=class_ids,
            device=device,
            names=names,
            requested_classes=requested_classes,
            minimum_count=minimum_count,
            input_dir=input_dir,
            output_dir=output_dir,
            logger=logger,
        )

        save_records(
            records=batch_records,
            manifest_path=manifest_path,
        )

        session_processed += len(
            batch_records
        )

        batch_passed = sum(
            record["status"] == "passed"
            for record in batch_records
        )

        batch_rejected = sum(
            record["status"] == "rejected"
            for record in batch_records
        )

        batch_invalid = sum(
            record["status"] == "invalid"
            for record in batch_records
        )

        total_passed += batch_passed
        total_rejected += batch_rejected
        total_invalid += batch_invalid

        total_processed = (
            skipped
            + session_processed
        )

        logger.info(
            "Batch %d/%d | "
            "processed=%s/%s | "
            "passed=%s | "
            "rejected=%s | "
            "invalid=%s",
            batch_index,
            len(batches),
            f"{total_processed:,}",
            f"{len(images):,}",
            f"{total_passed:,}",
            f"{total_rejected:,}",
            f"{total_invalid:,}",
        )

    logger.info(
        "Completed | "
        "total=%s | "
        "passed=%s | "
        "rejected=%s | "
        "invalid=%s",
        f"{len(images):,}",
        f"{total_passed:,}",
        f"{total_rejected:,}",
        f"{total_invalid:,}",
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

    run_filter(
        config
    )


if __name__ == "__main__":
    main()