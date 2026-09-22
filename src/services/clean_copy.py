import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = "config/config.yaml"

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


def get_images(base_path: Path) -> list[Path]:
    """Find images recursively below the base path."""
    return sorted(
        path
        for path in base_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_nvr_ip(
    image_path: Path,
    base_path: Path,
    ip_prefix: str,
) -> str | None:
    """Find the NVR IP directory in an image's relative path."""
    relative = image_path.relative_to(base_path)

    for part in relative.parts[:-1]:
        if part.startswith(ip_prefix):
            return part

    return None


def build_destinations(
    images: list[Path],
    base_path: Path,
    destination: Path,
    ip_prefix: str,
) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Build copy targets, disambiguating repeated filenames."""
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    skipped: list[Path] = []

    for image_path in images:
        nvr_ip = find_nvr_ip(
            image_path=image_path,
            base_path=base_path,
            ip_prefix=ip_prefix,
        )

        if nvr_ip is None:
            skipped.append(image_path)
            continue

        grouped[(nvr_ip, image_path.name)].append(image_path)

    copies: list[tuple[Path, Path]] = []

    for (nvr_ip, filename), sources in grouped.items():
        if len(sources) == 1:
            copies.append(
                (sources[0], destination / nvr_ip / filename)
            )
            continue

        for source in sources:
            relative = source.relative_to(base_path)
            name_parts = [
                part
                for part in relative.parts
                if part != nvr_ip
            ]
            unique_name = "__".join(name_parts)
            copies.append(
                (source, destination / nvr_ip / unique_name)
            )

    targets = [target for _, target in copies]

    if len(targets) != len(set(targets)):
        raise ValueError(
            "Multiple source images resolve to the same destination."
        )

    return copies, skipped


def copy_images(copies: list[tuple[Path, Path]]) -> None:
    """Copy images to their prepared destinations."""
    for source, target in copies:
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copy2(source, target)


def run_copier(config: dict[str, Any]) -> None:
    """Copy images into one directory per NVR IP."""
    copier_config = config["copy_by_nvr_ip"]
    base_path = Path(copier_config["base_path"]).resolve()
    destination = Path(copier_config["destination"]).resolve()
    ip_prefix = str(copier_config.get("ip_prefix", "192.168.2."))

    if not base_path.is_dir():
        raise FileNotFoundError(
            f"Base path not found: {base_path}"
        )

    if destination == base_path or destination.is_relative_to(base_path):
        raise ValueError(
            "Destination must be outside the base path."
        )

    images = get_images(base_path)
    copies, skipped = build_destinations(
        images=images,
        base_path=base_path,
        destination=destination,
        ip_prefix=ip_prefix,
    )

    copy_images(copies)

    print(f"Found: {len(images):,} images")
    print(f"Copied: {len(copies):,} images")
    print(f"Skipped without NVR IP: {len(skipped):,} images")
    print(f"Destination: {destination}")


def main() -> None:
    """Load configuration and run the copier."""
    config = load_config(CONFIG_PATH)
    run_copier(config)


if __name__ == "__main__":
    main()
