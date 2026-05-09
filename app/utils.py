from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class SplitManifest:
    key: str
    list_name: str
    image_paths: list[Path]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_image_uint8(path: Path, image) -> None:
    ensure_parent(path)
    cv2.imwrite(str(path), image)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def dump_yaml(path: Path, data: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def timestamped_output_dir(dataset_path: Path, output_root: Path | None) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = output_root if output_root else dataset_path.parent
    return base_dir / f"{dataset_path.name}-{timestamp}"


def extract_relative_image_path(entry: str) -> Path | None:
    cleaned = entry.strip().replace("\\", "/")
    if not cleaned:
        return None

    cleaned = cleaned.removeprefix("./")
    marker = "/images/"
    if marker in cleaned:
        suffix = cleaned.split(marker, 1)[1]
        return Path("images") / Path(suffix)
    if cleaned.startswith("images/"):
        return Path(cleaned)
    return Path(cleaned)


def image_to_label_path(image_path: Path) -> Path:
    if not image_path.parts or image_path.parts[0] != "images":
        raise ValueError(f"非法图片相对路径: {image_path}")
    return Path("labels").joinpath(*image_path.parts[1:]).with_suffix(".txt")


def build_augmented_image_path(image_path: Path, aug_index: int) -> Path:
    return image_path.with_name(f"{image_path.stem}_{aug_index:06d}{image_path.suffix}")


def build_labeled_image_path(image_path: Path, suffix: str) -> Path:
    image_subdir = Path(*image_path.parts[1:-1]) if len(image_path.parts) > 2 else Path()
    return Path("images") / "labeled" / image_subdir / f"{image_path.stem}_{suffix}{image_path.suffix}"


def build_split_listing(image_paths: list[Path], num_aug: int) -> list[str]:
    lines: list[str] = []
    for image_path in image_paths:
        for aug_index in range(1, num_aug + 1):
            lines.append(build_augmented_image_path(image_path, aug_index).as_posix())
    return lines


def build_manifests(dataset_root: Path, dataset_meta: dict) -> list[SplitManifest]:
    manifests: list[SplitManifest] = []
    for split_key in ("train", "val", "test"):
        split_entry = dataset_meta.get(split_key)
        list_name = Path(str(split_entry)).name if split_entry else f"{split_key}.txt"
        image_paths = _load_split_images(dataset_root, list_name)
        if not image_paths:
            image_paths = _discover_split_images(dataset_root, split_key)
        if image_paths:
            manifests.append(SplitManifest(key=split_key, list_name=list_name, image_paths=image_paths))
    return manifests


def _load_split_images(dataset_root: Path, list_name: str) -> list[Path]:
    list_path = dataset_root / list_name
    if not list_path.exists():
        return []

    results: list[Path] = []
    seen: set[Path] = set()
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        relative_path = extract_relative_image_path(raw_line)
        if relative_path is None:
            continue
        absolute_path = dataset_root / relative_path
        if (
            absolute_path.exists()
            and absolute_path.suffix.lower() in IMAGE_EXTENSIONS
            and relative_path not in seen
        ):
            results.append(relative_path)
            seen.add(relative_path)
    return results


def _discover_split_images(dataset_root: Path, split_key: str) -> list[Path]:
    images_root = dataset_root / "images"
    if not images_root.exists():
        return []

    results: list[Path] = []
    for image_path in images_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = image_path.relative_to(dataset_root)
        parts_lower = {part.lower() for part in relative_path.parts}
        if split_key in parts_lower:
            results.append(relative_path)

    return sorted(results)