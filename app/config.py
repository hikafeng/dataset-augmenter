from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_ENV_FILE = ".env"


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return default
    return int(value)


def _parse_float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    return float(value)


def _parse_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or not value.strip():
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class Settings:
    dataset_path: Path
    dataset_type: str
    num_aug: int
    output_root: Path | None
    save_labeled: bool
    resize_width: int
    resize_height: int
    transforms: list[str]
    horizontal_flip_p: float
    vertical_flip_p: float
    rotate_p: float
    rotate_limit: int


def load_settings(env_file: str | os.PathLike[str] | None = None) -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    resolved_env_file = Path(env_file) if env_file else project_root / DEFAULT_ENV_FILE
    if resolved_env_file.exists():
        load_dotenv(resolved_env_file)

    dataset_path = os.getenv(
        "AUG_DATASET_PATH",
        "dataset/detection/fiber-splitter-1k-yolo",
    )
    output_root = os.getenv("AUG_OUTPUT_ROOT")

    return Settings(
        dataset_path=(project_root / dataset_path).resolve()
        if not Path(dataset_path).is_absolute()
        else Path(dataset_path),
        dataset_type=os.getenv("AUG_DATASET_TYPE", "auto").strip().lower(),
        num_aug=_parse_int(os.getenv("AUG_NUM_AUG"), 3),
        output_root=(project_root / output_root).resolve()
        if output_root and not Path(output_root).is_absolute()
        else (Path(output_root) if output_root else None),
        save_labeled=_parse_bool(os.getenv("AUG_SAVE_LABELED"), False),
        resize_width=_parse_int(os.getenv("AUG_RESIZE_WIDTH"), 640),
        resize_height=_parse_int(os.getenv("AUG_RESIZE_HEIGHT"), 640),
        transforms=_parse_list(
            os.getenv("AUG_TRANSFORMS"),
            ["resize", "rotate", "hflip", "vflip"],
        ),
        horizontal_flip_p=_parse_float(os.getenv("AUG_HORIZONTAL_FLIP_P"), 0.5),
        vertical_flip_p=_parse_float(os.getenv("AUG_VERTICAL_FLIP_P"), 0.5),
        rotate_p=_parse_float(os.getenv("AUG_ROTATE_P"), 0.5),
        rotate_limit=_parse_int(os.getenv("AUG_ROTATE_LIMIT"), 90),
    )