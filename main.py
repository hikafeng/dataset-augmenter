from __future__ import annotations

import argparse
import logging

import tqdm

from app.annotations import DetectionAnnotationCodec, PoseAnnotationCodec
from app.augment import DatasetAugmenter
from app.config import load_settings
from app.pipeline import build_pipeline
from app.utils import build_manifests, build_split_listing, dump_yaml, load_yaml, timestamped_output_dir


LOGGER = logging.getLogger("dataset_augmenter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment detection and pose datasets")
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to the .env file used to configure the augmentation job",
    )
    return parser.parse_args()


def infer_dataset_type(requested_type: str, dataset_meta: dict) -> str:
    normalized_type = requested_type.lower()
    if normalized_type in {"detection", "pose"}:
        return normalized_type
    if normalized_type != "auto":
        raise ValueError(f"不支持的数据集类型: {requested_type}")
    return "pose" if "kpt_shape" in dataset_meta else "detection"


def write_dataset_metadata(output_dir, dataset_meta: dict, manifests) -> None:
    output_meta = dict(dataset_meta)
    output_meta["path"] = str(output_dir)
    for manifest in manifests:
        output_meta[manifest.key] = manifest.list_name
    dump_yaml(output_dir / "data.yaml", output_meta)


def run(env_file: str | None = None):
    settings = load_settings(env_file)
    dataset_meta = load_yaml(settings.dataset_path / "data.yaml")
    manifests = build_manifests(settings.dataset_path, dataset_meta)
    if not manifests:
        raise ValueError(f"未找到任何可处理的数据切分: {settings.dataset_path}")

    dataset_type = infer_dataset_type(settings.dataset_type, dataset_meta)
    codec = PoseAnnotationCodec() if dataset_type == "pose" else DetectionAnnotationCodec()
    pipeline = build_pipeline(settings, with_bboxes=codec.uses_bboxes)
    
    augmenter = DatasetAugmenter(
        codec=codec,
        pipeline=pipeline,
        num_aug=settings.num_aug,
        save_labeled=settings.save_labeled,
    )

    output_dir = timestamped_output_dir(settings.dataset_path, settings.output_root)
    write_dataset_metadata(output_dir, dataset_meta, manifests)

    for manifest in manifests:
        split_file_path = output_dir / manifest.list_name
        split_file_path.parent.mkdir(parents=True, exist_ok=True)
        split_file_path.write_text(
            "\n".join(build_split_listing(manifest.image_paths, settings.num_aug)) + "\n",
            encoding="utf-8",
        )

        for image_path in tqdm.tqdm(
            manifest.image_paths,
            desc=f"Augmenting {manifest.key}",
            leave=False,
        ):
            augmenter.augment_image(settings.dataset_path, image_path, output_dir)

    LOGGER.info("数据增强完成，输出目录: %s", output_dir)
    return output_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = parse_args()
    run(arguments.env_file)