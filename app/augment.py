from __future__ import annotations

from pathlib import Path

import cv2

from app.utils import (
    build_augmented_image_path,
    build_labeled_image_path,
    image_to_label_path,
    save_image_uint8,
)


class DatasetAugmenter:
    def __init__(self, codec, pipeline, num_aug: int, save_labeled: bool) -> None:
        self.codec = codec
        self.pipeline = pipeline
        self.num_aug = num_aug
        self.save_labeled = save_labeled

    def augment_image(self, dataset_root: Path, image_path: Path, output_root: Path) -> None:
        source_image_path = dataset_root / image_path
        source_label_path = dataset_root / image_to_label_path(image_path)

        image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {source_image_path}")
        if not source_label_path.exists():
            raise FileNotFoundError(f"缺少标签文件: {source_label_path}")

        annotations = self.codec.parse(
            source_label_path.read_text(encoding="utf-8").splitlines(),
            image.shape[:2],
        )
        last_transformed_image = None
        last_annotations = None
        for aug_index in range(1, self.num_aug + 1):
            transform_input = self.codec.build_transform_input(annotations)
            transformed = self.pipeline(image=image, **transform_input)
            transformed_image = transformed["image"]
            rebuilt_annotations = self.codec.rebuild(
                transformed,
                transformed_image.shape[:2],
                annotations,
            )

            output_image_rel = build_augmented_image_path(image_path, aug_index)
            output_label_rel = image_to_label_path(output_image_rel)
            output_image_path = output_root / output_image_rel
            output_label_path = output_root / output_label_rel

            label_lines = self.codec.serialize(rebuilt_annotations, transformed_image.shape[:2])
            output_label_path.parent.mkdir(parents=True, exist_ok=True)
            output_label_path.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
            save_image_uint8(output_image_path, transformed_image)

            last_transformed_image = transformed_image
            last_annotations = rebuilt_annotations

        if self.save_labeled and last_transformed_image is not None and last_annotations is not None:
            source_labeled_path = output_root / build_labeled_image_path(image_path, "source")
            aug_labeled_path = output_root / build_labeled_image_path(image_path, "aug")
            save_image_uint8(source_labeled_path, self.codec.draw_annotations(image, annotations))
            save_image_uint8(
                aug_labeled_path,
                self.codec.draw_annotations(last_transformed_image, last_annotations),
            )