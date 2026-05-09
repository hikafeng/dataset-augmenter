from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2


Point = tuple[float, float]
Box = tuple[float, float, float, float]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(slots=True)
class DetectionAnnotation:
    class_id: int
    points: list[Point]


@dataclass(slots=True)
class PoseAnnotation:
    class_id: int
    bbox: Box
    keypoints: list[Point]
    visibility: list[int]


class DetectionAnnotationCodec:
    uses_bboxes = False

    def parse(self, lines: list[str], image_shape: tuple[int, int]) -> list[DetectionAnnotation]:
        image_height, image_width = image_shape
        annotations: list[DetectionAnnotation] = []
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue

            tokens = line.split()
            if len(tokens) < 3 or (len(tokens) - 1) % 2 != 0:
                raise ValueError(f"检测标签格式错误，第 {line_number} 行: {raw_line}")

            points: list[Point] = []
            for index in range(1, len(tokens), 2):
                points.append(
                    (
                        float(tokens[index]) * image_width,
                        float(tokens[index + 1]) * image_height,
                    )
                )

            annotations.append(DetectionAnnotation(class_id=int(tokens[0]), points=points))
        return annotations

    def build_transform_input(self, annotations: list[DetectionAnnotation]) -> dict[str, Any]:
        return {
            "keypoints": [point for annotation in annotations for point in annotation.points],
        }

    def rebuild(
        self,
        transformed: dict[str, Any],
        output_shape: tuple[int, int],
        annotations: list[DetectionAnnotation],
    ) -> list[DetectionAnnotation]:
        transformed_points: list[Point] = [
            (float(x), float(y)) for x, y in transformed.get("keypoints", [])
        ]
        rebuilt: list[DetectionAnnotation] = []
        offset = 0

        for annotation in annotations:
            point_count = len(annotation.points)
            point_slice = transformed_points[offset : offset + point_count]
            if len(point_slice) != point_count:
                raise ValueError("增强后的检测关键点数量与原始标签不一致")
            rebuilt.append(
                DetectionAnnotation(class_id=annotation.class_id, points=point_slice)
            )
            offset += point_count

        return rebuilt

    def serialize(
        self,
        annotations: list[DetectionAnnotation],
        image_shape: tuple[int, int],
    ) -> list[str]:
        image_height, image_width = image_shape
        lines: list[str] = []

        for annotation in annotations:
            parts = [str(annotation.class_id)]
            for x, y in annotation.points:
                x_norm = _clamp(x / max(image_width, 1), 0.0, 1.0)
                y_norm = _clamp(y / max(image_height, 1), 0.0, 1.0)
                parts.extend((f"{x_norm:.6f}", f"{y_norm:.6f}"))
            lines.append(" ".join(parts))

        return lines

    def draw_annotations(self, image, annotations: list[DetectionAnnotation]):
        annotated = image.copy()
        for annotation in annotations:
            for x, y in annotation.points:
                cv2.circle(annotated, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
        return annotated


class PoseAnnotationCodec:
    uses_bboxes = False

    def parse(self, lines: list[str], image_shape: tuple[int, int]) -> list[PoseAnnotation]:
        image_height, image_width = image_shape
        annotations: list[PoseAnnotation] = []

        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue

            tokens = line.split()
            if len(tokens) < 8 or (len(tokens) - 5) % 3 != 0:
                raise ValueError(f"Pose 标签格式错误，第 {line_number} 行: {raw_line}")

            class_id = int(tokens[0])
            center_x, center_y, box_width, box_height = map(float, tokens[1:5])
            bbox = self._yolo_to_pascal(
                center_x,
                center_y,
                box_width,
                box_height,
                image_width,
                image_height,
            )

            keypoints: list[Point] = []
            visibility: list[int] = []
            for index in range(5, len(tokens), 3):
                x = float(tokens[index]) * image_width
                y = float(tokens[index + 1]) * image_height
                visible = int(float(tokens[index + 2]))
                keypoints.append((x, y))
                visibility.append(visible)

            annotations.append(
                PoseAnnotation(
                    class_id=class_id,
                    bbox=bbox,
                    keypoints=keypoints,
                    visibility=visibility,
                )
            )

        return annotations

    def build_transform_input(self, annotations: list[PoseAnnotation]) -> dict[str, Any]:
        keypoints: list[Point] = []
        for annotation in annotations:
            keypoints.extend(
                keypoint
                for keypoint, visible in zip(annotation.keypoints, annotation.visibility)
                if visible > 0
            )
            keypoints.extend(self._bbox_to_corners(annotation.bbox))

        return {
            "keypoints": keypoints,
        }

    def rebuild(
        self,
        transformed: dict[str, Any],
        output_shape: tuple[int, int],
        annotations: list[PoseAnnotation],
    ) -> list[PoseAnnotation]:
        output_height, output_width = output_shape
        transformed_keypoints = [
            (float(x), float(y)) for x, y in transformed.get("keypoints", [])
        ]

        rebuilt: list[PoseAnnotation] = []
        offset = 0
        for annotation in annotations:
            transformed_keypoint_count = sum(
                1 for visible in annotation.visibility if visible > 0
            )
            keypoint_slice = transformed_keypoints[
                offset : offset + transformed_keypoint_count
            ]
            bbox_corner_slice = transformed_keypoints[
                offset + transformed_keypoint_count : offset + transformed_keypoint_count + 4
            ]
            if len(keypoint_slice) != transformed_keypoint_count:
                raise ValueError("增强后的 pose 关键点数量与原始标签不一致")
            if len(bbox_corner_slice) != 4:
                raise ValueError("增强后的 pose bbox 角点数量与原始标签不一致")

            keypoints: list[Point] = []
            visibility: list[int] = []
            keypoint_index = 0
            for visible in annotation.visibility:
                if visible <= 0:
                    keypoints.append((0.0, 0.0))
                    visibility.append(0)
                    continue

                x, y = keypoint_slice[keypoint_index]
                keypoint_index += 1
                if (
                    x < 0
                    or y < 0
                    or x > output_width
                    or y > output_height
                ):
                    keypoints.append((0.0, 0.0))
                    visibility.append(0)
                else:
                    keypoints.append((x, y))
                    visibility.append(visible)

            rebuilt.append(
                PoseAnnotation(
                    class_id=annotation.class_id,
                    bbox=self._corners_to_bbox(bbox_corner_slice),
                    keypoints=keypoints,
                    visibility=visibility,
                )
            )
            offset += transformed_keypoint_count + 4

        return rebuilt

    def serialize(
        self,
        annotations: list[PoseAnnotation],
        image_shape: tuple[int, int],
    ) -> list[str]:
        image_height, image_width = image_shape
        lines: list[str] = []

        for annotation in annotations:
            center_x, center_y, box_width, box_height = self._pascal_to_yolo(
                annotation.bbox,
                image_width,
                image_height,
            )
            parts = [
                str(annotation.class_id),
                f"{center_x:.6f}",
                f"{center_y:.6f}",
                f"{box_width:.6f}",
                f"{box_height:.6f}",
            ]

            for (x, y), visible in zip(annotation.keypoints, annotation.visibility):
                if visible <= 0:
                    parts.extend(("0.000000", "0.000000", "0"))
                    continue

                x_norm = _clamp(x / max(image_width, 1), 0.0, 1.0)
                y_norm = _clamp(y / max(image_height, 1), 0.0, 1.0)
                parts.extend((f"{x_norm:.6f}", f"{y_norm:.6f}", str(int(visible))))

            lines.append(" ".join(parts))

        return lines

    def draw_annotations(self, image, annotations: list[PoseAnnotation]):
        annotated = image.copy()
        for annotation in annotations:
            x_min, y_min, x_max, y_max = annotation.bbox
            cv2.rectangle(
                annotated,
                (int(round(x_min)), int(round(y_min))),
                (int(round(x_max)), int(round(y_max))),
                (0, 255, 255),
                1,
            )
            for (x, y), visible in zip(annotation.keypoints, annotation.visibility):
                if visible > 0:
                    cv2.circle(
                        annotated,
                        (int(round(x)), int(round(y))),
                        2,
                        (0, 255, 0),
                        -1,
                    )
        return annotated

    @staticmethod
    def _yolo_to_pascal(
        center_x: float,
        center_y: float,
        box_width: float,
        box_height: float,
        image_width: int,
        image_height: int,
    ) -> Box:
        half_width = box_width * image_width / 2
        half_height = box_height * image_height / 2
        pixel_center_x = center_x * image_width
        pixel_center_y = center_y * image_height
        return (
            pixel_center_x - half_width,
            pixel_center_y - half_height,
            pixel_center_x + half_width,
            pixel_center_y + half_height,
        )

    @staticmethod
    def _pascal_to_yolo(bbox: Box, image_width: int, image_height: int) -> Box:
        x_min, y_min, x_max, y_max = bbox
        x_min, x_max = sorted((x_min, x_max))
        y_min, y_max = sorted((y_min, y_max))
        x_min = _clamp(x_min, 0.0, float(image_width))
        x_max = _clamp(x_max, 0.0, float(image_width))
        y_min = _clamp(y_min, 0.0, float(image_height))
        y_max = _clamp(y_max, 0.0, float(image_height))
        center_x = ((x_min + x_max) / 2) / max(image_width, 1)
        center_y = ((y_min + y_max) / 2) / max(image_height, 1)
        box_width = (x_max - x_min) / max(image_width, 1)
        box_height = (y_max - y_min) / max(image_height, 1)
        return (
            _clamp(center_x, 0.0, 1.0),
            _clamp(center_y, 0.0, 1.0),
            _clamp(box_width, 0.0, 1.0),
            _clamp(box_height, 0.0, 1.0),
        )

    @staticmethod
    def _bbox_to_corners(bbox: Box) -> list[Point]:
        x_min, y_min, x_max, y_max = bbox
        return [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ]

    @staticmethod
    def _corners_to_bbox(corners: list[Point]) -> Box:
        x_values = [x for x, _ in corners]
        y_values = [y for _, y in corners]
        return (min(x_values), min(y_values), max(x_values), max(y_values))