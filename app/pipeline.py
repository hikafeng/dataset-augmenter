import cv2
import albumentations as A

from app.config import Settings


def build_pipeline(settings: Settings, with_bboxes: bool) -> A.Compose:
    transforms = []
    for transform_name in settings.transforms:
        name = transform_name.strip().lower()
        if name == "resize":
            transforms.append(A.Resize(settings.resize_height, settings.resize_width))
        elif name == "hflip":
            transforms.append(A.HorizontalFlip(p=settings.horizontal_flip_p))
        elif name == "vflip":
            transforms.append(A.VerticalFlip(p=settings.vertical_flip_p))
        elif name == "rotate":
            transforms.append(
                A.SafeRotate(
                    angle_range=(-settings.rotate_limit, settings.rotate_limit),
                    border_mode=cv2.BORDER_REPLICATE,
                    p=settings.rotate_p,
                )
            )
        else:
            raise ValueError(f"不支持的增强步骤: {transform_name}")

    if not transforms:
        raise ValueError("AUG_TRANSFORMS 不能为空")

    compose_kwargs = {
        "keypoint_params": A.KeypointParams(coord_format="xy", remove_invisible=False),
    }
    if with_bboxes:
        compose_kwargs["bbox_params"] = A.BboxParams(
            coord_format="pascal_voc",
            label_fields=["bbox_labels"],
        )

    return A.Compose(transforms, **compose_kwargs)