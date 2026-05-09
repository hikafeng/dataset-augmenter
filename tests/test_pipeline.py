import unittest
from pathlib import Path

import numpy as np

from app.config import Settings
from app.pipeline import build_pipeline


class PipelineTests(unittest.TestCase):
    def test_rotate_preserves_keypoint_count(self):
        settings = Settings(
            dataset_path=Path("dataset"),
            dataset_type="pose",
            num_aug=1,
            output_root=None,
            save_labeled=False,
            resize_width=64,
            resize_height=64,
            transforms=["rotate"],
            horizontal_flip_p=0.0,
            vertical_flip_p=0.0,
            rotate_p=1.0,
            rotate_limit=45,
        )
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        keypoints = [(20.0, 20.0), (40.0, 40.0)]

        transformed = build_pipeline(settings, with_bboxes=False)(
            image=image,
            keypoints=keypoints,
        )

        self.assertEqual(len(transformed["keypoints"]), len(keypoints))