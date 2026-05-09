import unittest
from pathlib import Path

from app.utils import build_augmented_image_path, extract_relative_image_path, image_to_label_path


class UtilsTests(unittest.TestCase):
    def test_extract_relative_image_path_from_absolute_entry(self):
        entry = "/tmp/dataset/images/Test/test/000858.png"
        self.assertEqual(
            extract_relative_image_path(entry),
            Path("images/Test/test/000858.png"),
        )

    def test_image_to_label_path(self):
        self.assertEqual(
            image_to_label_path(Path("images/train/000001.jpg")),
            Path("labels/train/000001.txt"),
        )

    def test_build_augmented_image_path(self):
        self.assertEqual(
            build_augmented_image_path(Path("images/train/000001.jpg"), 7),
            Path("images/train/000001_000007.jpg"),
        )