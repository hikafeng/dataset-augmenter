import unittest

from app.annotations import DetectionAnnotationCodec, PoseAnnotationCodec


class AnnotationCodecTests(unittest.TestCase):
    def test_detection_roundtrip(self):
        codec = DetectionAnnotationCodec()
        lines = ["1 0.100000 0.200000 0.300000 0.400000"]

        annotations = codec.parse(lines, (100, 200))
        serialized = codec.serialize(annotations, (100, 200))

        self.assertEqual(serialized, lines)

    def test_pose_roundtrip(self):
        codec = PoseAnnotationCodec()
        lines = [
            "0 0.500000 0.500000 0.400000 0.200000 0.250000 0.300000 2 0.750000 0.700000 2"
        ]

        annotations = codec.parse(lines, (100, 200))
        serialized = codec.serialize(annotations, (100, 200))

        self.assertEqual(serialized, lines)

    def test_pose_rebuild_uses_bbox_corners(self):
        codec = PoseAnnotationCodec()
        annotations = codec.parse(
            [
                "0 0.500000 0.500000 0.400000 0.200000 0.250000 0.300000 2 0.000000 0.000000 0 0.750000 0.700000 2"
            ],
            (100, 200),
        )
        transform_input = codec.build_transform_input(annotations)
        transformed = {
            "keypoints": [
                (60.0, 35.0),
                (160.0, 75.0),
                (45.0, 25.0),
                (155.0, 30.0),
                (165.0, 75.0),
                (55.0, 80.0),
            ]
        }

        rebuilt = codec.rebuild(transformed, (100, 200), annotations)

        self.assertEqual(len(transform_input["keypoints"]), 6)
        self.assertEqual(rebuilt[0].bbox, (45.0, 25.0, 165.0, 80.0))
        self.assertEqual(rebuilt[0].keypoints[1], (0.0, 0.0))
        self.assertEqual(rebuilt[0].visibility, [2, 0, 2])