import os
import cv2
import albumentations as A
import datetime
import math

KEYPOINT_COLOR = (0, 255, 0)  # Green
WIDTH_RESIZE = 640
HIGH_RESIZE = 640

pipeline = A.Compose(
    [
        A.Resize(WIDTH_RESIZE, HIGH_RESIZE),  # Resize to 640x640
        # A.RandomCrop(width=600, height=600, p=0.5),  # 随机裁剪
        # A.CenterCrop(width=600, height=600, p=0.2),  # 中心裁剪
        # A.RandomResizedCrop(WIDTH_RESIZE, HIGH_RESIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33), p=0.3),  # 随机缩放裁剪
        A.HorizontalFlip(p=0.5),  # 左右翻转
        A.VerticalFlip(p=0.2),  # 上下翻转
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, p=0.5),  # 平移、缩放、旋转
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.8),  # 亮度对比度
        A.HueSaturationValue(hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20), p=0.5),  # 色调饱和度
        A.RGBShift(r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20), p=0.5),  # RGB通道抖动
        A.ChannelShuffle(p=0.1),  # 通道混合 随机改变RGB三个通道的顺序。
        A.GaussianBlur(p=0.3),  # 高斯模糊
        A.MotionBlur(blur_limit=7, p=0.2),  # 运动模糊
        A.MedianBlur(blur_limit=7, p=0.1),  # 中值模糊
        A.GaussNoise(std_range=(0.01,0.1)),  # 高斯噪声
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.1),  # ISO噪声
        A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=0.1),  # 直方图均衡
        A.InvertImg(p=0.05),  # 反色
        A.ToGray(p=0.1),  # 灰度
        A.CoarseDropout(num_holes_range=(1,8), hole_height_range=(0.01,0.04), hole_width_range=(0.01,0.04), p=0.3),  # 随机擦除
        # A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.3),  # 另一种擦除方式（和CoarseDropout类似）
        A.SafeRotate(limit=(-90, 90), p=0.5),  # 随机安全旋转
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 归一化

    ],
    keypoint_params=A.KeypointParams(format='xy',remove_invisible=False)
)