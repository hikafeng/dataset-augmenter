import os
import cv2

KEYPOINT_COLOR = (0, 255, 0)  # Green
WIDTH_RESIZE = 640
HIGH_RESIZE = 640

from app.pipeline import pipeline

def save_image_uint8(path, img):
    # if img.dtype != 'uint8':
    #     img = (img * 255).clip(0,255).astype('uint8') if img.max() <= 1.0 else img.astype('uint8')
    cv2.imwrite(path, img)

def fill_image_folder(txt_file_path, image_files, image_folder, NUM_AUG=3):
    with open(txt_file_path, 'w') as f:
        for image_file in image_files:
            # 获取图像的基本名称（不带路径和扩展名）
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            for i in range(1,NUM_AUG+1):
                # 写入增强后的图像名称
                f.write(f"images/{image_folder}/{base_name}_{i:06d}.jpg\n")