import os
import cv2
import albumentations as A
import datetime
import math
import numpy as np

from app.utils import *

KEYPOINT_COLOR = (0, 255, 0)  # Green
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 640

from app.pipeline import pipeline

def augment_dataset(abs_path, image_folder, image_name, num_aug, output_dir=None, save_labeled=False):
    """
    对输入的图像和对应的标签进行数据增强，并保存标注了 keypoints 的原图和增强后的图像。
    """
    if output_dir is None:
        output_dir = abs_path + f'-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    image_path = os.path.join(abs_path, 'images', image_folder, f'{image_name}.jpg')
    label_path = os.path.join(abs_path, 'labels', image_folder, f'{image_name}.txt')

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像：{image_path}")

    with open(label_path, 'r') as f:
        lines = f.readlines()

    idxs, idxs_len, keypoints = [], [], []
    img_h, img_w = image.shape[:2]
    for line in lines:
        tokens = line.strip().split()
        if len(tokens) < 3:
            continue
        cls = int(tokens[0])
        idxs.append(cls)
        mask_coords = tokens[1:]
        n_pairs = len(mask_coords) // 2
        idxs_len.append(n_pairs)
        for i in range(n_pairs):
            x = min(img_w - 1, math.floor(float(mask_coords[2 * i]) * img_w))
            y = min(img_h - 1, math.floor(float(mask_coords[2 * i + 1]) * img_h))
            keypoints.append((x, y))

    # 创建输出目录结构
    for subdir in [os.path.join(output_dir, 'images', image_folder),
                   os.path.join(output_dir, 'labels', image_folder),
                   os.path.join(output_dir, 'images', 'labeled')]:
        os.makedirs(subdir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(1, num_aug + 1):
        transformed_data = pipeline(image=image, keypoints=keypoints)
        transformed_image = transformed_data['image']
        transformed_keypoints = transformed_data.get("keypoints", [])

        # 标注原始和增强图像
        annotated_original = image.copy()
        for x, y in keypoints:
            cv2.circle(annotated_original, (int(x), int(y)), 2, KEYPOINT_COLOR, -1)

        annotated_transformed = transformed_image.copy()
        for x, y in transformed_keypoints:
            cv2.circle(annotated_transformed, (int(x), int(y)), 2, KEYPOINT_COLOR, -1)

        keypoints_strs = []
        idx_start = 0

        # --------- numpy 批量归一化和超限输出 ---------
        if transformed_keypoints:
            kp_array = np.array(transformed_keypoints, dtype=np.float32)  # shape: (N, 2)
            norm_kp = kp_array / np.array([[RESIZE_WIDTH, RESIZE_HEIGHT]], dtype=np.float32)
            mask = (norm_kp >= 1).any(axis=1)
            if np.any(mask):
                over_idx = np.where(mask)[0]
                # 批量输出超限点信息
                over_xy = kp_array[over_idx]
                over_norm = norm_kp[over_idx]
                for idx, (xy, norm_xy) in zip(over_idx, zip(over_xy, over_norm)):
                    print(f"[Warning] 图像 {base_name}_{i:06d}: keypoint 超出范围: "
                          f"({xy[0]:.2f}, {xy[1]:.2f}) -> ({norm_xy[0]:.6f}, {norm_xy[1]:.6f})")
        else:
            kp_array = np.zeros((0,2), dtype=np.float32)
            norm_kp = kp_array

        # --------- 生成 keypoints_strs ---------
        for idx_i, num in enumerate(idxs_len):
            idx_end = idx_start + num
            kp_slice = transformed_keypoints[idx_start:idx_end] if transformed_keypoints else []
            keypoints_str = f"{idxs[idx_i]}"
            if kp_slice and len(kp_slice) == num:
                # 用 numpy 方式拼接字符串
                norm_slice = norm_kp[idx_start:idx_end]
                keypoints_str += " " + " ".join([f"{x:.6f} {y:.6f}" for x, y in norm_slice])
            else:
                print(f"[Warning] 图像 {base_name}_{i:06d}: 类别 {idxs[idx_i]} keypoints 异常 - "
                      f"期望数量 {num}，实际获得 {len(kp_slice)}，内容: {kp_slice}")
            if keypoints_str.strip() == str(idxs[idx_i]):
                print(f"[Warning] 图像 {base_name}_{i:06d}: 生成空的 keypoints_str: '{keypoints_str}'")
            keypoints_strs.append(keypoints_str)
            idx_start = idx_end

        # 保存标签
        label_save_path = os.path.join(output_dir, 'labels', image_folder, f"{base_name}_{i:06d}.txt")
        with open(label_save_path, 'w') as f:
            f.write('\n'.join(keypoints_strs) + '\n')

        # 保存增强后的图像
        image_save_path = os.path.join(output_dir, 'images', image_folder, f"{base_name}_{i:06d}.jpg")
        save_image_uint8(image_save_path, transformed_image)

        # 保存带label的图片（仅保存最后一组）
        if save_labeled and i == num_aug:
            orig_labeled_path = os.path.join(output_dir, 'images', 'labeled', f"{base_name}_labeled.jpg")
            save_image_uint8(orig_labeled_path, annotated_original)
            trans_labeled_path = os.path.join(output_dir, 'images', 'labeled', f"{base_name}_aug_labeled.jpg")
            save_image_uint8(trans_labeled_path, annotated_transformed)

if __name__ == '__main__':
    augment_dataset(
        abs_path='/home/hika/work/dataset-augmenter/backend/dataset/fiber-splitter-10k',
        image_folder='test',
        image_name='000944',
        num_aug=3
    )