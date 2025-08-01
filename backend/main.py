import os
import glob
import tqdm
import datetime

from app.utils import fill_image_folder
from app.augment import augment_dataset

image_folders = ['train', 'test', 'val']

dataset_path = '/home/hika/work/dataset-augmenter/backend/dataset/fiber-splitter-1k'
NUM_AUG = 10  # 设置增强数量为3

DATATIME_NOW = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = dataset_path + f'-{DATATIME_NOW}'
# 遍历每个图像文件夹
for image_folder in tqdm.tqdm(image_folders, desc="Processing folders"):
    # 获取所有图像文件
    image_files = glob.glob(os.path.join(dataset_path, 'images', image_folder, '*.jpg'))
    
    # 复制data.yml 文件到输出目录
    data_yml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    os.makedirs(os.path.dirname(data_yml_path), exist_ok=True)
    os.system(f'cp {os.path.join(dataset_path, "data.yaml")} {data_yml_path}')
    
    # 生train.txt, test.txt, val.txt 文件
    txt_file_path = os.path.join(OUTPUT_DIR, f'{image_folder}.txt')
    fill_image_folder(txt_file_path, image_files, image_folder, NUM_AUG=NUM_AUG)

    for image_file in tqdm.tqdm(image_files, desc=f"Processing {image_folder}", leave=False):
        # 获取图像的基本名称（不带路径和扩展名）
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        try:
            # 调用数据增强函数
            augment_dataset(
                abs_path=dataset_path,
                image_folder=image_folder,
                image_name=base_name,
                num_aug=NUM_AUG,  # 设置增强数量为3
                output_dir = OUTPUT_DIR
            )
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

print("数据增强完成！")