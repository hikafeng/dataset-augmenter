## Dataset Augmenter

这是一个面向 YOLO 数据集的增强工具，当前支持两类标注格式：

- detection：项目当前已有的点集/轮廓标签格式
- pose：YOLO pose 的 bbox + keypoints + visibility 格式

## 功能

- 支持 detection 和 pose 两类数据集自动识别
- 支持从 .env 读取增强步骤和参数
- 自动兼容 jpg、jpeg、png、bmp、webp 图片
- 自动扫描 train、val、test，兼容大小写不一致和嵌套目录
- 自动重写输出数据集的 data.yaml 和 split 文件
- 可选导出带标注的可视化图片

## 配置

默认读取项目根目录下的 .env。

常用配置项：

- AUG_DATASET_PATH：待增强数据集根目录
- AUG_DATASET_TYPE：auto、detection、pose
- AUG_NUM_AUG：每张图增强次数
- AUG_TRANSFORMS：增强步骤，支持 resize、rotate、hflip、vflip
- AUG_SAVE_LABELED：是否导出可视化结果

## 运行

```bash
python main.py
```

如果要指定其它环境变量文件：

```bash
python main.py --env-file .env.example
```

## 输出

程序会在源数据集同级目录生成一个带时间戳的新目录，例如：

```text
dataset/detection/my-dataset-1k-yolo-20260509-153000
```

输出目录中会包含：

- 增强后的 images 和 labels
- 重写后的 data.yaml
- 对应的 train.txt、val.txt、test.txt 或原始大小写文件名
- 可选的 images/labeled 可视化结果

## 开发验证

```bash
python -m py_compile main.py app/*.py
python -m unittest discover -s tests -p 'test_*.py'
```
