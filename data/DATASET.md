# Dataset Preparation Guidelines

This document provides guidelines for preparing the datasets used in RaPO.

## Image Classification

We use four datasets for class-incremental image classification: TinyImageNet, CUB-200, ImageNet-R, and ImageNet-A. The download links are listed below:

- TinyImageNet: [Link](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
- CUB-200-2011: [Link](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- ImageNet-R: [Link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view)
- ImageNet-A: [Link](https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view)

After downloading the datasets, use the following command to construct a continual-learning dataset. For example:

```bash
python data/create_image_cls_cil_dataset.py \
  --dataset inr \
  --raw-root /path/to/imagenet-r \
  --copy-mode symlink
```

Use `--copy-mode copy` if symlinks are not suitable.

The expected folder structure is:

```text
data/image_cls_cil_dataset/
  imagenet-r/
    train/
    train_5shots/
    test/
    metadata.json
```

We also provide the extracted sample IDs in `data/image_cls_cil` to support reproducibility.

## Video Classification

Download the UCF101 dataset from [OpenDataLab](https://opendatalab.com/OpenDataLab/UCF101), and ensure that its layout matches the following structure:

```text
/path/to/UCF101/raw/
  UCF-101/
    <RawClassName>/
      *.avi
  ucfTrainTestlist/
    classInd.txt
    trainlist01.txt
    testlist01.txt
```


Then use the following command to construct the continual-learning dataset:

```bash
python data/create_video_cls_cil_ucf101_dataset.py \
  --raw-root /path/to/UCF101/raw \
  --copy-mode symlink
```

Use `--copy-mode copy` if symlinks are not suitable.

The expected folder structure is:

```text
data/video_cls_cil_dataset/
  UCF101/
    cil_split01/
      train/
        <Class_Name>/
          *.avi
      train_5shots/
        <Class_Name>/
          *.avi
      test/
        <Class_Name>/
          *.avi
      metadata.json
```

## Object Detection

Prepare COCO 2017 according to the [mmcv guidelines](https://github.com/open-mmlab/mmdetection/blob/2.x/docs/en/1_exist_data_model.md), and ensure that its layout matches the following structure:

```text
/path/to/coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
    *.jpg
  val2017/
    *.jpg
```

The training image list is provided in `data/object_det_cil_dataset`. `train_5shots.jsonl` contains 206 training images, and fixed seeds are used to control the class order: 5-task uses seeds `136`, `377`, and `639`, while 10-task uses seeds `277`, `305`, and `738`. The reason behind this is COCO contains many multi-label samples. If the dataset and learning order are constructed arbitrarily, the same image may appear in different tasks. Even when the available annotations differ across tasks, this can still introduce a degree of replay. We therefore carefully select the few-shot subset and class order to minimize this issue.

You can also rebuild the continual-learning dataset as follows:

```bash
python data/create_object_det_cil_dataset.py \
  --coco-root /path/to/coco \
  --source-train-jsonl data/object_det_cil_dataset/train_5shots.jsonl \
  --source-val-jsonl data/object_det_cil_dataset/val.jsonl \
  --source-categories data/object_det_cil_dataset/categories.json \
  --output-dir data/object_det_cil_dataset_rebuilt
```
