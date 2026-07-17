# Person Detection in Low-Light Environments: Evaluation of an Annotation-Free Approach
Manuscript ID: IEEE LATAM Submission ID 10779 
Authors:
* Isabelli Pinto Gomes
* Flávio Luis de Mello

Affiliation:

Machine Intelligence and Computing Models Laboratory (IM2C/DEL/Poli/UFRJ)

Eletronics and Computer Engineering Department

Polytechnic School

Federal University of Rio de Janeiro

Brazil

## 📌 Overview

This repository presents an annotation-free approach for person detection in low-light environments using YOLOv5. The project evaluates how different video enhancement techniques impact detection performance under challenging lighting conditions.

The repository includes:

* Source code for training, inference, and analysis
* Dataset in YOLO format
* Original and processed videos
* Reproducible experiments and results

## 🎯 Objective

The goal of this project is to assess the robustness of object detection models in low-light scenarios without requiring additional manual annotations, leveraging preprocessing techniques to improve detection performance.


## 🧠 Methodology

The approach consists of:

1. Training a YOLOv5 model on a standard dataset
2. Applying different enhancement techniques to low-light videos:

   * No light (baseline)
   * Infrared (IR)
   * Shadow recovery (Filmora)
   * Brightness/contrast enhancement (Veed)
3. Running inference on processed videos
4. Comparing confidence scores across frames


## 📂 Project Structure

```
.
├── code/           # Training, inference, and analysis scripts
├── dataset/        # Images and labels in YOLO format
├── video/          # Raw and processed videos
├── requirements.txt
└── README.md
```

## 📂 Code Structure

* `generate_datasets.py` → Creates datasets with different lighting conditions
* `augment_datasets.py` → Applies data augmentation techniques
* `generate_script_video.py` → Runs YOLOv5 detection on videos
* `generate_comparison_graphs.py` → Generates confidence comparison plots

## 📊 Dataset

The dataset includes multiple variations with different lighting conditions:

* Original datasets
* Augmented datasets
* Low-light simulated datasets

Annotations follow the YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

Videos used for evaluation are available in the `video/` folder.


## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/IsabelliGomes/low-light-person-detection-yolo
cd low-light-person-detection-yolo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Clone YOLOv5 (Ultralytics)

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 4. Run the pipeline

#### Generate datasets

```bash
python code/generate_datasets.py
```

#### Apply augmentations

```bash
python code/augment_datasets.py
```

#### Run detection on videos

```bash
python code/generate_script_video.py
```

#### Generate comparison graphs

```bash
python code/generate_comparison_graphs.py
```


## 🚀 Training

```bash
python train.py --img 640 --batch 16 --epochs 50 --data dataset/dataset.yaml --weights yolov5s.pt
```

## 🔍 Inference

```bash
python detect.py --weights best.pt --img 640 --source video/
```

## 📊 Results

The results include:

* Frame-by-frame confidence scores
* Comparative graphs across different lighting conditions
* CSV files with predictions

Example comparisons:

* Baseline vs Infrared (IR)
* Filmora (shadow recovery) vs Veed (brightness/contrast)

## 🔁 Reproducibility

All experiments can be reproduced using:

* The provided dataset configuration (`dataset.yaml`)
* The scripts in the `code/` directory
* The videos available in the repository


## ⚠️ Notes

* Ensure correct paths when running training and inference
* This project uses [YOLOv5 from Ultralytics](https://github.com/ultralytics/yolov5)


## 📄 Academic Context

This project was developed as part of undergraduate research at UFRJ.


## 📚 Keywords

low-light, computer vision, YOLOv5, object detection, deep learning, person detection

