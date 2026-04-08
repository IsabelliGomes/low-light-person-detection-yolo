# Low-Light Person Detection using YOLOv5

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
├── videos/         # Raw and processed videos
├── results/        # Predictions and generated graphs
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

Clone YOLOv5 and install dependencies:

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## 🚀 Training

```bash
python train.py --img 640 --batch 16 --epochs 50 --data dataset/dataset.yaml --weights yolov5s.pt
```

## 🔍 Inference

```bash
python detect.py --weights best.pt --img 640 --source videos/
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

