# 🩺 YOLOv5 Polyp Detection

![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-orange)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-informational)
![License](https://img.shields.io/badge/License-Ultralytics-green)

Automatic **polyp detection from endoscopic images** using **YOLOv5**, trained on a **Roboflow Polyp Detection dataset**.
The full pipeline—dataset preparation, training, evaluation, and inference—is implemented in a Jupyter Notebook.

---

## 📌 Overview

Colorectal cancer prevention relies heavily on accurate polyp detection.
This project applies **state-of-the-art object detection (YOLOv5)** to identify polyps in medical images with high accuracy and speed.

### 🔑 Highlights

* YOLOv5 (Ultralytics) object detection
* Transfer learning with pretrained weights
* Trained on a Roboflow-hosted medical dataset
* End-to-end workflow in a single notebook

---

## 🧠 Model

* **Architecture:** YOLOv5
* **Framework:** PyTorch
* **Task:** Single-class object detection (polyp)
* **Approach:** Fine-tuning pretrained weights

---

## 📦 Dataset

**Polyp Detection Dataset – Roboflow**

🔗 **Dataset & Model Link**
[https://app.roboflow.com/polyp-e78ji/polyp_detection-k9te7/models](https://app.roboflow.com/polyp-e78ji/polyp_detection-k9te7/models)

### Dataset Features

* Bounding-box annotations
* YOLOv5-ready export
* Train / Validation / Test split
* Medical endoscopy images

### Dataset Structure

```
dataset/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/
    ├── train/
    ├── valid/
    └── test/
```

### Annotation Format

```
<class_id> <x_center> <y_center> <width> <height>
```

(All values normalized between 0 and 1)

---

## 📂 Project Structure

```
.
├── Copy_of_yolov5polypdetection.ipynb   # Main notebook
├── dataset/                             # Roboflow dataset
├── runs/                                # YOLOv5 outputs
├── weights/                             # Saved model weights
└── README.md                            # Documentation
```

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* YOLOv5 dependencies

### Optional Local Setup

```bash
git clone https://github.com/ultralytics/yolov5.git
pip install -r yolov5/requirements.txt
```

---

## 🚀 Usage

1. Open the notebook:

   ```bash
   jupyter notebook Copy_of_yolov5polypdetection.ipynb
   ```
2. Download the dataset from Roboflow in **YOLOv5 format**
3. Update dataset paths if required
4. Run cells sequentially:

   * Install dependencies
   * Train the model
   * Evaluate performance
   * Run inference

---

## 📊 Training Outputs

YOLOv5 automatically logs:

* Training & validation loss
* Precision, Recall, mAP
* Best & last model checkpoints

Saved in:

```
runs/train/
```

---

## 🔍 Inference Results

* Bounding boxes around detected polyps
* Confidence scores per detection
* Output images saved to:

```
runs/detect/
```

---

## 🧪 Applications

* Medical image analysis
* Colonoscopy screening assistance
* Computer vision research in healthcare
* Academic and educational use

---

## ⚠️ Disclaimer

🚨 **Not for clinical use**
This project is intended **only for research and educational purposes** and is not a certified medical diagnostic system.

---

## 📜 License

* **YOLOv5:** Ultralytics License
* **Dataset:** Roboflow Dataset License

Please review respective sources for full licensing terms.

---

## 🙌 Acknowledgements

* **Roboflow Polyp Detection Dataset**
* **Ultralytics YOLOv5**
* Open-source medical imaging community

---

## 📬 Contact

Contributions, issues, and improvements are welcome.

⭐ If you find this project useful, consider starring the repository!
