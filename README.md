# 🚦 Nighttime Traffic Sign Detection (Two-Stage Detector)

> **Project Status:** 🚧 Under Development (Work In Progress)

This project implements a two-stage detector system to classify four primary traffic sign categories: **no_parking**, **pedestrian_crossing**, **yield**, and **no_entry**. 

The pipeline combines a Region Proposal Network (RPN) built on ResNet-50/FPN features, a RoI Align pooling mechanism, and a custom CNN classifier.

---

## 🛠️ Execution Environment (Kaggle Server)

This project is designed to be executed on **Kaggle** servers using Jupyter notebooks.

* **Dataset:** The dataset is stored in the project repository under `traffic-signs-dataset` (in YOLO format) and was imported into the Kaggle environment to fuel the training pipeline.
* **Paths:** Data paths, annotation paths, and saving/upload directories (`saving_dir`) are configured relative to the standard Kaggle session tree structure (e.g., `/kaggle/input/...`).

---

## 📂 Project Structure

```text
├── traffic-signs-dataset/        # Dataset (images and annotations in YOLO format)
│   ├── train/ (images & labels)
│   ├── valid/ (images & labels)
│   └── test/  (images & labels)
├── saving_dir/                   # Training artifacts and model weights (Kaggle session specific)
└── my_CNN.ipynb                  # Main experimentation and training notebook
```

---

## ⚙️ Pipeline and Architecture

* **Exploratory Analysis:** Class distribution check and visual inspection of training samples.
* **Region Proposals:** Utilization and fine-tuning of an RPN (initially pretrained on COCO) to propose candidate regions of interest, handling challenging low-light nighttime scenes.
* **Extraction & Classification:** Applying RoI Align on pooled features to feed a custom CNN responsible for classifying each region into one of the four sign classes or background.
* **Current Focus & Next Steps:**
  * **Current Phase:** Actively tuning and adjusting the custom CNN parameters and training configuration.
  * **Next Phase:** Transitioning to a Vision Transformer (ViT) approach to benchmark classification performance and computational cost.
* **Optimization:** Performance evaluation and tuning of Non-Maximum Suppression (NMS) thresholds.

---

## 💻 Technologies & Libraries

* **Language:** Python
* **Deep Learning:** PyTorch, Torchvision
* **Data Processing:** NumPy, Pandas, Matplotlib, OpenCV
* **Platform:** Kaggle Notebooks

---

## 👤 Author

**Abdelhak Nasmane**  
* 3rd-year Engineering Student at ENSEEIHT (Computer Vision, 3D & Multimodal AI)  
* [LinkedIn Profile](https://www.linkedin.com/in/abdelhak-nasmane/) | [GitHub](https://github.com/NASMAN7)
