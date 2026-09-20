# 🚦 Nighttime Traffic Sign Detection (Two-Stage Detector)

> **Project Status:** 🚧 Under Development (Work In Progress)

This project implements a two-stage detector for four traffic sign categories, including low-light nighttime scenes: **no_parking**, **pedestrian_crossing**, **yield**, and **no_entry**.

The pipeline combines:
1. A **Region Proposal Network (RPN)** built on a COCO-pretrained ResNet-50/FPN backbone, fine-tuned on the traffic-sign dataset,
2. **RoI Align** pooling on the proposed regions,
3. A small **custom CNN classifier** that assigns each pooled region to one of the four sign classes or background.

`Image → ResNet-50/FPN → RPN proposals (per-level NMS) → global NMS → RoI Align → CNN class scores`

---

## 📊 Results (Test Set)

Final model: **RPN objectness 0.8**, **RPN NMS 0.3**, **global NMS 0.0**, **BG/FG IoU 0.25/0.75**, **ROI IoU 0.75**, **class-weighting exponent (alpha) 0.5**.

**Per-class scores (test ROIs):**

| Class | Support (ROIs) | Precision | Recall | F1 |
|---|---|---|---|---|
| no_parking | 17 | 59.09% | 76.47% | 66.67% |
| pedestrian_crossing | 7 | 38.89% | 100.00% | 56.00% |
| yield | 5 | 31.25% | 100.00% | 47.62% |
| no_entry | 21 | 94.74% | 85.71% | 90.00% |

**Summary:**

| Alpha | Test images | Test loss (all ROIs, unweighted) | Selected sign ROIs | Correct sign ROIs | Sign ROI accuracy | Sign macro F1 |
|---|---|---|---|---|---|---|
| 0.5 | 70 | 0.20435 | 50 | 43 | 86.00% | 65.07% |

These are **ROI-level classification metrics** on proposals that survived the RPN and NMS stages — not an object-level detection score (mAP). A sign with no surviving proposal is not counted here. `pedestrian_crossing` and `yield` have very few test ROIs (7 and 5), so their precision/recall should be read with that small support in mind rather than as a stable estimate.

---

## ⚙️ Pipeline and What Has Been Done

The full experiment, with every intermediate table and plot, is in the notebook (`traffic-sign-detection.ipynb`):

1. **Exploratory analysis** — class distribution per split, visual inspection of training samples (day/night, scale, occlusion).
2. **Pretrained RPN baseline** — anchor-level recall/precision/F1 swept over 4 objectness thresholds × 3 BG/FG IoU pairs (12 configurations), evaluated before any weight update.
3. **RPN fine-tuning** — the 12 configurations fine-tuned from the COCO initialization (backbone frozen), then re-evaluated on train and validation to confirm the gain holds beyond the training images.
4. **Candidate selection** — the two best RPN configurations (by validation recall/precision trade-off) exported and reused as fixed proposal generators.
5. **Joint NMS grid search** — RPN-level NMS × global (cross-FPN-level) NMS × the 2 RPN candidates (18 configurations), each training a fresh CNN classifier head; selection by sign macro F1 rather than raw loss.
6. **Class-weighting sweep** — the winning NMS configuration retrained with three class-weighting exponents (alpha = 0.1, 0.3, 0.5) to counter background/foreground imbalance; alpha = 0.5 selected on validation.
7. **Final export & test evaluation** — the alpha = 0.5 checkpoint (`my_faster_rcnn.pt`) evaluated once on the held-out test split (results above).
8. **Qualitative check** — predictions vs. ground truth visualized on test images to sanity-check what the numbers mean in practice.

---

## 📂 Repository Structure

```text
├── traffic-signs-dataset/   # Dataset (images + YOLO-format annotations)
│   ├── train/ (images & labels)
│   ├── valid/ (images & labels)
│   └── test/  (images & labels)
├── traffic-sign-detection.ipynb             # Full training/experimentation notebook (all steps above, with results)
├── predict.py                # Standalone inference script (no notebook required)
├── requirements.txt          # Dependencies for predict.py
└── my_faster_rcnn.pt         # Exported final checkpoint (detector + classifier + config)
```

---

## 🛠️ About the Kaggle Paths

The notebook was developed and run on **Kaggle**, so its path variables (`data_path`, `saving_dir`, `/kaggle/working/...`) point to the Kaggle session's file tree, not to this repository's layout. If you want to **re-run the notebook** (rather than just use `predict.py`), you only need to edit the paths near the top of the notebook:

```python
data_path  = "path/to/traffic-signs-dataset"   # instead of /kaggle/input/...
saving_dir = "path/to/where/you/save/checkpoints"
```

Everything downstream (dataloaders, checkpoints, plots) follows from these two variables. `predict.py`, on the other hand, does **not** depend on any of this — it only needs the exported checkpoint file.

---

## 🚀 Quickstart: Inference

```bash
pip install -r requirements.txt

python predict.py \
  --checkpoint my_faster_rcnn.pt \
  --images path/to/image1.jpg path/to/image2.jpg \
  --score-threshold 0.5 \
  --output-dir preds/
```

This prints the detected boxes/classes/scores to the console and, if `--output-dir` is given, saves an annotated copy of each image.

---

## 🔭 Next Steps

* **Swap the CNN classifier for a Vision Transformer (ViT)** on the pooled RoI features, to benchmark classification accuracy and inference cost against the current small CNN head.
* Extend evaluation to an **object-level metric (mAP)** to complement the current ROI-level scores.
* Improve robustness on low-support classes (`pedestrian_crossing`, `yield`) — likely via more training data or targeted augmentation rather than architecture changes alone.

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
