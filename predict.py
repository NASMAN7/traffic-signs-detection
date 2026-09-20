"""
Inference script for the traffic-sign Faster R-CNN + custom CNN classifier.

Loads the exported checkpoint (my_faster_rcnn.pt) and runs the model on one
or more images, printing the detected boxes and optionally saving an
annotated copy of each image.

Usage:
    python predict.py --checkpoint my_faster_rcnn.pt --images img1.jpg img2.jpg
    python predict.py --checkpoint my_faster_rcnn.pt --images img1.jpg --score-threshold 0.5 --output-dir preds/

The model architecture (My_RPN, faster_R_CNN) and the load/predict helpers
below are copied as-is from the training notebook so that this file can run
independently of it.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.ops import nms


# --------------------------------------------------------------------------
# Model definitions (unchanged from the training notebook)
# --------------------------------------------------------------------------

class My_RPN(nn.Module):
    """COCO-pretrained ResNet-50/FPN backbone + RPN + RoI Align."""

    def __init__(self, nms_thresh=0.0, objectness_thresh=0.0, fg_iou=0.7, bg_iou=0.3):
        super().__init__()

        self.nms_thresh = nms_thresh
        self.objectness_thresh = objectness_thresh
        self.fg_iou = fg_iou
        self.bg_iou = bg_iou
        base_model = fasterrcnn_resnet50_fpn(
            weights=None,
            min_size=640,
            max_size=4000,
            rpn_nms_thresh=nms_thresh,
            rpn_score_thresh=objectness_thresh,
            rpn_fg_iou_thresh=fg_iou,
            rpn_bg_iou_thresh=bg_iou,
        )

        self.preprocessor = base_model.transform
        self.backbone = base_model.backbone
        self.rpn = base_model.rpn
        self.roi_align = base_model.roi_heads.box_roi_pool

    def forward(self, images, targets=None, return_roi_features=True):
        images, targets = self.preprocessor(images, targets)
        features = self.backbone(images.tensors)
        proposals, rpn_losses = self.rpn(images, features, targets)
        if not return_roi_features:
            return None, rpn_losses
        roi_features = self.roi_align(features, proposals, images.image_sizes)
        return roi_features, rpn_losses


class faster_R_CNN(nn.Module):
    """Frozen My_RPN detector + a small CNN classifying pooled ROI features."""

    def __init__(self, detector=None, roi_iou_threshold=0.75, global_nms_threshold=None):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(8, 5),
        )
        if detector is None:
            raise ValueError("A trained RPN must be provided.")
        if global_nms_threshold is not None and not 0 <= global_nms_threshold <= 1:
            raise ValueError("Global NMS must be between 0 and 1, or None to disable it.")
        self.global_nms_threshold = global_nms_threshold
        self.roi_iou_threshold = roi_iou_threshold
        self.detector = detector
        self.detector.requires_grad_(False)
        self.detector.eval()

    @torch.no_grad()
    def get_proposals(self, images, targets=None):
        self.detector.eval()
        image_list, resized_targets = self.detector.preprocessor(images, targets)
        features = self.detector.backbone(image_list.tensors)

        if self.global_nms_threshold is None:
            proposals, _ = self.detector.rpn(image_list, features, resized_targets)
        else:
            rpn = self.detector.rpn
            feature_list = list(features.values())
            logits_per_level, deltas_per_level = rpn.head(feature_list)
            anchors = rpn.anchor_generator(image_list, feature_list)
            anchors_per_level = [level[0].numel() for level in logits_per_level]
            logits, deltas = concat_box_prediction_layers(logits_per_level, deltas_per_level)
            decoded_boxes = rpn.box_coder.decode(deltas, anchors).reshape(len(anchors), -1, 4)
            proposals, scores = rpn.filter_proposals(
                decoded_boxes, logits, image_list.image_sizes, anchors_per_level,
            )
            proposals = [
                boxes[nms(boxes, image_scores, self.global_nms_threshold)]
                for boxes, image_scores in zip(proposals, scores)
            ]
        return features, proposals, resized_targets, image_list.image_sizes

    def forward(self, images, targets=None):
        with torch.no_grad():
            features, proposals, resized_targets, image_sizes = self.get_proposals(images, targets)
            roi_features = self.detector.roi_align(features, proposals, image_sizes)
        logits = self.main(roi_features)
        return logits, None


# --------------------------------------------------------------------------
# Checkpoint loading and inference helpers
# --------------------------------------------------------------------------

def load_classifier_checkpoint(saved_model, device):
    """Rebuild the frozen detector + classifier from an exported checkpoint dict."""
    detector_config = {
        key: float(saved_model[key])
        for key in ("nms_thresh", "objectness_thresh", "bg_iou", "fg_iou")
    }
    model = faster_R_CNN(
        detector=My_RPN(**detector_config),
        roi_iou_threshold=float(saved_model["roi_iou_threshold"]),
        global_nms_threshold=saved_model.get("global_nms_threshold"),
    )
    model.load_state_dict(saved_model["checkpoints"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_image_list(model, images, score_threshold=0.0):
    """Run inference on a list of image tensors (C, H, W), no annotations needed.

    Returns one list per image, each a list of
    [x1, y1, x2, y2, class_id, probability] entries in the original image's
    pixel coordinates. Background predictions and scores below the threshold
    are dropped.
    """
    model.eval()
    model_device = next(model.parameters()).device
    all_predictions = []

    for image in images:
        features, proposals, _, image_sizes = model.get_proposals(
            [image.to(model_device)], targets=None
        )
        boxes = proposals[0]

        if len(boxes) == 0:
            all_predictions.append([])
            continue

        roi_features = model.detector.roi_align(features, proposals, image_sizes)
        probabilities = model.main(roi_features).softmax(dim=1)
        scores, labels = probabilities.max(dim=1)

        keep = (labels != 0) & (scores >= score_threshold)

        height, width = image.shape[-2:]
        resized_height, resized_width = image_sizes[0]
        scale = boxes.new_tensor([
            width / resized_width, height / resized_height,
            width / resized_width, height / resized_height,
        ])

        final_boxes = (boxes[keep] * scale).cpu().tolist()
        final_labels = labels[keep].cpu().tolist()
        final_scores = scores[keep].cpu().tolist()

        image_preds = [
            [box[0], box[1], box[2], box[3], label, score]
            for box, label, score in zip(final_boxes, final_labels, final_scores)
        ]
        all_predictions.append(image_preds)

    return all_predictions


def load_image(path):
    """Read an image file, fix EXIF orientation, and convert to a float tensor."""
    with Image.open(path) as img:
        image = img.convert("RGB")
        image = ImageOps.exif_transpose(image)
    return transforms.ToTensor()(image), image


def draw_and_save(pil_image, predictions, class_names, output_path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(pil_image)
    ax.axis("off")
    for x1, y1, x2, y2, label, score in predictions:
        ax.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="red", linewidth=2,
        ))
        ax.text(
            x1, max(0, y1 - 3), f"{class_names[label]} {score:.2f}",
            color="white", fontsize=9,
            bbox=dict(facecolor="red", alpha=0.8, edgecolor="none"),
        )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                         help="Path to the exported checkpoint (my_faster_rcnn.pt).")
    parser.add_argument("--images", type=Path, nargs="+", required=True,
                         help="One or more image files to run inference on.")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                         help="Minimum softmax score to keep a detection (default: 0.5).")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="If set, save an annotated copy of each image here.")
    parser.add_argument("--device", type=str, default=None,
                         help="cuda or cpu (default: cuda if available).")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Loading checkpoint from {args.checkpoint} on {device} ...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    class_names = checkpoint["class_names"]
    model = load_classifier_checkpoint(checkpoint, device)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in args.images:
        tensor, pil_image = load_image(image_path)
        predictions = predict_image_list(model, [tensor], score_threshold=args.score_threshold)[0]

        print(f"\n{image_path}: {len(predictions)} detection(s)")
        for x1, y1, x2, y2, label, score in predictions:
            print(f"  {class_names[label]:20s} score={score:.2f} "
                  f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

        if args.output_dir:
            out_path = args.output_dir / f"{image_path.stem}_pred.png"
            draw_and_save(pil_image, predictions, class_names, out_path)
            print(f"  Saved annotated image to {out_path}")


if __name__ == "__main__":
    main()