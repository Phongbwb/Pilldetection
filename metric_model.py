from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
import torch
import supervision as sv
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import time
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from matplotlib.patches import Rectangle
from fvcore.nn import FlopCountAnalysis
from ultralytics.utils import ops
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# === Collate Function ===
def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]
    return pixel_values, labels
def pairwise_iou(boxes1, boxes2):
    """
    boxes1, boxes2: numpy arrays of shape (N, 4) in xyxy format
    Return: IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    ious = np.zeros((len(boxes1), len(boxes2)))

    for i in range(len(boxes1)):
        x1, y1, x2, y2 = boxes1[i]
        for j in range(len(boxes2)):
            x1g, y1g, x2g, y2g = boxes2[j]

            xi1 = max(x1, x1g)
            yi1 = max(y1, y1g)
            xi2 = min(x2, x2g)
            yi2 = min(y2, y2g)

            inter_w = max(xi2 - xi1, 0)
            inter_h = max(yi2 - yi1, 0)
            inter_area = inter_w * inter_h

            union = area1[i] + area2[j] - inter_area
            ious[i, j] = inter_area / union if union > 0 else 0.0

    return ious

def analyze_failures(predictions, targets, iou_threshold=0.5, conf_threshold=0.1):
    """
    Phân tích lỗi: false positives, false negatives, wrong class, localization errors, duplicate detections.
    predictions: list of supervision.Detections
    targets: list of supervision.Detections
    """
    failures = {
        "false_positives": [],
        "false_negatives": [],
        "wrong_class": [],
        "localization_errors": [],
        "duplicate_detections": [],
        "low_confidence": []
    }

    for idx, (pred, gt) in enumerate(zip(predictions, targets)):
        image_id = f"{idx}"
        ious = pairwise_iou(pred.xyxy, gt.xyxy) if len(pred) > 0 and len(gt) > 0 else np.zeros((len(pred), len(gt)))

        matched_gt = set()
        matched_pred = set()

        for pred_idx in range(len(pred)):
            pred_box = pred.xyxy[pred_idx]
            pred_label = pred.class_id[pred_idx]
            pred_score = pred.confidence[pred_idx] if pred.confidence is not None else 1.0

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx in range(len(gt)):
                if gt_idx in matched_gt:
                    continue
                iou = ious[pred_idx, gt_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                gt_label = gt.class_id[best_gt_idx]

                if pred_label != gt_label:
                    failures["wrong_class"].append({
                        "image_id": image_id,
                        "pred_box": pred_box,
                        "pred_label": pred_label,
                        "gt_box": gt.xyxy[best_gt_idx],
                        "gt_label": gt_label,
                        "iou": best_iou
                    })
                elif best_iou < 0.75:
                    failures["localization_errors"].append({
                        "image_id": image_id,
                        "pred_box": pred_box,
                        "pred_label": pred_label,
                        "gt_box": gt.xyxy[best_gt_idx],
                        "gt_label": gt_label,
                        "iou": best_iou
                    })

                if pred_score < conf_threshold:
                    failures["low_confidence"].append({
                        "image_id": image_id,
                        "pred_box": pred_box,
                        "pred_label": pred_label,
                        "score": pred_score
                    })
            else:
                failures["false_positives"].append({
                    "image_id": image_id,
                    "pred_box": pred_box,
                    "pred_label": pred_label,
                    "score": pred_score
                })

        # Các ground truth không khớp -> false negatives
        for gt_idx in range(len(gt)):
            if gt_idx not in matched_gt:
                failures["false_negatives"].append({
                    "image_id": image_id,
                    "gt_box": gt.xyxy[gt_idx],
                    "gt_label": gt.class_id[gt_idx]
                })

        # Duplicate predictions
        seen = {}
        for pred_idx in matched_pred:
            key = tuple(pred.xyxy[pred_idx].astype(int))
            if key in seen:
                failures["duplicate_detections"].append({
                    "image_id": image_id,
                    "pred_box": pred.xyxy[pred_idx],
                    "pred_label": pred.class_id[pred_idx],
                    "score": pred.confidence[pred_idx]
                })
            else:
                seen[key] = 1

    return failures

# === Custom RT-DETR Dataset ===
class RTDETRDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]
        self.img_size = img_size
        self.processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.resize = transforms.Resize((img_size, img_size))

        # === Tự động xác định số class từ file nhãn ===
        all_class_ids = []
        for img_file in self.image_files:
            label_path = os.path.join(root_dir, img_file.replace("images", "labels").rsplit(".", 1)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        class_id = int(float(parts[0]))
                        all_class_ids.append(class_id)

        self.num_classes = max(all_class_ids) + 1 if all_class_ids else 1

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.resize(image)
        width, height = image.size

        label_path = image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        boxes, class_ids = [], []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, cx, cy, w, h = map(float, parts)
                    boxes.append([cx, cy, w, h])
                    class_ids.append(int(class_id))

        encoding = self.processor(images=image, return_tensors="pt")
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32),
            "class_labels": torch.tensor(class_ids, dtype=torch.int64) if class_ids else torch.empty((0,), dtype=torch.int64),
            "size": torch.tensor([height, width]),
            "image_id": self.image_files[idx]
        }

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": target
        }
class FasterRCNNDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]
        self.img_size = img_size
        self.resize = transforms.Resize((img_size, img_size))
        self.to_tensor = transforms.ToTensor()

        # Detect classes
        self.num_classes = 1
        for f in self.image_files:
            label_path = os.path.join(root_dir.replace("images", "labels"), f.rsplit(".", 1)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, "r") as file:
                    for line in file:
                        class_id = int(line.strip().split()[0])
                        self.num_classes = max(self.num_classes, class_id + 1)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.resize(image)
        tensor_img = self.to_tensor(image)
        width, height = image.size

        label_path = image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        boxes, class_ids = [], []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    class_id, cx, cy, w, h = map(float, line.strip().split())
                    boxes.append([cx, cy, w, h])
                    class_ids.append(int(class_id))

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4))
        class_ids = torch.tensor(class_ids, dtype=torch.int64) if class_ids else torch.empty((0,), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "class_labels": class_ids,
            "size": torch.tensor([height, width]),
            "image_id": self.image_files[idx]
        }

        return tensor_img, target
    
# === Failure Visualization ===
def tensor_to_pil(img_tensor):
    img_tensor = img_tensor.detach().cpu()
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.permute(1, 2, 0).numpy()
        img_tensor = (img_tensor * 255).astype(np.uint8)
        return Image.fromarray(img_tensor)
    return None

def visualize_detection_results(image, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, 
                              class_names=None, save_path=None, title="Detection Results"):
    """Visualize detection results with GT and predictions"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Convert tensor to numpy if needed
    if hasattr(image, 'numpy'):
        image = image.numpy()
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    # Normalize image if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    ax.imshow(image)
    
    # Draw ground truth boxes (green)
    for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        rect = Rectangle((x1, y1), width, height, linewidth=2, 
                        edgecolor='green', facecolor='none', linestyle='-')
        ax.add_patch(rect)
        
        label_text = f"GT: {class_names.get(label.item(), label.item()) if class_names else label.item()}"
        ax.text(x1, y1-5, label_text, color='green', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Draw prediction boxes (red)
    for i, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        rect = Rectangle((x1, y1), width, height, linewidth=2, 
                        edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        label_text = f"Pred: {class_names.get(label.item(), label.item()) if class_names else label.item()} ({score:.2f})"
        ax.text(x1, y2+15, label_text, color='red', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.tight_layout()
    return fig

def visualize_failures_comprehensive(dataloader, predictions, targets, model_name, 
                                   class_names=None, max_samples=5):
    """Comprehensive failure visualization"""
    print(f"Analyzing and visualizing failures for {model_name}...")
    
    # Analyze failures
    failures = analyze_failures(predictions, targets)
    
    # Create output directory
    output_dir = f"failure_analysis_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Print failure statistics
    print("\n" + "="*50)
    print(f"FAILURE ANALYSIS FOR {model_name}")
    print("="*50)
    for failure_type, failure_list in failures.items():
        print(f"{failure_type.replace('_', ' ').title()}: {len(failure_list)}")
    
    # Create failure statistics plot
    failure_counts = {k.replace('_', ' ').title(): len(v) for k, v in failures.items()}
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(failure_counts.keys(), failure_counts.values(), 
                   color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3'])
    plt.title(f'Failure Analysis for {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Failures', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/failure_statistics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Collect image data
    image_data = {}
    for images, batch_targets in dataloader:
        for i, target in enumerate(batch_targets):
            image_id = target.get('image_id', f"img_{len(image_data)}")
            image_data[str(image_id)] = {
                'image': images[i],
                'target': target
            }
    
    # Visualize each failure type
    for failure_type, failure_list in failures.items():
        if not failure_list:
            continue
            
        failure_dir = os.path.join(output_dir, failure_type)
        os.makedirs(failure_dir, exist_ok=True)
        
        sample_failures = failure_list[:max_samples]
        
        for idx, failure in enumerate(sample_failures):
            image_id = str(failure['image_id'])
            
            if image_id not in image_data:
                continue
                
            image = image_data[image_id]['image']
            target = image_data[image_id]['target']
            
            # Find corresponding prediction
            pred = None
            for p_idx, p in enumerate(predictions):
                if str(targets[p_idx].get('image_id', p_idx)) == image_id:
                    pred = p
                    break
            
            if pred is None:
                pred = {
                    'boxes': torch.zeros((0, 4)),
                    'labels': torch.zeros((0,), dtype=torch.int64),
                    'scores': torch.zeros((0,))
                }
            
            title = f"{failure_type.replace('_', ' ').title()} - Image {image_id}"
            save_path = os.path.join(failure_dir, f"{failure_type}_{idx}_{image_id}.png")
            
            fig = visualize_detection_results(
                image, 
                target['boxes'], 
                target['labels'],
                pred['boxes'],
                pred['labels'],
                pred['scores'],
                class_names=class_names,
                save_path=save_path,
                title=title
            )
            plt.close(fig)
    
    # Create confusion matrix for wrong class predictions
    if failures['wrong_class']:
        true_labels = []
        pred_labels = []
        
        for failure in failures['wrong_class']:
            true_labels.append(failure['gt_label'].item())
            pred_labels.append(failure['pred_label'].item())
        
        if class_names:
            unique_labels = sorted(set(true_labels + pred_labels))
            label_names = [class_names.get(i, str(i)) for i in unique_labels]
        else:
            unique_labels = sorted(set(true_labels + pred_labels))
            label_names = [str(i) for i in unique_labels]
        
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Confusion Matrix for Wrong Class Predictions - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nFailure analysis saved to: {output_dir}/")
    return failures
        
# === RT-DETR Evaluation ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_fasterrcnn_custom(model_path=None, dataset=None, batch_size=8, threshold=0.1):

    num_classes = getattr(dataset, "num_classes", 2)  
    print(f"Number of classes detected: {num_classes}")


    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"[WARNING] load_state_dict failed, trying partial load: {e}")
        # Bỏ qua head nếu mismatch
        filtered_ckpt = {k: v for k, v in checkpoint.items() if "box_predictor" not in k}
        model.load_state_dict(filtered_ckpt, strict=False)
    model = model.to(DEVICE).eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: tuple(zip(*x)), shuffle=False)

    # Measure FLOPs
    dummy_input = torch.randn(1, 3, dataset.img_size, dataset.img_size).to(DEVICE)
    try:
        flops = FlopCountAnalysis(model, (dummy_input,)).total()
        flops_gflops = round(flops / 1e9, 2)
    except Exception as e:
        print(f"[WARNING] FLOPs error: {e}")
        flops_gflops = "N/A"

    predictions, targets = [], []
    true_labels, pred_labels = [], []
    start_time = time.time()

    with torch.no_grad():
        for images, anns in tqdm(dataloader, desc="Evaluating Faster R-CNN"):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                output = outputs[i]
                ann = anns[i]
                h, w = ann["size"]

                boxes = output["boxes"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                labels = output["labels"].detach().cpu().numpy()

                conf_mask = scores >= threshold
                boxes = boxes[conf_mask]
                labels = labels[conf_mask]
                scores = scores[conf_mask]

                detections = sv.Detections(xyxy=boxes, class_id=labels, confidence=scores)
                predictions.append(detections)

                gt_boxes = ann["boxes"]
                gt_labels = ann["class_labels"]
                if gt_boxes.numel() > 0:
                    xyxy_gt = sv.xcycwh_to_xyxy(gt_boxes.cpu().numpy()) * np.array([w, h, w, h])
                    targets.append(sv.Detections(xyxy=xyxy_gt, class_id=gt_labels.cpu().numpy()))
                else:
                    targets.append(sv.Detections.empty())

                if len(labels) > 0 and len(gt_labels) > 0:
                    true_labels.extend(gt_labels.cpu().numpy().tolist())
                    pred_labels.extend(labels.tolist())

    elapsed = time.time() - start_time
    fps = len(dataset) / elapsed if elapsed > 0 else 0.0

    map_metric = sv.MeanAveragePrecision.from_detections(predictions=predictions, targets=targets)
    visualize_failures_comprehensive(dataloader, predictions, targets, model_name="FasterRCNN")

    return {
        'model': 'FasterRCNN',
        'mAP@0.5': round(map_metric.map50.item(), 4),
        'FPS': round(fps, 2),
        'VRAM_MB': torch.cuda.max_memory_allocated() // 1024**2 if torch.cuda.is_available() else 0,
        'FLOPs_GFLOPs': flops_gflops,
        'failures': None
    }
    
def eval_rtdetr_custom(model_path, dataset, batch_size=8, threshold=0.03):
    from fvcore.nn import FlopCountAnalysis
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    import supervision as sv
    import numpy as np
    import time
    import torch
    from torch.utils.data import DataLoader

    processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    model = AutoModelForObjectDetection.from_pretrained(model_path).to(DEVICE).eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

    # === Measure FLOPs ===
    dummy_image = torch.randn(1, 3, dataset.img_size, dataset.img_size).to(DEVICE)
    dummy_pixel_values = dummy_image  # Model expects pixel_values directly

    # Wrap model for correct FLOP tracing
    class WrappedRTDETR(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values)

    wrapped_model = WrappedRTDETR(model)
    with torch.no_grad():
        flops = FlopCountAnalysis(wrapped_model, (dummy_pixel_values,)).total()
        flops_gflops = round(flops / 1e9, 2)

    predictions, targets = [], []
    true_labels, pred_labels = [], []
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating RT-DETR"):
            pixel_values, labels = batch
            pixel_values = pixel_values.to(DEVICE)

            outputs = model(pixel_values=pixel_values)
            sizes = [tuple(label["size"]) for label in labels]
            results = processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=threshold)

            detections_batch = [sv.Detections.from_transformers(pred) for pred in results]
            targets_batch = []

            for gt, (h, w) in zip(labels, sizes):
                boxes = gt["boxes"]
                class_ids = gt["class_labels"]
                if boxes.numel() == 0 or class_ids.numel() == 0:
                    continue
                try:
                    boxes_np = sv.xcycwh_to_xyxy(boxes.cpu().numpy()) * np.array([w, h, w, h])
                    detections = sv.Detections(xyxy=boxes_np, class_id=class_ids.cpu().numpy())
                    targets_batch.append(detections)
                except Exception as e:
                    print(f"[WARNING] Skipping invalid GT: {e}")
                    continue

            predictions.extend(detections_batch)
            targets.extend(targets_batch)
            for p, t in zip(detections_batch, targets_batch):
                if p.class_id.size > 0 and t.class_id.size > 0:
                    pred_labels.extend(p.class_id.tolist())
                    true_labels.extend(t.class_id.tolist())

    elapsed = time.time() - start_time
    fps = len(dataset) / elapsed if elapsed > 0 else 0.0

    map_metric = sv.MeanAveragePrecision.from_detections(predictions=predictions, targets=targets)
    visualize_failures_comprehensive(dataloader, predictions, targets, model_name="RT-DETR")

    return {
        'model': 'RT-DETR',
        'mAP@0.5': round(map_metric.map50.item(), 4),
        'FPS': round(fps, 2),
        'VRAM_MB': torch.cuda.max_memory_allocated() // 1024**2 if torch.cuda.is_available() else 0,
        'FLOPs_GFLOPs': flops_gflops,
        'failures': None
    }


def eval_yolo_ultralytics(model_path, data_yaml_path, img_size=224, batch_size=8):
    model = YOLO(model_path).to(DEVICE).eval()

    # mAP eval
    metrics = model.val(data=data_yaml_path, split='test', imgsz=img_size, conf=0.001, iou=0.5, batch=batch_size)

    # Custom dataset
    dataset = RTDETRDataset(root_dir='augmented_dataset_small/test/images', img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    predictions, targets = [], []
    start_time = time.time()

    # === Measure FLOPs ===
    dummy_image = torch.randn(1, 3, img_size, img_size).to(DEVICE)

    try:
        # model.model is the actual nn.Module in YOLOv11
        wrapped_model = model.model
        with torch.no_grad():
            flops = FlopCountAnalysis(wrapped_model, (dummy_image,)).total()
            flops_gflops = round(flops / 1e9, 2)
    except Exception as e:
        print(f"[WARNING] Could not compute FLOPs for YOLOv11: {e}")
        flops_gflops = "N/A"
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating YOLOv11"):
            images = images.to(DEVICE)
            preds = model(images)[0]  # (NMS) output

            for pred, label in zip(preds, labels):
                boxes = pred.boxes.xyxy.cpu().numpy() if pred.boxes else []
                labels_pred = pred.boxes.cls.cpu().numpy() if pred.boxes else []
                scores = pred.boxes.conf.cpu().numpy() if pred.boxes else []
                detections = sv.Detections(xyxy=boxes, class_id=labels_pred.astype(int), confidence=scores)
                predictions.append(detections)

                gt_boxes = label["boxes"]
                gt_labels = label["class_labels"]
                if gt_boxes.numel() > 0 and gt_labels.numel() > 0:
                    boxes_np = sv.xcycwh_to_xyxy(gt_boxes.cpu().numpy()) * np.array([img_size]*4)
                    targets.append(sv.Detections(xyxy=boxes_np, class_id=gt_labels.cpu().numpy()))
                else:
                    targets.append(sv.Detections.empty())

    elapsed = time.time() - start_time
    fps = len(dataset) / elapsed if elapsed > 0 else 0.0

    visualize_failures_comprehensive(dataloader, predictions, targets, model_name="YOLOv11")

    return {
        'model': 'YOLOv11',
        'mAP@0.5': round(metrics.box.map50, 4),
        'FPS': round(fps, 2),
        'VRAM_MB': torch.cuda.max_memory_allocated() // 1024**2 if torch.cuda.is_available() else 0,
        'FLOPs_GFLOPs': flops_gflops,
        'failures': None
    }

# === Result Table ===
def save_comparison_table(results):
    df = pd.DataFrame(results)
    df.to_csv("evaluation_summary.csv", index=False)
    print("\n=== Evaluation Summary ===")
    print(df.to_string(index=False))

# === Main Entry ===
def main():
    img_size = 224
    batch_size = 8

    print("\n--- Evaluating YOLOv11 ---")
    yolo_result = eval_yolo_ultralytics(
        model_path="runs/train/yolov11_lr0.0001_bs10/weights/last.pt",
        data_yaml_path="augmented_dataset_small/epillID.yaml",
        img_size=img_size,
        batch_size=batch_size
    )

    print("\n--- Evaluating RT-DETR ---")
    rtdetr_dataset = RTDETRDataset(root_dir="augmented_dataset_small/test/images", img_size=img_size)
    rtdetr_result = eval_rtdetr_custom(
        model_path="checkpoints/rtdetr/lr0.00001",
        dataset=rtdetr_dataset,
        batch_size=batch_size
    )

    print("\n--- Evaluating Faster R-CNN ---")
    rtdetr_dataset = FasterRCNNDataset(root_dir="augmented_dataset_small/test/images", img_size=img_size)
    fasterrcnn_result = eval_fasterrcnn_custom(
        model_path="checkpoints/fasterrcnn_lr0,001/fasterrcnn_resnet50_epoch_50.pth",
        dataset=rtdetr_dataset,  # sử dụng chung dataset cho dễ so sánh
        batch_size=batch_size
    )

    save_comparison_table([yolo_result, rtdetr_result, fasterrcnn_result])

if __name__ == "__main__":
    main()
