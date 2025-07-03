import os
import glob
import json
import random
import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from pycocotools.cocoeval import COCOeval
import re
from collections import defaultdict

# ==== Set seed
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ==== Dataset
class CocoTransform:
    def __call__(self, image, target):
        return F.to_tensor(image), target

def get_coco_dataset(img_dir, ann_file):
    return CocoDetection(root=img_dir, annFile=ann_file, transforms=CocoTransform())

def collate_fn(batch):
    return tuple(zip(*batch))

# ==== Load model
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ==== Process targets
def process_targets(targets, images, device):
    valid_images, processed_targets = [], []
    for i, target in enumerate(targets):
        boxes, labels = [], []
        for obj in target:
            x, y, w, h = obj["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(obj["category_id"])
        if boxes:
            processed_targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                "labels": torch.tensor(labels, dtype=torch.int64).to(device),
            })
            valid_images.append(images[i])
    return valid_images, processed_targets

# ==== Training function
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss, total_batches = 0.0, 0
    loop = tqdm(data_loader, desc=f" Epoch {epoch}", leave=False)
    for images, targets in loop:
        images = [img.to(device) for img in images]
        images, targets = process_targets(targets, images, device)
        if not targets:
            continue

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        loop.set_postfix(loss=f"{loss.item():.4f}")

    print(f" Epoch {epoch} Avg Loss: {total_loss / total_batches:.4f}")

# ==== Validation loss
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.train()  # Needed to compute loss
    total_loss, total_batches = 0.0, 0
    loop = tqdm(data_loader, desc=" Evaluating", leave=False)
    for images, targets in loop:
        images = [img.to(device) for img in images]
        images, targets = process_targets(targets, images, device)
        if not targets:
            continue

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        total_loss += loss.item()
        total_batches += 1
        loop.set_postfix(loss=f"{loss.item():.4f}")

    print(f" Validation Loss: {total_loss / total_batches:.4f}")

# ==== Evaluate mAP
@torch.no_grad()
def evaluate_map(model, data_loader, dataset, device):
    model.eval()
    coco_gt = dataset.coco
    coco_results = []

    print(" Running mAP evaluation...")
    for images, targets in tqdm(data_loader, desc=" Inference for mAP"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
            image_id = targets[i][0].get("image_id", i)
            for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
                x1, y1, x2, y2 = box.tolist()
                coco_results.append({
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(score.item(), 4)
                })
    if len(coco_results) == 0:
        print("No results!")
        return 0.0

    with open("coco_val_predictions.json", "w") as f:
        json.dump(coco_results, f, indent=2)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]

# ==== Resume checkpoint
def load_latest_checkpoint(model, optimizer=None, checkpoint_dir="checkpoints"):
    # Tìm tất cả các checkpoint khớp pattern
    checkpoint_files = sorted([
        f for f in glob.glob(os.path.join(checkpoint_dir, "fasterrcnn_resnet50_epoch_*_lr_*.pth"))
        if not f.endswith("_optimizer.pth")
    ])

    if not checkpoint_files:
        print(" No checkpoint found. Starting from scratch.")
        return model, optimizer, 0

    # Nhóm các checkpoint theo learning rate
    checkpoints_by_lr = defaultdict(list)
    for f in checkpoint_files:
        basename = os.path.basename(f)
        match = re.search(r"epoch_(\d+)_lr_([0-9.]+)", basename)
        if match:
            epoch = int(match.group(1))
            lr = match.group(2)
            checkpoints_by_lr[lr].append((epoch, f))

    # Lấy checkpoint có epoch lớn nhất cho mỗi lr
    latest_checkpoints = []
    for lr, ckpts in checkpoints_by_lr.items():
        latest_ckpt = max(ckpts, key=lambda x: x[0])  # theo epoch
        latest_checkpoints.append((latest_ckpt[0], lr, latest_ckpt[1]))  # (epoch, lr, filepath)

    # Chọn checkpoint có epoch lớn nhất trong tất cả lr
    final_ckpt = max(latest_checkpoints, key=lambda x: x[0])  # chọn epoch lớn nhất
    final_epoch, final_lr, final_path = final_ckpt

    print(f" Loading checkpoint from: {final_path}")
    model.load_state_dict(torch.load(final_path))

    opt_path = final_path.replace(".pth", "_optimizer.pth")
    if optimizer and os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path))
        print(f" Loaded optimizer from: {opt_path}")

    return model, optimizer, final_epoch

def trainer(model, optimizer, train_images_dir, val_images_dir, train_annotation_dir, val_annotation_dir, device, num_epochs, batch_size=4, num_workers=4):
    train_dataset = get_coco_dataset(train_images_dir, train_annotation_dir)
    val_dataset = get_coco_dataset(val_images_dir, val_annotation_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # === Model ===
    model = get_model(num_classes).to(device)

    # === Optimizer & Scheduler ===
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # === Load latest checkpoint if exists ===
    model, optimizer, start_epoch = load_latest_checkpoint(model, optimizer, "checkpoints")

    # === Train loop ===
    for epoch in range(start_epoch + 1, num_epochs + 1):
        print(f"\n Epoch {epoch}/{num_epochs}")
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        evaluate(model, val_loader, device)
        lr_scheduler.step()

        model_path = f"checkpoints/fasterrcnn_resnet50_epoch_{epoch}_lr_{lr:.4f}.pth"
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), model_path.replace(".pth", "_optimizer.pth"))
        print(f" Model saved: {model_path}")

    # === Final mAP ===
    print("\n Final mAP Evaluation")
    final_map=evaluate_map(model, val_loader, val_dataset, device)
    return final_map
# ==== MAIN
if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    lr = 0.0001
    # === Config ===
    num_classes = 2384
    batch_size = 4
    num_workers = 4
    num_epochs = 50

    # === Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    trainer(
        model=None,  # Model will be initialized inside trainer
        optimizer=None,  # Optimizer will be initialized inside trainer
        train_images_dir="augmented_dataset_small/train/images",
        val_images_dir="augmented_dataset_small/val/images",
        train_annotation_dir="augmented_dataset_small/train/annotations.json",
        val_annotation_dir="augmented_dataset_small/val/annotations.json",
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_workers=num_workers
    )
