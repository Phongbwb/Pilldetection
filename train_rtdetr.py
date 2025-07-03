import transformers
print(transformers.__version__)
print("Transformers version:", transformers.__version__)
print("TrainingArguments location:", transformers.TrainingArguments.__module__)
print(transformers.__file__)
import os
import json
import time
import shutil
import numpy as np
import supervision as sv
from datetime import timedelta
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForObjectDetection,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
)
import albumentations as A
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import TrainerCallback, TrainerControl, TrainerState
from functools import partial
from torch.nn.utils import clip_grad_norm_
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import json

with open("augmented_dataset_small/val/annotations.json", "r") as f:
    data = json.load(f)

scale_x = 480 / 224
scale_y = 480 / 224

for ann in data["annotations"]:
    x, y, w, h = ann["bbox"]
    ann["bbox"] = [x * scale_x, y * scale_y, w * scale_x, h * scale_y]

with open("augmented_dataset_small/val/annotations_scaled.json", "w") as f:
    json.dump(data, f, indent=2)

print(" Đã scale bbox lên đúng kích thước 480x480.")

def validate_coco_annotations(annotation_path):
    coco = COCO(annotation_path)
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)

    bad_boxes = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        if not all(map(np.isfinite, [x, y, w, h])):
            bad_boxes.append((ann["id"], "NaN/Inf"))
        elif w <= 0 or h <= 0:
            bad_boxes.append((ann["id"], f"width/height <= 0: w={w}, h={h}"))
        elif x < 0 or y < 0:
            bad_boxes.append((ann["id"], f"x/y < 0: x={x}, y={y}"))

    print(f"Tổng số box lỗi: {len(bad_boxes)}")
    for ann_id, reason in bad_boxes:
        print(f"Lỗi box id={ann_id}: {reason}")
        
        
class PyTorchDetectionDataset(Dataset):
    def __init__(self, dataset: sv.DetectionDataset, processor, transform=None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def annotations_as_coco(self, image_id, categories, boxes):
        annotations = []
        for category, bbox in zip(categories, boxes):
            x1, y1, x2, y2 = bbox
            annotations.append({
                "image_id": image_id,
                "category_id": category,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "iscrowd": 0,
                "area": (x2 - x1) * (y2 - y1),
            })
        return {"image_id": image_id, "annotations": annotations}

    def __getitem__(self, idx):
        try:
            _, image, annotations = self.dataset[idx]
            image = image[..., ::-1].copy()

            boxes = annotations.xyxy
            categories = annotations.class_id

            if self.transform:
                try:
                    transformed = self.transform(image=image, bboxes=boxes, category=categories)
                    image = transformed["image"]
                    boxes = transformed["bboxes"]
                    categories = transformed["category"]
                except Exception as e:
                    print(f"[Lỗi transform tại idx={idx}]: {e}")
                    boxes, categories = [], []

            # Lọc box hợp lệ
            valid_boxes, valid_cats = [], []
            for box, cat in zip(boxes, categories):
                if (
                    len(box) == 4 and
                    all(np.isfinite(box)) and
                    box[2] > box[0] and
                    box[3] > box[1]
                ):
                    valid_boxes.append([float(x) for x in box])
                    valid_cats.append(int(cat))
                else:
                    print(f"[Invalid Box] idx={idx} | box={box}, cat={cat}")

            if len(valid_boxes) == 0 or not np.isfinite(valid_boxes).all():
                valid_boxes = [[0.25, 0.25, 0.75, 0.75]]
                valid_cats = [0]

            formatted = self.annotations_as_coco(idx, valid_cats, valid_boxes)
            result = self.processor(images=image, annotations=formatted, return_tensors="pt")
            out = {k: v[0] for k, v in result.items()}

            if not torch.isfinite(out["labels"]["boxes"]).all():
                print(f"[NaN Warning] idx={idx} | boxes={out['labels']['boxes']}")
                print(f"[Fallback Box] Sẽ thay thế bằng box giả.")
                out["labels"]["boxes"] = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
                out["labels"]["class_labels"] = torch.tensor([0], dtype=torch.int64)

            return out

        except Exception as e:
            print(f"[Lỗi toàn phần __getitem__] idx={idx}: {e}")
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            formatted = self.annotations_as_coco(idx, [0], [[0.0, 0.0, 1.0, 1.0]])
            result = self.processor(images=dummy_image, annotations=formatted, return_tensors="pt")
            return {k: v[0] for k, v in result.items()}


def validate_dataset(dataset):
    for idx in range(len(dataset)):
        _, _, annotations = dataset[idx]
        boxes = annotations.xyxy
        if boxes is None or len(boxes) == 0 or np.isnan(boxes).any():
            print(f"[ERROR] Invalid boxes at index {idx}")
            
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch],
    }

def compute_metrics(eval_result, processor, id2label):
    preds, targets = eval_result.predictions, eval_result.label_ids

    image_sizes = torch.tensor([x["size"] for x in targets], device=DEVICE)

    true_boxes = []
    for batch, sizes in zip(targets, image_sizes):
        for t, (h, w) in zip(batch, sizes):
            boxes = sv.xcycwh_to_xyxy(t["boxes"]).to(DEVICE) * torch.tensor([w, h, w, h], device=DEVICE)
            labels = t["class_labels"].to(DEVICE)
            true_boxes.append({"boxes": boxes, "labels": labels})

    # Ensure predictions are also on GPU
    logits = torch.tensor(preds[1], device=DEVICE)
    pred_boxes = torch.tensor(preds[2], device=DEVICE)

    processed_preds = processor.post_process_object_detection(
        type("Output", (), {"logits": logits, "pred_boxes": pred_boxes}),
        threshold=0.01,
        target_sizes=image_sizes
    )

    # Move everything to GPU
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=False).to(DEVICE)
    metric.update(processed_preds, true_boxes)
    results = metric.compute()

    return {k: round(v.item(), 4) for k, v in results.items()}


def eval_rtdetr_mAP(model, dataset, processor, batch_size=8, threshold=0.05):
    from tqdm.auto import tqdm  # auto chọn thanh tqdm phù hợp
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)

    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating mAP"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"]  # List[Dict[str, Tensor]]

            # Inference
            outputs = model(pixel_values=pixel_values)

            # Lấy kích thước ảnh gốc cho mỗi sample
            sizes = [tuple(label["size"]) for label in labels]  # [(H, W), ...]

            # Post-process output của mô hình
            results = processor.post_process_object_detection(
                outputs, target_sizes=sizes, threshold=threshold
            )

            # Xử lý prediction
            detections_batch = [
                sv.Detections.from_transformers(pred) for pred in results
            ]

            # Xử lý ground-truth
            targets_batch = []
            for gt, (h, w) in zip(labels, sizes):
                boxes = gt["boxes"]
                class_ids = gt["class_labels"]

                # Kiểm tra dữ liệu hợp lệ
                if boxes.numel() == 0 or class_ids.numel() == 0:
                    continue

                try:
                    boxes_np = sv.xcycwh_to_xyxy(boxes.cpu().numpy()) * np.array([w, h, w, h])
                    detections = sv.Detections(
                        xyxy=boxes_np,
                        class_id=class_ids.cpu().numpy()
                    )
                    targets_batch.append(detections)
                except Exception as e:
                    print(f"[WARNING] Skipping invalid GT: {e}")
                    continue

            predictions.extend(detections_batch)
            targets.extend(targets_batch)

    # Nếu không có prediction hoặc target nào hợp lệ
    if len(predictions) == 0 or len(targets) == 0:
        print(" Không có prediction hoặc target hợp lệ.")
        return {"map50": 0.0, "map75": 0.0, "map50_95": 0.0}

    # Tính mAP
    map_metric = sv.MeanAveragePrecision.from_detections(
        predictions=predictions,
        targets=targets
    )

    print(f" mAP@50:      {map_metric.map50:.4f}")
    print(f"mAP@75:      {map_metric.map75:.4f}")
    print(f"mAP@50:95:   {map_metric.map50_95:.4f}")

    return {
        "map50": round(map_metric.map50.item(), 4),
        "map75": round(map_metric.map75.item(), 4),
        "map50_95": round(map_metric.map50_95.item(), 4),
    }
    
class CustomTrainer(Trainer):
    def training_step(self, model, inputs, num_items):
        # Kiểm tra NaN trong tất cả ảnh của batch
        labels = inputs.get("labels", [])
        for i, label in enumerate(labels):
            boxes = label.get("boxes", None)
            if boxes is not None and isinstance(boxes, torch.Tensor) and not torch.isfinite(boxes).all():
                print(f"[NaN] Bỏ qua ảnh thứ {i} trong batch. Boxes: {boxes}")
                return torch.tensor(0.0, requires_grad=True).to(model.device)

        # Huấn luyện bình thường
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

        if self.args.local_rank in [-1, 0]:
            print(f"[Grad Norm] Step {self.state.global_step}: {grad_norm.item():.4f}")

        return loss.detach()
            
    
def train_rtdetr(
    base_dir,
    checkpoint,
    output_dir,
    epochs=50,
    batch_size=16,
    lr=0.0001,
    image_size=480,
    logging="tensorboard",
    resume=True
):
    validate_coco_annotations("augmented_dataset_small/train/annotations.json")
    
    train_transform = A.Compose([
        A.NoOp()
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category"],
        clip=True,
        min_area=25,
        min_visibility=0.1,
    ))

    valid_transform = A.Compose([
        A.NoOp()
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category"],
        clip=True,
        min_area=1
    ))

    print(" Loading dataset...")
    ds_train = sv.DetectionDataset.from_coco(
        os.path.join(base_dir, "train/images"),
        os.path.join(base_dir, "train/annotations.json")
    )
    ds_val = sv.DetectionDataset.from_coco(
        os.path.join(base_dir, "val/images"),
        os.path.join(base_dir, "val/annotations_scaled.json")
    )
    validate_dataset(ds_train)
    validate_dataset(ds_val)
    num_labels = len(ds_train.classes)
    id2label = {i: name for i, name in enumerate(ds_train.classes)}
    label2id = {name: i for i, name in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(
    checkpoint,
    do_resize=True,
    size={"width": image_size, "height": image_size},
    resample=Image.BILINEAR,  # kiểm soát resize
)

    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    model.config.id2label = id2label
    model.config.label2id = label2id

    train_dataset = PyTorchDetectionDataset(ds_train, processor, transform=train_transform)
    val_dataset = PyTorchDetectionDataset(ds_val, processor, transform=valid_transform)
    
    for idx in range(len(train_dataset)):
        try:
            sample = train_dataset[idx]
            boxes = sample["labels"]["boxes"]
            if not torch.isfinite(boxes).all():
                print(f"[NaN] idx={idx} | boxes={boxes}")
        except Exception as e:
            print(f"[Lỗi đọc idx={idx}]: {e}")


    last_checkpoint = None
    if resume and os.path.isdir(output_dir):
        ckpts = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if ckpts:
            last_checkpoint = max(ckpts, key=os.path.getmtime)
            print(f" Resuming from: {last_checkpoint}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        num_train_epochs=epochs,
        save_strategy="epoch",
        eval_strategy="no",
        prediction_loss_only=False,
        learning_rate=lr,
        logging_dir=os.path.join(output_dir, "logs"),
        eval_accumulation_steps=1,
        logging_strategy="steps",
        logging_steps=200,
        remove_unused_columns=False,
        save_total_limit=2,
        metric_for_best_model="eval_map",
        max_grad_norm=1.0,
        greater_is_better=True,
        dataloader_num_workers=2,
        lr_scheduler_type="constant",
        report_to=logging,
        resume_from_checkpoint=last_checkpoint if resume else None,

    )
    for i in range(len(train_dataset)):
        item = train_dataset[i]
        boxes = item['labels']['boxes']
        if not torch.isfinite(boxes).all():
            print(f"[ERROR] NaN ở train_dataset index={i}: {boxes}")
            
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=partial(compute_metrics, processor=processor, id2label=id2label),
    )

    metrics_history = []
    class EvalLoggerCallback(TrainerCallback):
        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.is_local_process_zero and kwargs.get("metrics"):
                metrics_history.append({
                    "epoch": state.epoch,
                    "step": state.global_step,
                    "metrics": kwargs["metrics"]
                })
    trainer.add_callback(EvalLoggerCallback())

    start = time.time()
    trainer.train(resume_from_checkpoint=last_checkpoint if resume else None)
    duration = timedelta(seconds=round(time.time() - start))
    print(f" Thời gian huấn luyện: {duration}")

    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    metrics = eval_rtdetr_mAP(model, val_dataset, processor, batch_size=8, threshold=0.05)
    final_map = metrics.get("map50", 0.0)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(output_dir, "metrics_history.json"), "w") as f:
        json.dump(metrics_history, f, indent=4)

    with open(os.path.join(output_dir, "final_map.txt"), "w") as f:
        f.write(f"{final_map:.4f}\n")

    with open(os.path.join(output_dir, "train_time.txt"), "w") as f:
        f.write(str(duration))

    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        best_ckpt_file = os.path.join(output_dir, "best_checkpoint.txt")
        with open(best_ckpt_file, "w") as f:
            f.write(best_ckpt)
        print(f" Best checkpoint: {best_ckpt}")

        best_model_dir = os.path.join(output_dir, "best_model")
        if os.path.exists(best_model_dir):
            shutil.rmtree(best_model_dir)
        shutil.copytree(best_ckpt, best_model_dir)
        print(f" Copied best checkpoint to: {best_model_dir}")
    else:
        print(" Không tìm thấy checkpoint tốt nhất!")

    print(" Train kết thúc.")
    return final_map


def main():
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    train_rtdetr(
        base_dir="augmented_dataset_small",
        checkpoint="PekingU/rtdetr_r50vd_coco_o365",
        output_dir="checkpoints/rtdetr",
        epochs=50,
        batch_size=10,
        lr=0.0001,
        image_size=480,
        logging="tensorboard",
        resume=True
    )

if __name__ == "__main__":
    main()
