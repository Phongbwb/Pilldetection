# Re-run after reset

import os
import cv2
import json
import numpy as np
import random
from tqdm import tqdm

# === CONFIG ===
COCO_JSON_PATH = "dataset/dataset_small/annotations.json"
IMAGE_DIR = "dataset/dataset_small/epillIDsmall"
SAVE_IMG_DIR = "augmented/images"
SAVE_JSON_PATH = "augmented/coco_augmented.json"
AUG_PER_IMAGE = 2
IMAGE_SIZE = 640

os.makedirs(SAVE_IMG_DIR, exist_ok=True)

# === Load COCO JSON ===
with open(COCO_JSON_PATH, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# === Helper mappings ===
img_id_to_file = {img["id"]: img["file_name"] for img in images}
img_id_to_ann = {}
for ann in annotations:
    img_id_to_ann.setdefault(ann["image_id"], []).append(ann)

# === Tracking output ===
aug_images = []
aug_annotations = []
next_img_id = max(img["id"] for img in images) + 1
next_ann_id = max(ann["id"] for ann in annotations) + 1

# === Helper functions ===
def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]

def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

def flip_horizontal(image, boxes):
    flipped = cv2.flip(image, 1)
    h, w = image.shape[:2]
    new_boxes = boxes.copy()
    new_boxes[:, 0] = w - boxes[:, 2]
    new_boxes[:, 2] = w - boxes[:, 0]
    return flipped, new_boxes

def rotate_90(image, boxes):
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    h, w = image.shape[:2]
    new_boxes = []
    for x1, y1, x2, y2 in boxes:
        new_boxes.append([y1, w - x2, y2, w - x1])
    return rotated, np.array(new_boxes)

def resize_with_boxes(image, boxes, size):
    h, w = image.shape[:2]
    scale_x = size / w
    scale_y = size / h
    resized = cv2.resize(image, (size, size))
    boxes = boxes.astype(np.float32)  
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return resized, boxes


def load_image_and_boxes(img_info):
    img_id = img_info["id"]
    img_path = os.path.join(IMAGE_DIR, img_info["file_name"])
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    anns = img_id_to_ann.get(img_id, [])
    boxes = []
    labels = []
    for ann in anns:
        boxes.append(xywh_to_xyxy(ann["bbox"]))
        labels.append(ann["category_id"])
    if not boxes:
        return None, None, None
    return img, np.array(boxes), labels

# === Mosaic ===
def mosaic_augment(img_infos):
    final_img = np.zeros((IMAGE_SIZE * 2, IMAGE_SIZE * 2, 3), dtype=np.uint8)
    final_boxes = []
    final_labels = []
    yc, xc = IMAGE_SIZE, IMAGE_SIZE
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i, info in enumerate(img_infos):
        img, boxes, labels = load_image_and_boxes(info)
        if img is None:
            continue
        img, boxes = resize_with_boxes(img, boxes, IMAGE_SIZE)
        h, w = IMAGE_SIZE, IMAGE_SIZE
        row, col = positions[i]
        x1, y1 = col * IMAGE_SIZE, row * IMAGE_SIZE
        final_img[y1:y1 + h, x1:x1 + w] = img
        boxes[:, [0, 2]] += x1
        boxes[:, [1, 3]] += y1
        final_boxes.extend(boxes.tolist())
        final_labels.extend(labels)

    final_img = cv2.resize(final_img, (IMAGE_SIZE, IMAGE_SIZE))
    scale = 0.5
    final_boxes = np.array(final_boxes) * scale
    return final_img, final_boxes, final_labels

# === CutMix ===
def cutmix_augment(img1, boxes1, labels1, img2, boxes2, labels2):
    h, w = img1.shape[:2]
    lam = np.random.beta(1.0, 1.0)
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    mixed_img = img1.copy()
    mixed_img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    new_boxes = boxes1.tolist()
    new_labels = labels1[:]

    for box, label in zip(boxes2, labels2):
        bx1, by1, bx2, by2 = box
        if bx1 >= x1 and by1 >= y1 and bx2 <= x2 and by2 <= y2:
            new_boxes.append([bx1, by1, bx2, by2])
            new_labels.append(label)
    return mixed_img, np.array(new_boxes), new_labels

# === Augment Loop ===
for _ in tqdm(range(len(images) * AUG_PER_IMAGE)):
    base_info = random.choice(images)
    base_img, base_boxes, base_labels = load_image_and_boxes(base_info)
    if base_img is None:
        continue
    base_img, base_boxes = resize_with_boxes(base_img, base_boxes, IMAGE_SIZE)

    # Flip
    flip_img, flip_boxes = flip_horizontal(base_img, base_boxes.copy())
    # Rotate
    rot_img, rot_boxes = rotate_90(base_img, base_boxes.copy())
    # Mosaic
    mosaic_img_infos = random.sample(images, 4)
    mosaic_img, mosaic_boxes, mosaic_labels = mosaic_augment(mosaic_img_infos)
    # CutMix
    cut_img2_info = random.choice(images)
    img2, boxes2, labels2 = load_image_and_boxes(cut_img2_info)
    if img2 is None:
        continue
    img2, boxes2 = resize_with_boxes(img2, boxes2, IMAGE_SIZE)
    cutmix_img, cutmix_boxes, cutmix_labels = cutmix_augment(base_img, base_boxes, base_labels, img2, boxes2, labels2)

    aug_data = [
        (flip_img, flip_boxes, base_labels),
        (rot_img, rot_boxes, base_labels),
        (mosaic_img, mosaic_boxes, mosaic_labels),
        (cutmix_img, cutmix_boxes, cutmix_labels),
    ]

    for img_aug, boxes_aug, labels_aug in aug_data:
        filename = f"aug_{next_img_id}.jpg"
        cv2.imwrite(os.path.join(SAVE_IMG_DIR, filename), img_aug)
        aug_images.append({
            "id": next_img_id,
            "width": img_aug.shape[1],
            "height": img_aug.shape[0],
            "file_name": filename
        })
        for box, label in zip(boxes_aug, labels_aug):
            x1, y1, x2, y2 = clip_box(box, img_aug.shape[1], img_aug.shape[0])
            w, h = x2 - x1, y2 - y1
            if w > 1 and h > 1:
                aug_annotations.append({
                    "id": next_ann_id,
                    "image_id": next_img_id,
                    "category_id": label,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                next_ann_id += 1
        next_img_id += 1

# === Save final COCO JSON ===
final_json = {
    "images": aug_images,
    "annotations": aug_annotations,
    "categories": categories
}

os.makedirs(os.path.dirname(SAVE_JSON_PATH), exist_ok=True)
with open(SAVE_JSON_PATH, "w") as f:
    json.dump(final_json, f, indent=2)

