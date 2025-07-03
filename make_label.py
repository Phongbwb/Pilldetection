import os
import json

image_dir = "dataset/dataset_small/epillIDsmall"
output_json_path = "dataset/dataset_small/annotations.json"

# Bắt đầu tạo dữ liệu COCO
coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

class_name_to_id = {}
next_class_id = 0
annotation_id = 0
image_id = 0

for filename in os.listdir(image_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        # Lấy class từ tên file
        class_name = filename.split('_')[0]
        if class_name not in class_name_to_id:
            class_name_to_id[class_name] = next_class_id
            next_class_id += 1

        class_id = class_name_to_id[class_name]

        # Thêm vào categories nếu chưa có
        if not any(c['id'] == class_id for c in coco['categories']):
            coco['categories'].append({
                "id": class_id,
                "name": class_name
            })

        # Thêm ảnh vào "images"
        coco['images'].append({
            "id": image_id,
            "file_name": filename,
            "width": 224,
            "height": 224
        })

        # Thêm annotation: bbox full ảnh
        coco['annotations'].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_id,
            "bbox": [0, 0, 224, 224],
            "area": 224 * 224,
            "iscrowd": 0
        })

        annotation_id += 1
        image_id += 1

# Ghi ra file JSON
with open(output_json_path, "w") as f:
    json.dump(coco, f, indent=4)

print(" Done!")

# In mapping class → id
print("\nClass to ID mapping:")
for name, idx in class_name_to_id.items():
    print(f"{name}: {idx}")
