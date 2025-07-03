import os
import json
import shutil
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm


def convert_coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h


def write_yolo_format(coco_data, split_dir):
    yolo_dir = os.path.join(split_dir, "labels_yolo")
    os.makedirs(yolo_dir, exist_ok=True)
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    with tqdm(total=len(coco_data['annotations']), desc=f"YOLO: {os.path.basename(split_dir)}") as pbar:
        for ann in coco_data['annotations']:
            img = image_id_to_info[ann['image_id']]
            w, h = img['width'], img['height']
            x_center, y_center, bw, bh = convert_coco_bbox_to_yolo(ann['bbox'], w, h)

            yolo_path = os.path.join(yolo_dir, os.path.splitext(img['file_name'])[0] + ".txt")
            with open(yolo_path, "a") as f:
                f.write(f"{ann['category_id']} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
            pbar.update(1)


def write_voc_format(coco_data, split_dir):
    voc_dir = os.path.join(split_dir, "annotations_voc")
    os.makedirs(voc_dir, exist_ok=True)
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    anns_per_image = {}
    for ann in coco_data['annotations']:
        anns_per_image.setdefault(ann['image_id'], []).append(ann)

    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    for img_id, ann_list in tqdm(anns_per_image.items(), desc=f"VOC: {os.path.basename(split_dir)}"):
        img_info = image_id_to_info[img_id]
        filename = img_info['file_name']
        w, h = img_info['width'], img_info['height']

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "filename").text = filename
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = "3"

        for ann in ann_list:
            x, y, bw, bh = ann['bbox']
            x_min = int(x)
            y_min = int(y)
            x_max = int(x + bw)
            y_max = int(y + bh)

            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = categories[ann['category_id']]
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(x_min)
            ET.SubElement(bndbox, "ymin").text = str(y_min)
            ET.SubElement(bndbox, "xmax").text = str(x_max)
            ET.SubElement(bndbox, "ymax").text = str(y_max)

        tree = ET.ElementTree(annotation)
        xml_path = os.path.join(voc_dir, os.path.splitext(filename)[0] + ".xml")
        tree.write(xml_path)


def filter_coco(coco_dict, image_ids):
    image_ids = set(image_ids)
    new_images = [img for img in coco_dict['images'] if img['id'] in image_ids]
    new_annots = [ann for ann in coco_dict['annotations'] if ann['image_id'] in image_ids]
    return {
        "images": new_images,
        "annotations": new_annots,
        "categories": coco_dict['categories']
    }


def export_dataset(split_name, coco_data, img_dir, output_dir):
    split_img_dir = os.path.join(output_dir, split_name, "images")
    os.makedirs(split_img_dir, exist_ok=True)

    for img in tqdm(coco_data['images'], desc=f"Copying {split_name} images"):
        src_path = os.path.join(img_dir, img['file_name'])
        dst_path = os.path.join(split_img_dir, img['file_name'])
        shutil.copyfile(src_path, dst_path)

    with open(os.path.join(output_dir, split_name, "annotations.json"), "w") as f:
        json.dump(coco_data, f)


def convert_all_splits_to_yolo_and_voc(base_dir="split_coco"):
    for split in ["train", "val", "test"]:
        ann_path = os.path.join(base_dir, split, "annotations.json")
        if not os.path.exists(ann_path):
            continue
        with open(ann_path, "r") as f:
            coco_data = json.load(f)

        split_dir = os.path.join(base_dir, split)
        write_yolo_format(coco_data, split_dir)
        write_voc_format(coco_data, split_dir)


def coco_split_train_val_test(
    augmented_json, augmented_img_dir,
    original_json, original_img_dir,
    output_dir="split_coco"
):
    os.makedirs(output_dir, exist_ok=True)

    with open(augmented_json, "r") as f:
        aug_coco = json.load(f)
    with open(original_json, "r") as f:
        ori_coco = json.load(f)

    aug_images = {img['id']: img for img in aug_coco['images']}
    ori_images = {img['id']: img for img in ori_coco['images']}

    class_to_ori_images = defaultdict(list)
    for ann in ori_coco['annotations']:
        class_to_ori_images[ann['category_id']].append(ann['image_id'])

    val_ids, test_ids = set(), set()
    for cat_id, img_ids in class_to_ori_images.items():
        img_ids = list(set(img_ids))[:2]
        if len(img_ids) >= 2:
            random.shuffle(img_ids)
            val_ids.add(img_ids[0])
            test_ids.add(img_ids[1])
        elif len(img_ids) == 1:
            val_ids.add(img_ids[0])

    train_ids = set(img['id'] for img in aug_coco['images'])

    train_coco = filter_coco(aug_coco, train_ids)
    val_coco = filter_coco(ori_coco, val_ids)
    test_coco = filter_coco(ori_coco, test_ids)

    export_dataset("train", train_coco, augmented_img_dir, output_dir)
    export_dataset("val", val_coco, original_img_dir, output_dir)
    export_dataset("test", test_coco, original_img_dir, output_dir)

    convert_all_splits_to_yolo_and_voc(output_dir)

if __name__ == "__main__":
    coco_split_train_val_test(
    augmented_json="augmented/coco_augmented.json",
    augmented_img_dir="augmented/images",
    original_json="dataset/dataset_small/annotations.json",
    original_img_dir="dataset/dataset_small/epillIDsmall",
    output_dir="augmented_dataset_small"
)