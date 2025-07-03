import os
import yaml
from glob import glob

def extract_label(filename):
    # Ví dụ: "0002-3228_0_0.jpg" → "0002-3228"
    return filename.split("_")[0]

def generate_label2id(image_dir):
    label_set = set()
    for img_path in glob(os.path.join(image_dir, "*.jpg")):
        filename = os.path.basename(img_path)
        label = extract_label(filename)
        label_set.add(label)

    sorted_labels = sorted(list(label_set))
    return {label: idx for idx, label in enumerate(sorted_labels)}

def create_yolo_yaml(label2id, output_path="augmented_dataset_small/epillID.yaml"):
    yaml_dict = {
        'path': 'augmented_dataset_small',
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {v: k for k, v in label2id.items()}
    }

    with open(output_path, 'w') as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
    print(f" YAML file generated at: {output_path}")

if __name__ == "__main__":
    image_dir = "dataset/dataset_small/epillIDsmall"
    label2id = generate_label2id(image_dir)
    create_yolo_yaml(label2id)
