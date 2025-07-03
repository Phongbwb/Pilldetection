import os
import csv
import json
import tempfile
import logging
from datetime import datetime
import torch
from train_fasterrcnn import trainer
from train_rtdetr import train_rtdetr
from trainer_yolo import train_yolov11

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fine_tune(model_name, config):
    if model_name == "yolov11":
        return train_yolov11(config)
    elif model_name == "rt_detr":
        return train_rtdetr(
            base_dir=config["base_dir"],
            checkpoint_name=config["checkpoint_name"],
            output_dir=os.path.join("checkpoints", config["name"]),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"]
        )
    elif model_name == "faster_rcnn":
        return trainer(
            train_images_dir=config["train_images_dir"],
            val_images_dir=config["val_images_dir"],
            train_annotation_dir=config["train_annotation_dir"],
            val_annotation_dir=config["val_annotation_dir"],
            device=DEVICE,
            num_epochs=config["num_epochs"],
            batch_size=config["batch_size"],
            num_workers=config.get("num_workers", 4)
        )

def main():
    logging.basicConfig(level=logging.INFO)
    config = {
        "data_path": "epillID_2.yaml",
        "name": "yolov11_finetuned",
        "img_size": 416,
        "epochs": 50,
        "batch_size": 8,
        "lr": 0.0001,
        "num_classes": 2384  
    }
    
    model_name = "yolov11"  
    fine_tune(model_name, config)

if __name__ == "__main__":
    main()