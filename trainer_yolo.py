from ultralytics import YOLO
import argparse
import os
import pandas as pd
import evaluate

def train_yolov11(config):
    run_dir = os.path.join("runs/train", config['name'])
    checkpoint_path = os.path.join(run_dir, "weights", "last.pt")

    # Nếu checkpoint tồn tại thì resume
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        model.train(resume=True)
    else:
        print(f"Starting new training with weights: {config.get('weights', 'yolo11n.pt')}")
        model = YOLO(config.get("weights", "yolo11n.pt"))
        model.train(
            data=config['data_path'],
            epochs=config['epochs'],
            imgsz=config['img_size'],
            batch=config['batch_size'],
            name=config['name'],
            project='runs/train',
            save=True,
            save_period=1,
            augment=False
        )
    print(" Training YOLOv11 completed.")

    # Try to read results.csv to get mAP
    results_csv = os.path.join("runs/train", config['name'], "results.csv")
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        if 'metrics/mAP_0.5:0.95' in df.columns:
            final_map = df['metrics/mAP_0.5:0.95'].iloc[-1]
            print(f" Final mAP@0.5:0.95 = {final_map}")
            return final_map
        elif 'metrics/mAP50-95' in df.columns:
            final_map = df['metrics/mAP50-95'].iloc[-1]
            print(f" Final mAP@0.5:0.95 = {final_map}")
            return final_map
        else:
            print(" mAP column not found in results.csv.")
    else:
        print(" results.csv not found.")

    #return 0.0  # fallback nếu không đọc được

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="yolov11n_run")
    parser.add_argument("--weights", type=str, default="yolov11n.pt", help="Pretrained weights path")
    args = parser.parse_args()

    map_score = train_yolov11(vars(args))
    print(f"Returned mAP: {map_score}")
