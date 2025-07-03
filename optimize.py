import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fine_tune import fine_tune
from copy import deepcopy
from itertools import product
import pandas as pd
"""
      
            

        "retinanet": {
            "lr": [0.001, 0.0001, 0.00001],
            "batch_size": [4,8],
            "focal_alpha": [0.25, 0.5],
            
        },
        "ssd": {
            "lr": [0.001, 0.0001, 0.00001],
            "batch_size": [4,8],
            "img_size": [300, 512],
            
        },
         
           """
HYPERPARAM_GRID = {
        "yolov11": {
            "lr": [0.001, 0.0001, 0.00001],
            "batch_size": [10],
            "img_size": [416]
        }, 
        "rt_detr": {
        "lr": [0.001, 0.0001, 0.00001],
        "batch_size": [10],
        "epochs": [50],
        

    },
        "faster_rcnn": {
            "lr": [ 0.001, 0.0001, 0.00001],
            "batch_size": [10],
  
        
    }
 
}

def grid_search(model_name, param_grid):
    keys = list(param_grid.keys())
    best_map = -1
    best_config = None
    history = []

    for values in product(*[param_grid[k] for k in keys]):
        config = dict(zip(keys, values))
        if model_name == 'faster_rcnn':
            config['train_images_dir'] = "augmented_dataset_small/train/images"
            config['val_images_dir'] = "augmented_dataset_small/val/images"
            config['num_classes'] = 2384
            config['train_annotation_dir'] = "augmented_dataset_small/train/annotations.json"
            config['val_annotation_dir'] = "augmented_dataset_small/val/annotations.json"
            config['epochs'] = 50
        elif model_name == 'yolov11':
            config['data_path'] = 'augmented_dataset_small/epillID.yaml'
            config['epochs'] = 50
        elif model_name == 'rt_detr':
            config['base_dir'] = 'augmented_dataset_small'
            config['checkpoint_name'] = "PekingU/rtdetr_r50vd_coco_o365"
            config['epochs'] = 50
            

        config['name'] = f"{model_name}_lr{config['lr']}_bs{config['batch_size']}"
        
        print(f" Testing {model_name} config: {config}")
        mAP = fine_tune(model_name, config)
        print(f" mAP: {mAP}")
        torch.cuda.empty_cache()
        record = deepcopy(config)
        record["mAP"] = mAP
        history.append(record)

        if mAP > best_map:
            best_map = mAP
            best_config = deepcopy(config)

    os.makedirs("configs", exist_ok=True)
    with open(f"configs/{model_name}.yaml", "w") as f:
        yaml.dump(best_config, f)

    # Save and plot results
    df = pd.DataFrame(history)
    df.to_csv(f"configs/{model_name}_search_results.csv", index=False)
    plot_correlation(model_name, df)

    print(f" Best config for {model_name}: {best_config}, mAP: {best_map}")

def plot_correlation(model_name, df):
    os.makedirs("plots", exist_ok=True)
    for col in df.columns:
        if col == "mAP":
            continue
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[col], y=df["mAP"])
        plt.title(f"{model_name.upper()} - {col} vs mAP")
        plt.xlabel(col)
        plt.ylabel("mAP")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{model_name}_{col}_vs_mAP.png")
        plt.close()

def main():
    for model_name, grid in HYPERPARAM_GRID.items():
        print(f"\n Starting hyperparameter search for {model_name.upper()}")
        grid_search(model_name, grid)

if __name__ == "__main__":
    main()