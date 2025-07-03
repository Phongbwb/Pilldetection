import os
import shutil
from transformers import AutoModelForObjectDetection

checkpoint_dir = "checkpoints/rtdetr/checkpoint-7630"
temp_dir = checkpoint_dir + "_tmp"

# Bước 1: Load model
model = AutoModelForObjectDetection.from_pretrained(checkpoint_dir)

# Bước 2: Save model to temp dir as .safetensors
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir, exist_ok=True)

print(f"[INFO] Saving to temp dir: {temp_dir}")
model.save_pretrained(temp_dir, safe_serialization=True)

# Bước 3: Move .safetensors back to original checkpoint dir
src = os.path.join(temp_dir, "model.safetensors")
dst = os.path.join(checkpoint_dir, "model.safetensors")

# Delete existing .safetensors or .bin if needed
for f in ["pytorch_model.bin", "model.safetensors"]:
    path = os.path.join(checkpoint_dir, f)
    if os.path.exists(path):
        print(f"[INFO] Removing old file: {path}")
        os.remove(path)

shutil.copy(src, dst)
print(f"[SUCCESS] Copied .safetensors to: {dst}")

# Bước 4: Clean up
shutil.rmtree(temp_dir)
print("[INFO] Temp folder cleaned up.")