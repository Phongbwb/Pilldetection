
# Object Detection Evaluation: YOLOv11 vs RT-DETR vs Faster R-CNN

## Mục tiêu

Dự án này cung cấp một pipeline đầy đủ để:
- Huấn luyện và đánh giá mô hình **Faster R-CNN** trên tập COCO.
- Đánh giá hiệu năng các mô hình:
  - [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
  - [RT-DETR (from HuggingFace)](https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365)
  - [Faster R-CNN (TorchVision)]
- So sánh hiệu quả theo các chỉ số:
  - `mAP@0.5`
  - `FPS`
  - `VRAM`
  - `FLOPs`
  - Phân tích lỗi chi tiết (false positives, false negatives, wrong class...)

## Cấu trúc thư mục

```
.
├── augmented_dataset_small/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── epillID.yaml
├── checkpoints/
│   ├── fasterrcnn_*.pth
│   └── rtdetr/
├── runs/train/yolov11_lr.../weights/
│   └── last.pt
├── failure_analysis_YOLOv11/
├── failure_analysis_RT-DETR/
├── failure_analysis_FasterRCNN/
├── evaluation_summary.csv
├── train_fasterrcnn.py
├── metric_model.py   <-- file chính để chạy so sánh
└── README.md
```

## Yêu cầu hệ thống

- Python 3.8+
- GPU (khuyến nghị có CUDA)
- Thư viện:
  ```bash
  pip install -r requirements.txt
  ```

<details>
<summary><strong>Yêu cầu gợi ý trong <code>requirements.txt</code></strong></summary>

```txt
torch>=1.12
torchvision
ultralytics
transformers
supervision
fvcore
tqdm
numpy
matplotlib
seaborn
pandas
scikit-learn
pycocotools
Pillow
```

</details>

## Cách sử dụng

### 1. Huấn luyện Faster R-CNN (nếu cần)
```bash
python train_fasterrcnn.py
```
> Ghi chú: Model được lưu tại thư mục `checkpoints/`.

### 2. Chạy đánh giá và so sánh tất cả mô hình:
```bash
python metric_model.py
```

### 3. Kết quả
- File CSV tổng hợp: `evaluation_summary.csv`
- Biểu đồ lỗi từng mô hình: `failure_analysis_<model_name>/`
- mAP và confusion matrix: lưu trong các thư mục tương ứng.

## Kết quả đầu ra ví dụ (trích từ `evaluation_summary.csv`)

| model       | mAP@0.5 | FPS   | VRAM_MB | FLOPs_GFLOPs |
|-------------|---------|-------|---------|---------------|
| YOLOv11     | 0.7311  | 23.28 | 2627    | 0.64          |
| RT-DETR     | 0.0113  | 23.9  | 2629    | 12.3          |
| FasterRCNN  | 0.0047  | 17.55 | 2958    | 146.14        |

## Phân tích lỗi

- Các loại lỗi được phân tích:
  - False Positives
  - False Negatives
  - Wrong Class
  - Localization Errors
  - Duplicate Detections
  - Low Confidence

- Hình ảnh lỗi và thống kê được lưu tại:
  - `failure_analysis_<model>/failure_statistics.png`
  - `failure_analysis_<model>/*.png`

## Tuỳ chỉnh

- Thay đổi tập dữ liệu đầu vào trong `augmented_dataset_small/`
- Chỉnh `img_size`, `batch_size`, `conf_threshold` trong file `metric_model.py`

## Ghi chú

- Dataset yêu cầu cấu trúc `YOLO-format` (`.txt` label theo `[class_id, cx, cy, w, h]`)
- Nếu bạn dùng định dạng COCO, cần convert sang YOLO format để phù hợp.

## Liên hệ

Mọi thắc mắc hoặc đóng góp vui lòng liên hệ [nguyenthephongctn@gamil.com].
