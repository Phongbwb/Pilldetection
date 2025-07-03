from ultralytics import YOLO

def get_yolov11_model(num_classes, pretrained=True):
    model_path = "yolov11.pt" if pretrained else "yolov11-scratch.pt"
    model = YOLO(model_path)
    model.model.model[-1].nc = num_classes  # Update number of classes
    model.model.names = [f"class_{i}" for i in range(num_classes)]
    return model