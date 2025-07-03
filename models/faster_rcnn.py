from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_faster_rcnn_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Xác định số đặc trưng đầu vào của lớp phân loại hiện tại
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Thay thế lớp phân loại bằng số lớp mong muốn
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
