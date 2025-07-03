from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

def get_retinanet_model(num_classes, pretrained=True):
    model = retinanet_resnet50_fpn(pretrained=pretrained)
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_features, num_anchors=num_anchors, num_classes=num_classes
    )
    return model