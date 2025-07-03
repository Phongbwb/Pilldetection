from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

def get_ssd_model(num_classes, pretrained=True):
    model = ssd300_vgg16(pretrained=pretrained)
    in_channels = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes
    )
    return model