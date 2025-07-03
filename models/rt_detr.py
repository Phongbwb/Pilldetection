from transformers import AutoModelForObjectDetection

def get_rt_detr_model(num_classes, pretrained=True):
    model_name = "IDEA-Research/RT-DETR"  # Replace with specific model version
    model = AutoModelForObjectDetection.from_pretrained(model_name, num_labels=num_classes)
    return model