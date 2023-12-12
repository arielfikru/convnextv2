import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.convnext import ConvNeXt_Small_Weights
from config import Config

def load_convnext_model():
    """Load a pre-trained ConvNeXT model and modify it for fine-tuning."""
    weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
    model = models.convnext_small(weights=weights)

    # Replace the classifier head with a new one for fine-tuning
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, Config.NUM_CLASSES)

    return model

if __name__ == "__main__":
    model = load_convnext_model()
    print(model)
