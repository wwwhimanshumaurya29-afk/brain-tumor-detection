
import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=3, pretrained=True):
    """
    Create a ResNet18 model for 3-class brain tumor classification.
    """

    weights = (
        models.ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.resnet18(weights=weights)

    # Replace the final classification layer
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model


if __name__ == "__main__":
    model = create_model(num_classes=3, pretrained=True)

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    print("RESNET18 MODEL TEST")
    print("==============================")
    print("Architecture: ResNet18")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Number of classes:", output.shape[1])
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("Model test successful!")
