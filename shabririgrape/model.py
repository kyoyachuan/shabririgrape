from torch import nn
from torchvision import models

from . import MODEL_TYPE, DATASET


def get_model(model_name: str, pretrained_mode: str) -> nn.Module:
    """
    Get model.

    Args:
        model_name (str): Model name.
        pretrained_mode (str): Pretrained mode.

    Raises:
        ValueError: If model_name is not resnet18 or resnet50.

    Returns:
        nn.Module: Model.
    """
    if model_name == MODEL_TYPE.RESNET18:
        weights = None if pretrained_mode == MODEL_TYPE.WITHOUT_PRETRAINED else models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    elif model_name == MODEL_TYPE.RESNET50:
        weights = None if pretrained_mode == MODEL_TYPE.WITHOUT_PRETRAINED else models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f'We only allow the following models for experiment: '
                         f'{MODEL_TYPE.RESNET18} and {MODEL_TYPE.RESNET50}')

    if pretrained_mode == MODEL_TYPE.FIXED_PRETRAINED:
        for param in model.parameters():
            param.requires_grad = False

    last_layers = model.fc.in_features
    model.fc = nn.Linear(last_layers, DATASET.NUM_CLASSES)

    return model
