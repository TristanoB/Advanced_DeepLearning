import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
from torchmetrics.classification import ConfusionMatrix

def precompute_features(
    model: models.ResNet, dataset: torch.utils.data.Dataset, device: torch.device
) -> torch.utils.data.Dataset:
    """
    Create a new dataset with the features precomputed by the model.

    If the model is $f \circ g$ where $f$ is the last layer and $g$ is
    the rest of the model, it is not necessary to recompute $g(x)$ at
    each epoch as $g$ is fixed. Hence you can precompute $g(x)$ and
    create a new dataset
    $\mathcal{X}_{\text{train}}' = \{(g(x_n),y_n)\}_{n\leq N_{\text{train}}}$

    Arguments:
    ----------
    model: models.ResNet
        The model used to precompute the features
    dataset: torch.utils.data.Dataset
        The dataset to precompute the features from
    device: torch.device
        The device to use for the computation

    Returns:
    --------
    torch.utils.data.Dataset
        The new dataset with the features precomputed
    """
    features_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    features_extractor.eval()
    all_features = []
    all_labels = []
    # iterate over the old dataset
    with torch.no_grad():
        for images, labels in dataset: 
            images = images.to(device)
            labels = torch.as_tensor(labels, device=device)
            features = features_extractor(images.unsqueeze(0))  
            all_features.append(features.squeeze(0))  
            all_labels.append(labels)
    all_features = torch.stack(all_features)  # concat the features
    all_labels = torch.stack(all_labels)
    return torch.utils.data.TensorDataset(all_features.cpu(), all_labels.cpu())
