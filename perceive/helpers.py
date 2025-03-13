# utils.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from os import listdir
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, Union


# def extract_features_torchvision(model, image_tensor):
#     """Extract features from torchvision models (e.g., ResNet, EfficientNet)."""
#     model.eval()
#     with torch.inference_mode():
#         features = model(image_tensor)
#     return features.squeeze(-1).squeeze(-1)

# def extract_features_dino(model, image_tensor):
#     """Extract features from DINO models (TorchHub)."""
#     model.eval()
#     with torch.inference_mode():
#         features = model(image_tensor)
#     return features

# def extract_features_huggingface(model, image_tensor):
#     """Extract features from Hugging Face models (e.g., ViT, CLIP)."""
#     model.eval()
#     with torch.inference_mode():
#         outputs = model(image_tensor)
#         return outputs.last_hidden_state.mean(dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_weights_name(function_name):
    """Format the weights attribute name from the function name."""
    return "_".join(map(str.capitalize, function_name.split("_"))).replace("net", "Net")

def compute_hf_embedding(model_fn, processor, image):
    """Compute the embedding for a Hugging Face model with a processor."""
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_fn(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

@lru_cache(maxsize=500)
def compute_embedding(model, image_tensor):
    """Compute embeddings using the provided model."""
    with torch.inference_mode():
        model.eval()
        
        # Modelos Vision Transformer ou Swin (transformers do Torch)
        if hasattr(model, 'forward_features'):
            return model.forward_features(image_tensor)
        
        # Modelos do Hugging Face
        elif isinstance(model, tuple):  # (modelo, processor)
            model_, processor = model
            return compute_hf_embedding(model_, processor, image_tensor)

        # Modelos padrão do TorchVision
        elif hasattr(model, 'global_pool'):  # EfficientNet, MobileNet, etc.
            x = model.features(image_tensor)
            return model.global_pool(x)

        elif hasattr(model, 'avgpool'):  # ResNet, DenseNet, etc.
            x = model.conv1(image_tensor)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            return torch.flatten(x, 1)  # Flatten para vetor de embedding

        else:
            raise ValueError("Modelo não reconhecido para extração de embeddings!")

class ImageFolderDataset(Dataset):
    """Custom Dataset for loading images from a folder."""
    def __init__(self, folder_path, preprocess_fn):
        self.image_paths = [Path(folder_path) / f for f in listdir(folder_path)]
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.preprocess_fn(image_path)
        return image
    
@dataclass
class Model:
    source: str
    function: str
    weights: Union[str, None] = None
    repo: Union[str, None] = None

    def __post_init__(self):
        if self.source.lower() not in ["torchvision", "torchhub", "huggingface"]:
            raise TypeError("Model's source must be from either torchvision, torchhub or huggingface")

@dataclass
class Yaml_Schema:
    __slots__ = ['base_path', 'models']
    base_path: Union[str, Path]
    models: Dict[str, Model]
    def __post_init__(self):
        if not isinstance(self.base_path, (str, Path)):
            raise ValueError(f"base_path must be a string or a Path, got {type(self.base_path)}")