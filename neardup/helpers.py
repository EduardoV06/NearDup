# utils.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from os import listdir

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


def compute_embedding(model, image_tensor):
    """Compute embeddings using the provided model.
       For Hugging Face models, image_tensor is expected to be a PIL image.
    """
    with torch.inference_mode():
        if hasattr(model, 'forward_features'):
            model.eval()
            return model.forward_features(image_tensor)
        elif isinstance(model, tuple):  # Hugging Face model (model, processor)
            model_, processor = model
            return compute_hf_embedding(model_, processor, image_tensor)
        else:
            model.eval()
            return model(image_tensor).squeeze(-1).squeeze(-1)

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
    
