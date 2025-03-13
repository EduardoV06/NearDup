# core.py
from .models import load_model
from .preprocess import load_and_preprocess_image, process_image_safely
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from os import listdir
from PIL import Image
from .models import load_models_from_yaml  # Load inside function to avoid circular imports
from .helpers import ImageFolderDataset, compute_hf_embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_similarity_imgs(model_name, img1_path, img2_path, similarity_metric="cosine", yaml_path="models.yaml"):
    """Compute similarity between two images using a selected model."""

    cfg = load_models_from_yaml(yaml_path)
    model_obj = cfg.models.get(model_name)
    if model_obj is None:
        raise ValueError(f"Unknown model: {model_name}")

    model = load_model(model_obj)

    # Handling Hugging Face models
    if isinstance(model, tuple):
        model_fn = model[0]
        processor = model[1]

        # Open images
        img1 = process_image_safely(img1_path)
        img2 = process_image_safely(img2_path)

        # Compute embeddings using the helper function
        embedding1 = compute_hf_embedding(model_fn, processor, img1)
        embedding2 = compute_hf_embedding(model_fn, processor, img2)

        if similarity_metric == "cosine":
            similarity = F.cosine_similarity(embedding1, embedding2, dim=1).item()

        return similarity
    # Handling standard PyTorch models
    model.to(device)

    # Preprocess and move image tensors to the device
    image1 = load_and_preprocess_image(img1_path).to(device)
    image2 = load_and_preprocess_image(img2_path).to(device)

    # Compute embeddings and similarity
    embedding1 = model(image1)
    embedding2 = model(image2)

    return F.cosine_similarity(embedding1, embedding2, dim=1).item()




def compute_similarity_img_folder(model_name, img_path, folder_path, similarity_metric="cosine", yaml_path="models.yaml"):
    """
    Compute cosine similarity between a single target image and all images in a folder.
    Supports both standard PyTorch models and Hugging Face models (returned as (model_fn, processor)).
    """
    # Load model configuration and model
    cfg = load_models_from_yaml(yaml_path)
    model_obj = cfg.models.get(model_name)
    if model_obj is None:
        raise ValueError(f"Unknown model: {model_name}")
    model = load_model(model_obj)

    # Process the target image
    if isinstance(model, tuple):
        # Hugging Face style: model is (model_fn, processor)
        model_fn, processor = model
        target_img = process_image_safely(img_path)

        target_embedding = compute_hf_embedding(model_fn, processor, target_img)  # shape: (1, hidden_dim)
    else:
        # Standard PyTorch model; assume load_and_preprocess_image returns a tensor.
        image_tensor = load_and_preprocess_image(img_path)
        with torch.no_grad():
            target_embedding = model(image_tensor)  # shape: (1, emb_dim)

    # Process folder images
    if isinstance(model, tuple):
        # Hugging Face: process images one-by-one (processor may not support batching easily)
        model_fn, processor = model
        folder_paths = [str(Path(folder_path) / f) for f in listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        embeddings_list = []
        for img_file in folder_paths:
            img = process_image_safely(img_file)

            embedding = compute_hf_embedding(model_fn, processor, img)
            embeddings_list.append(embedding)
        folder_embeddings = torch.cat(embeddings_list, dim=0)
    else:
        # Standard model: use DataLoader for batch processing
        folder_dataset = ImageFolderDataset(folder_path, load_and_preprocess_image)
        folder_loader = DataLoader(folder_dataset, batch_size=32, shuffle=False)
        embeddings_list = []
        with torch.no_grad():
            for batch in folder_loader:
                embeddings = model(batch.squeeze(0))
                embeddings_list.append(embeddings)
        folder_embeddings = torch.cat(embeddings_list, dim=0)

    # Compute cosine similarity between the target and each folder image (vectorized)
    if similarity_metric == "cosine":
        similarity = F.cosine_similarity(target_embedding, folder_embeddings, dim=1, eps=1e-6)
    return similarity.numpy()


def compute_similarity_two_folders(model_name, folder1_path, folder2_path, similarity_metric="cosine", yaml_path="models.yaml"):
    """
    Compute pairwise cosine similarity between all images in folder1 and folder2.
    Supports both standard PyTorch models and Hugging Face models (returned as (model_fn, processor)).
    Returns a similarity matrix of shape (len(folder1), len(folder2)).
    """
    # Load model configuration and model
    cfg = load_models_from_yaml(yaml_path)
    model_obj = cfg.models.get(model_name)
    if model_obj is None:
        raise ValueError(f"Unknown model: {model_name}")
    model = load_model(model_obj)

    # Helper function to get embeddings from a folder
    def get_folder_embeddings_hf(folder_path):
        if isinstance(model, tuple):
            # Hugging Face: process images one-by-one
            model_fn, processor = model
            img_paths = [str(Path(folder_path) / f) for f in listdir(folder_path)]
            embeddings_list = []
            for img_file in img_paths:
                img = process_image_safely(img_file)
                embedding = compute_hf_embedding(model_fn, processor, img)
                embeddings_list.append(embedding)
            return torch.cat(embeddings_list, dim=0)
        else:
            # Standard model: use DataLoader for batching
            dataset = ImageFolderDataset(folder_path, load_and_preprocess_image)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            embeddings_list = []
            with torch.no_grad():
                for batch in loader:
                    embeddings = model(batch.squeeze(0))
                    embeddings_list.append(embeddings)
            return torch.cat(embeddings_list, dim=0)

    # Get embeddings for both folders
    folder1_embeddings = get_folder_embeddings_hf(folder1_path)
    folder2_embeddings = get_folder_embeddings_hf(folder2_path)

    # Compute pairwise cosine similarity:
    # Expand dims to compute cosine similarity between every pair:
    if similarity_metric == "cosine":
        similarity_matrix = F.cosine_similarity(
            folder1_embeddings.unsqueeze(1),  # shape: (N, 1, emb_dim)
            folder2_embeddings.unsqueeze(0),  # shape: (1, M, emb_dim)
            dim=2, eps=1e-6
        )
    return similarity_matrix.numpy()