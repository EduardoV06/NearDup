
import torchvision.models as tv_models
from torch import hub
from transformers import AutoModel, AutoImageProcessor
from .helpers import format_weights_name
from os import path
from yaml import safe_load

def load_model(config):
    source = config.get("source")
    function_name = config.get("function")

    if source == "torchvision":
        model_fn = getattr(tv_models, function_name)
        weights = config.get("weights", None)

        if weights:
            weights_attr = weights.split(".")[-1]  # e.g., IMAGENET1K_V1
            try:
                weights = getattr(tv_models, format_weights_name(function_name) + "_Weights")[weights_attr]
            except AttributeError:
                print(f"Warning: Weights attribute not found for {function_name}. Using default weights.")
                weights = None
        
        # Ensure DEFAULT weights are used if no valid weights were provided
        if weights is None:
            weights = getattr(tv_models, format_weights_name(function_name) + "_Weights").DEFAULT
        
        model = model_fn(weights=weights)
    elif source == "torchhub":
        repo = config.get("repo")
        model = hub.load(repo, function_name)
    elif source == "huggingface":
        repo = config.get("repo")
        model_ = AutoModel.from_pretrained(repo)
        processor = AutoImageProcessor.from_pretrained(repo)
        model = (model_, processor)
    else:
        raise ValueError(f"Unknown source: {source}")
    return model

def load_models_from_yaml(yaml_file):
    """Load and return models configuration from the YAML file."""
    if not path.exists(yaml_file):
        raise ValueError(f"The YAML file '{yaml_file}' does not exist.")
    with open(yaml_file, "r") as f:
        cfg = safe_load(f)
    return cfg