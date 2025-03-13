
import torchvision.models as tv_models
from torch import hub
from transformers import AutoModel, AutoImageProcessor, logging
from .helpers import format_weights_name, Yaml_Schema
from os import path
from yaml import safe_load
from pathlib import Path

logging.set_verbosity_error()

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
        processor = AutoImageProcessor.from_pretrained(repo, use_fast=True)
        model = (model_, processor)
    else:
        raise ValueError(f"Unknown source: {source}")
    return model

def load_models_from_yaml(yaml_file: str) -> Yaml_Schema:
    """Load and return models configuration from the YAML file."""
    
    if not Path(yaml_file).exists():
        raise ValueError(f"The YAML file '{yaml_file}' does not exist.")
    
    with open(yaml_file, "r") as f:
        try:
            cfg = safe_load(f)
        except Exception as e:
            raise ValueError(f"Error loading YAML file '{yaml_file}': {str(e)}")

    try:
        valid_cfg = Yaml_Schema(**cfg)
    except Exception as e:
        raise ValueError(f"Error validating the configuration: {str(e)}")
    
    return valid_cfg