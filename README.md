# Perceive: A CLI Tool and Python Library for Image Similarity Comparison

## Overview
Perceive is a lightweight, customizable tool for comparing image similarity. It is designed to be used both as a command-line interface (CLI) and as a Python library. With support for multiple backbone models (via Torchvision, Hugging Face, or TorchHub) and a YAML-driven configuration, Perceive emphasizes efficiency, scalability, and ease of use.

## Key Features
- **Customizable Model Selection:** Define and load any pre-trained model by configuring a YAML file.
- **Flexible Input Handling:** Compare two images, an image against a folder, or two folders.
- **Multiple Similarity Metrics:** Compute cosine similarity and more.
- **Output Options:** Display results on the console or save them as `.npz`/`.pickle` files.
- **Library & CLI Integration:** Use core functions directly in your Python projects or via a ready-to-use command-line tool.

## Installation
Perceive will be distributed via various channels:

### Homebrew (Planned)
```sh
brew install Perceive
```

### APT (Planned)
```sh
sudo apt-get install Perceive
```

### PyPI
```
pip install perceive
```


### Manual Installation
Clone the repository and install dependencies:
```sh
git clone https://github.com/yourrepo/perceive.git
cd perceive
pip install -r requirements.txt
```

## QuickStart

### As a Python Library
For example, to compare two image folders directly from your code:
```python
import perceive as pc 


# Compare two folders directly
similarity_matrix = pc.compute_similarity_two_folders(
    model_name="clip",
    folder1_path="path/to/folder1",
    folder2_path="path/to/folder2",
    yaml_path="models.yaml"
)
print("Folder vs. Folder similarity matrix:", similarity_matrix)
```

### As a CLI Tool
Perceive also provides a CLI interface. The tool automatically selects the appropriate comparison mode based on the provided arguments.

#### Examples

- **Basic Image Comparison (Image vs. Image):**
  ```sh
  perceive -f image1.jpg image2.jpg -m vit_hf -yp models.yaml
  ```

- **Compare an Image Against a Folder (Image vs. Folder):**
  ```sh
  perceive -f image.jpg -d folder_path -m resnet -yp models.yaml
  ```

- **Compare Two Folders (Folder vs. Folder):**
  ```sh
  perceive -d folder1 folder2 -m clip -yp models.yaml -o npz -of results.npz
  ```

> **Note:**  
> - The `-f/--files` option accepts one or two image files.  
> - The `-d/--directory` option accepts one or two paths. Use one path to compare an image (from `-f`) against a folder or two paths to compare two folders.
> - The `-o/--output` option lets you choose to print the results or save them (as `.npz` or `.pickle`).

## Usage Details
Perceive's CLI automatically determines the comparison mode:
- **Two image files (-f):** Compares the images directly.
- **One image file (-f) and one folder (-d):** Compares the image against all images in the folder.
- **Two folders (-d):** Computes a similarity matrix between images in both folders.
  
Additional options such as model selection (`-m`), YAML configuration (`-yp`), similarity metric (`-sm`), and output formatting (`-o` and `-of`) are available for fine-tuning.

## Configuration (YAML Schema)
Perceive uses a YAML configuration file to manage model selection and paths. This allows you to define models, specify weights, and set a `base_path` to simplify file references.

```yaml
base_path: "/Users/yourname/dataset"  # Optional: define a root directory for images

models:
  resnet:
    source: "torchvision"
    function: "resnet50"
    weights: "ResNet50_Weights.DEFAULT"
  vit_hf:
    source: "huggingface"
    repo: "google/vit-base-patch16-224"
    function: "from_pretrained"
  dino_8b:
    source: "torchhub"
    repo: "facebookresearch/dino:main"
    function: "dino_vits8"
```

### Benefits of `base_path`
- **Simplified file referencing:**  
  Instead of using full paths, if `base_path` is set you can simply reference files relatively:
  ```sh
  perceive -f image1.jpg image2.jpg -m vit_hf -yp models.yaml
  ```

## Optimization Opportunities

1. **Expanded Model & Embedding Choices:**  
   - Support additional models (e.g. DINO) to improve feature extraction.
   - Allow users to specify any model via the YAML configuration without modifying the code.

2. **Folder-Based Comparisons:**  
   - Compare an image to an entire folder (image vs. folder).
   - Compare two folders directly (folder vs. folder) and return the top‑k most similar pairs.

3. **Batch Processing:**  
   - Optimize processing pipelines for GPU acceleration.

4. **Alternative Similarity Metrics:**  
   - Incorporate additional metrics like SSIM, FSIM, and RMSE for a comprehensive similarity evaluation.

## New Features to Consider
- **Interactive Mode:** A CLI mode with interactive prompts for model and metric selection.
- **Advanced Explainability:** Expand Grad-CAM and Grad-CAM++ options for deeper model insights.
- **Customizable Visualization:** Options for different colormaps, transparency settings, and multiple output formats.
- **Caching:** Store computed embeddings to avoid redundant calculations.
---

Perceive is designed to strike a balance between flexibility, usability, and performance. Contributions, feedback, and suggestions are warmly welcomed!
