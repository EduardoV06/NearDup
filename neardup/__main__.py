#!/usr/bin/env python3

import argparse
import os

from neardup.models import  load_models_from_yaml, load_model
from neardup.core import (
    compute_similarity_imgs,
    compute_similarity_img_folder,
    compute_similarity_two_folders
)

def main():
    parser = argparse.ArgumentParser(description="NearDup: Image Similarity Comparison Tool")
    
    # Argument for image files or folder paths
    parser.add_argument(
        "-f", "--files",
        nargs="*",
        type=str,
        help="Two image files to compute similarity between."
    )
    parser.add_argument(
        "-d", "--directory",
        nargs="*",
        type=str,
        help="Two directories to compare (folder vs folder)."
    )
    parser.add_argument(
        "-m", "--model", 
        default="resnet50", 
        help="Model for feature extraction (default: resnet50)."
    )
    parser.add_argument(
        "-yp", "--yaml_path", 
        default="models.yaml", 
        help="Path to the models' YAML lookup table."
    )
    parser.add_argument(
        "-sm", "--similarity_metric", 
        default="cosine", 
        choices=["cosine", "ssim", "fsim", "rmse"],
        help="Similarity metric to use (default: cosine)."
    )

    parser.add_argument(
        "-o", "--output", 
        default="print", 
        choices=["print", "npz", "pickle"],
        help="Output format for folder comparisons. 'print' displays on console; 'npz' or 'pickle' saves results to a file."
    )
    parser.add_argument(
        "-of", "--output_file", 
        type=str, 
        default="",
        help="Optional output file name. If not provided, defaults will be used based on the output format."
    )
    
    
    args = parser.parse_args()

    # Load the configuration from the YAML file
    cfg = load_models_from_yaml(args.yaml_path)
    model_obj = cfg["models"].get(args.model)
    if model_obj is None:
        raise ValueError(f"Model {args.model} not found in the YAML config.")

    # Load the model
    model = load_model(model_obj)

    # Define base path for relative file paths in YAML config
    base_path = cfg.get("base_path", "")

    # Check if comparing two images or two folders
    if args.files and not args.directory:
        # Compare two images
        file1 = os.path.join(base_path, args.files[0]) if base_path else args.files[0]
        file2 = os.path.join(base_path, args.files[1]) if base_path else args.files[1]

        similarity = compute_similarity_imgs(
            model_name=args.model, 
            img1_path=file1, 
            img2_path=file2, 
            yaml_path=args.yaml_path,
            similarity_metric=args.similarity_metric
        )

        print(f"Using model: {args.model}\nSimilarity Score: {similarity:.4f}")


    elif args.directory:
        
        if not args.files:
            folder1 = os.path.join(base_path, args.directory[0]) if base_path else args.directory[0]
            folder2 = os.path.join(base_path, args.directory[1]) if base_path else args.directory[1]

            # Perform batch processing (folder-to-folder)
            similarities = compute_similarity_two_folders(
                model_name=args.model,
                folder1_path=folder1,
                folder2_path=folder2,
                yaml_path=args.yaml_path,
                similarity_metric=args.similarity_metric
            )
        else:
            file1 = os.path.join(base_path, args.files) if base_path else args.files[0]
            folder1 = os.path.join(base_path, args.directory) if base_path else args.directory[0]


            print("\n\n\n",file1)
            print(folder1, "\n\n\n")

            # Compare image-to-folder
            similarities = compute_similarity_img_folder(
                model_name=args.model, 
                img_path=file1, 
                folder_path=folder1, 
                yaml_path=args.yaml_path, 
                similarity_metric=args.similarity_metric
            )

        if args.output == "print":
            print("Similarity Results:")
            print(similarities)
        else:
            # Determine default file name if not provided.
            if args.output_file:
                output_filename = args.output_file
            else:
                output_filename = "output.npz" if args.output == "npz" else "output.pkl"
            
            if args.output == "npz":
                import numpy as np

                np.savez(output_filename, similarities=similarities)
                print(f"Similarity results saved to {output_filename}")
            elif args.output == "pickle":
                import pickle

                with open(output_filename, "wb") as f:
                    pickle.dump(similarities, f)
                print(f"Similarity results saved to {output_filename}")
    
    else:
        print("Error: You must provide either two image files (-f) or two directories (-d) for comparison or one of each")

if __name__ == "__main__":
    main()
