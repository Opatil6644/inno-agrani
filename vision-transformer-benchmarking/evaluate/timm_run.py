"""
PURPOSE:
This script serves as a unified entry point for performing inference across 
different model architectures and libraries. It automatically routes an input 
image to the appropriate processing pipeline based on the model provider.
"""

# 1. FIX: Patch amdsmi at the very start
import sys
from types import ModuleType
if 'amdsmi' not in sys.modules:
    fake_amdsmi = ModuleType('amdsmi')
    fake_amdsmi.AmdSmiException = Exception
    sys.modules['amdsmi'] = fake_amdsmi

import torch
import timm
import json
import time
import argparse
import warnings
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor
from transformers import AutoImageProcessor, AutoModel
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.image_utils import load_image
    

warnings.filterwarnings("ignore")

import csv
from pathlib import Path


def get_model_details_from_csv(model_name, config_path=None):
    DEFAULT_CSV_PATH = "config/model_details.csv"
    """
    Searches for model details. 
    Checks config_path first (if provided); if not found, checks DEFAULT_CSV_PATH.
    """
    # Create a list of paths to check in order of priority
    search_paths = []
    if config_path:
        search_paths.append(Path(config_path))
    
    # Add default path if it's not already the primary path
    if Path(DEFAULT_CSV_PATH) not in search_paths:
        search_paths.append(Path(DEFAULT_CSV_PATH))

    for current_path in search_paths:
        if not current_path.exists():
            # If the user specifically provided a path that doesn't exist, we skip it
            # and try the next one (default), or print a warning.
            continue

        try:
            with open(current_path, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('model', '').strip() == model_name:
                        print(f"Reading configuration from: {current_path}")
                        return {
                            "tp_size": row.get('tensor_parallel', '').strip(),
                            "trust_remote": row.get('trust_remote_code', '').strip().lower(),
                            "model_commit_id": row.get('model_commit_id', '').strip()
                        }
        except KeyError as e:
            print(f"Warning: Missing column {e} in {current_path}. Skipping...")
            continue

    # If the loop finishes without returning, the model wasn't in ANY file
    print(f"Error: Model '{model_name}' not found in {[str(p) for p in search_paths]}")
    sys.exit(1)

# --- Logic for Falconsai / Transformers Models ---
# UPDATED: Now takes model_name and img_path as arguments
def run_nsfw_detection(model_name, img_path, model_commit_id):
    print(f"Running NSFW: {model_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load model and processor dynamically based on model_name
        model = AutoModelForImageClassification.from_pretrained(model_name, revision=model_commit_id).to(device)
        processor = ViTImageProcessor.from_pretrained(model_name, revision=model_commit_id)
        model.eval()

        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predicted_label_idx = logits.argmax(-1).item()
        label_name = model.config.id2label[predicted_label_idx]

        print(f"RESULT: {label_name.upper()}")
        print(f"DEVICE USED: {device}\n")
        
    except Exception as e:
        print(f"ERROR in NSFW detection for {model_name}: {e}")

# --- Logic for TIMM Models ---
def model_inference(model_name, img_path, labels):
    print(f"Running TIMM: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        img = Image.open(img_path).convert("RGB")
        model = timm.create_model(model_name, pretrained=True)
        model.to(device)
        model.eval()

        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        input_tensor = transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        best_index = torch.argmax(output, dim=1).item()
        best_label = labels[best_index]

        print(f"RESULT: {best_label}")
        print(f"DEVICE USED: {device}\n")

    except Exception as e:
        print(f"ERROR in TIMM model {model_name}: {str(e)}\n")

def run_fbdino_detection(model_name, img_path, model_commit_id):
    print(f"Running facebook: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True, revision=model_commit_id)
    model = AutoModel.from_pretrained(model_name, revision=model_commit_id)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 4. Open and Preprocess Image
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    # 5. Extract Features
    with torch.no_grad():
        # Prepare image and move tensors to device
        inputs = processor(images=img, return_tensors="pt").to(device)
        
        # Pass through the model
        outputs = model(**inputs)
        image_embedding = outputs.last_hidden_state[:, 0, :]

    # 6. Output Results
    print(f"EMBEDDING SHAPE: {list(image_embedding.shape)}")
    print(f"DEVICE USED: {device}")

def run_paligemma_detection(model_name, img_path, model_commit_id):
    image = load_image(img_path)

    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", revision=model_commit_id).eval()
    processor = PaliGemmaProcessor.from_pretrained(model_name, revision=model_commit_id)

    # Leaving the prompt blank for pre-trained models
    prompt = ""
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        print(decoded)


# --- Main Entry Point with Routing ---
def main():
    parser = argparse.ArgumentParser(description="Multi-model Evaluation Router")
    parser.add_argument("--model_name", required=True, help="Name of the model (timm/... or falcon...)")
    parser.add_argument("--img_path", required=True, help="Path of the input image")
    args = parser.parse_args()

    details = get_model_details_from_csv(args.model_name)
    
    model_commit_id = details['model_commit_id']


    # Load ImageNet labels (needed for TIMM models)
    try:
        with open("config/imagenet_classes.json", "r") as f:
            class_map = json.load(f)
        labels = [class_map[str(i)][1] for i in range(len(class_map))]
    except FileNotFoundError:
        labels = []
        print("Warning: imagenet_classes.json not found. TIMM classification may fail.")

    # --- ROUTING LOGIC ---
    
    # Check for TIMM prefix
    if args.model_name.startswith("hf-hub:"):
        model_inference(args.model_name, args.img_path, labels)

    # Check for Falcon prefix
    elif args.model_name.startswith("Falconsai"):
        run_nsfw_detection(args.model_name, args.img_path, model_commit_id)

    elif args.model_name.startswith("facebook"):
        run_fbdino_detection(args.model_name, args.img_path, model_commit_id)

    elif args.model_name.startswith("google"):
        run_paligemma_detection(args.model_name, args.img_path, model_commit_id)

    else:
        print(f"Invalid  Model name")
        print(f"Received: {args.model_name}")

if __name__ == "__main__":
    main()