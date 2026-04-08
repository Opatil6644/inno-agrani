"""
====================================================================================
    Purpose:
    This script updates selected fields inside config.json based on the values
    provided through command-line arguments. It is used during evaluation setup to
    dynamically inject runtime-specific parameters such as HuggingFace token, dataset
    paths, result paths, sample count and batch-size.

    What This Script Does:
    Loads the existing config.json safely
    Accepts arguments for:
        - hf_token
        - coco_dataset_path
        - results_path
        - num_samples
        - batch-size
    Updates the provided fields
    Writes back the updated JSON with proper formatting
    Prints status messages for each field updated/missing
====================================================================================
"""


import json
import argparse


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Update config.json values")
    parser.add_argument("--hf_token", help="user hf token")
    parser.add_argument("--coco_dataset_path", help="Path to save coco dataset")
    parser.add_argument("--results_path", help="Path to save evaluation results")
    parser.add_argument("--num_samples", help="Number of sample images to generate")
    parser.add_argument("--batch_size", help="Batch-size used while generating images")
    parser.add_argument("--save_gen_img", help="Whether to save generated images(true/false)")
    parser.add_argument("--config_json_path", help="Config json file path")
    args = parser.parse_args()

    # Read JSON
    with open(args.config_json_path, "r") as f:
        data = json.load(f)


    if args.hf_token:
         # Check if hf_token is same
        if str(data["hf_token"]) == str(args.hf_token):
            print("hf_token same not updated")
        else:
            data["hf_token"] = args.hf_token
            print(f"hf_token updated")
    else:
        print("hf_token not found")

    if args.coco_dataset_path:
        # Check if coco_data_path same
        if str(data["coco_data_path"]) == str(args.coco_dataset_path):
            print("coco_data_path same not updated")
        else:
            data["coco_data_path"] = args.coco_dataset_path
            print(f"coco_data_path updated")
    else:
        print("coco_data_path not found")

    if args.results_path:
        # Check if results_path same
        if str(data["results_path"]) == str(args.results_path):
            print("results_path same not updated")
        else:
            data["results_path"] = args.results_path
            print(f"results_path updated")
    else:
        print("results_path not found")

    
    if args.num_samples:
        # Check if num of samples same
        if str(data["num_samples"]) == str(args.num_samples):
            print("num_samples same not updated")
        else:
            data["num_samples"] = args.num_samples
            print(f"num_samples updated")
    else:
        print("num_samples not found")

    
    if args.batch_size:
        # Check if batch size same
        if str(data["batch_size"]) == str(args.batch_size):
            print("batch_size same not updated")
        else:
            data["batch_size"] = args.batch_size
            print(f"batch_size updated")
    else:
        print("batch_size not found")


    
    if args.save_gen_img:
        # Check if save_gen_img size same
        if str(data["save_gen_img"]) == str(args.save_gen_img):
            print("save_gen_img same not updated")
        else:
            data["save_gen_img"] = args.save_gen_img
            print(f"save_gen_img updated")
    else:
        print("save_gen_img not found")


    # Write JSON back
    with open(args.config_json_path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
