"""
====================================================================================
    Purpose:
    This script handles secure authentication to the HuggingFace Hub by reading the
    user's access token from config.json. It performs the following tasks:

    Loads and validates config.json
    Extracts the `hf_token` required for authentication
    Performs a safe, non-interactive HuggingFace login
    Provides clear status message
====================================================================================
"""

import json
from huggingface_hub import login
import argparse

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Update config.json values")
    parser.add_argument("--config_json_path", help="Config json file path")
    args = parser.parse_args()


    # Step 1: Load config.json
    try:
        with open(args.config_json_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load {args.config_json_path}: {e}")
        exit(1)

    # Step 2: Read token from JSON
    hf_token = config.get("hf_token")

    if not hf_token:
        print("No `hf_token` found in config.json")
        exit(1)

    # Step 3: Authenticate with HuggingFace
    try:
        login(token=hf_token)
        print("HuggingFace Login Successful!")
    except Exception as e:
        print(f"HuggingFace login failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()