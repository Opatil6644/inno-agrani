"""
====================================================================================
    Purpose:
    This script updates the `config.json` file with a new system path (`sys_path`)
    provided via command-line arguments. It is typically used during environment
    setup to dynamically inject absolute paths into the evaluation configuration.

    What This Script Does:
    Parses command-line arguments to read the updated sys_path  
    Loads the existing config.json  
    Safely updates the "sys_path" field  
    Writes changes back with proper JSON formatting  
    Prints a clean success message upon completion  
====================================================================================
"""


import json
import argparse

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Update config.json values")
    parser.add_argument("sys_path", help="Value for sys_path")
    parser.add_argument("config_json_path", help="Config json file path")
    args = parser.parse_args()

    # Read JSON
    with open(args.config_json_path, "r") as f:
        data = json.load(f)

    # Update sys_path
    data["sys_path"] = args.sys_path

    # Write JSON back
    with open(args.config_json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Config file updated successfully")


if __name__ == "__main__":
    main()
