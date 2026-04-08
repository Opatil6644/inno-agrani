"""
This script performs a deep-dive analysis of Hugging Face CausalLM architectures. 
It captures structural metadata, tensor shapes, and resource consumption by:
1. Authenticating with Hugging Face and loading a model's configuration.
2. Registering forward hooks on every sub-module to capture real-time input/output 
   tensor shapes during a forward pass.
3. Measuring CPU (RSS) and GPU (Allocated) memory usage incrementally across layers.
4. Generating an interactive, collapsible HTML tree-map of the model that highlights:
   - Parameter counts and disk size (MB) per module.
   - Exact activation shapes flowing through the network.
"""


from huggingface_hub import login
import json
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import gc
import psutil
import os
from collections import defaultdict
import time
import csv
from pathlib import Path
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

hook_results = defaultdict(list)


# --- Hook function to capture memory and shape ---
def capture_mem_and_shape_hook(module, input, output):
    module_path = getattr(module, '_full_module_path', 'N/A')

    input_shape = "N/A"
    if module_path == "":
        if isinstance(input, tuple) and len(input) == 2 and isinstance(input[1], dict):
            if 'input_ids' in input[1] and isinstance(input[1]['input_ids'], torch.Tensor):
                input_shape = tuple(input[1]['input_ids'].shape)
            elif 'inputs_embeds' in input[1] and isinstance(input[1]['inputs_embeds'], torch.Tensor):
                input_shape = tuple(input[1]['inputs_embeds'].shape)
            else:
                if isinstance(input[0], tuple) and len(input[0]) > 0 and isinstance(input[0][0], torch.Tensor):
                    input_shape = tuple(input[0][0].shape)
                else:
                    input_shape = f"<ComplexRootInput>"
        else:
            if isinstance(input, torch.Tensor):
                input_shape = tuple(input.shape)
            elif isinstance(input, tuple):
                shapes = []
                for inp_item in input:
                    if isinstance(inp_item, torch.Tensor):
                        shapes.append(tuple(inp_item.shape))
                    else:
                        shapes.append(f"<{type(inp_item).__name__}>")
                input_shape = tuple(shapes) if shapes else "N/A"
            else:
                input_shape = f"<{type(input).__name__}>"

    elif isinstance(input, torch.Tensor):
        input_shape = tuple(input.shape)
    elif isinstance(input, tuple):
        shapes = []
        for inp_item in input:
            if isinstance(inp_item, torch.Tensor):
                shapes.append(tuple(inp_item.shape))
            elif isinstance(inp_item, (int, float, bool, str)):
                shapes.append(str(inp_item))
            elif isinstance(inp_item, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in inp_item):
                shapes.append([tuple(x.shape) for x in inp_item])
            else:
                shapes.append(f"<{type(inp_item).__name__}>")
        input_shape = tuple(shapes) if shapes else "N/A"
    else:
        input_shape = f"<{type(input).__name__}>"

    output_shape = "N/A"
    if isinstance(output, torch.Tensor):
        output_shape = tuple(output.shape)
    elif hasattr(output, 'logits') and isinstance(output.logits, torch.Tensor):
        output_shape = tuple(output.logits.shape)
    elif hasattr(output, 'last_hidden_state') and isinstance(output.last_hidden_state, torch.Tensor):
        output_shape = tuple(output.last_hidden_state.shape)
    elif isinstance(output, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in output):
        output_shape = [tuple(x.shape) for x in output]
    elif isinstance(output, dict):
        output_shape = {k: tuple(v.shape) if isinstance(v, torch.Tensor) else f"<{type(v).__name__}>" for k, v in output.items()}
    else:
        output_shape = f"<{type(output).__name__}>"

    current_cpu_mem = cpu_mem_mb()
    current_gpu_mem = gpu_mem_mb()

    hook_results[module_path].append({
        "cpu_mem_mb": current_cpu_mem,
        "gpu_mem_mb": current_gpu_mem,
        "input_shape": input_shape,
        "output_shape": output_shape
    })


def gpu_mem_mb():
    """Returns the current GPU memory allocated in MB."""
    if device == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def cpu_mem_mb():
    """Returns the current CPU resident set size (RSS) memory in MB."""
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


# building html file
def build_enriched_tree(module, current_path, depth=0, max_depth=None):
    name = module.__class__.__name__
    module._full_module_path = current_path

    param_count = sum(p.numel() for p in module.parameters(recurse=False))

    param_size_mb = 0
    if param_count > 0:
        first_param = next(iter(module.parameters(recurse=False)), None)
        if first_param is not None:
            dtype_size_bytes = first_param.element_size()
            param_size_mb = (param_count * dtype_size_bytes) / (1024 * 1024)

    highlight = 'activation' if isinstance(module, (nn.ReLU, nn.SiLU, nn.GELU)) else \
                'linear' if isinstance(module, nn.Linear) else ''

    children = []
    total_params = param_count
    total_param_size_mb = param_size_mb

    if max_depth is None or depth < max_depth:
        for child_name, child_module in module.named_children():
            new_child_path = f"{current_path}.{child_name}" if current_path else child_name
            child_tree = build_enriched_tree(child_module, new_child_path, depth + 1, max_depth)
            total_params += child_tree["total_params"]
            total_param_size_mb += child_tree["total_param_size_mb"]
            children.append(child_tree)

    mem_info = hook_results.get(current_path, [])
    first_mem_entry = mem_info[-1] if mem_info else {}

    return {
        "name": name,
        "path": current_path,
        "params": param_count,
        "total_params": total_params,
        "param_size_mb": param_size_mb,
        "total_param_size_mb": total_param_size_mb,
        "depth": depth,
        "highlight": highlight,
        "children": children,
        "input_shape": first_mem_entry.get("input_shape", "N/A"),
        "output_shape": first_mem_entry.get("output_shape", "N/A"),
        "cpu_mem_after": first_mem_entry.get("cpu_mem_mb", "N/A"),
        "gpu_mem_after": first_mem_entry.get("gpu_mem_mb", "N/A")
    }

def format_shape(shape):
    """Formats shapes for display, especially for tuples of shapes and dicts."""
    if isinstance(shape, tuple) and all(isinstance(s, (tuple, str, list, dict)) for s in shape):
        return f"({', '.join(format_shape(s) for s in shape)})"
    elif isinstance(shape, list) and all(isinstance(s, (tuple, str, list, dict)) for s in shape):
        return f"[{', '.join(format_shape(s) for s in shape)}]"
    elif isinstance(shape, dict):
        return "{" + ", ".join(f"'{k}': {format_shape(v)}" for k, v in shape.items()) + "}"
    return str(shape)


def render_tree_html_enriched(tree):
    style = ""
    if tree["highlight"] == "activation":
        style = ' style="color:green;"'
    elif tree["highlight"] == "linear":
        style = ' style="color:blue;"'

    label = f"{tree['name']} — {tree['total_params']:,} params"
    if tree['total_param_size_mb'] > 0:
        label += f" ({tree['total_param_size_mb']:.2f} MB)"

    info_parts = []
    if tree["input_shape"] != "N/A" and str(tree["input_shape"]) not in ["<NoInput>", "<ComplexRootInput>", "<class 'tuple'>"]:
        info_parts.append(f"Input: {format_shape(tree['input_shape'])}")

    if tree["output_shape"] != "N/A" and str(tree["output_shape"]) != "<NoOutput>":
        info_parts.append(f"Output: {format_shape(tree['output_shape'])}")

    cpu_mem_str = "N/A"
    gpu_mem_str = "N/A"

    if isinstance(tree['cpu_mem_after'], (int, float)):
        cpu_mem_str = f"{tree['cpu_mem_after']:.2f}"
    if isinstance(tree['gpu_mem_after'], (int, float)):
        gpu_mem_str = f"{tree['gpu_mem_after']:.2f}"

    if cpu_mem_str != "N/A" or gpu_mem_str != "N/A":
        if cpu_mem_str != "N/A" and gpu_mem_str != "N/A":
            info_parts.append(f"CPU: {cpu_mem_str} MB | GPU: {gpu_mem_str} MB")
        elif cpu_mem_str != "N/A":
            info_parts.append(f"CPU: {cpu_mem_str} MB")
        elif gpu_mem_str != "N/A":
            info_parts.append(f"GPU: {gpu_mem_str} MB")

    if info_parts:
        label += " | " + " | ".join(info_parts)

    if not tree["children"]:
        return f"<li{style}>{label}</li>"

    children_html = "".join(render_tree_html_enriched(c) for c in tree["children"])
    return f"<li><span class='caret'{style}>{label}</span><ul class='nested'>{children_html}</ul></li>"


def build_html_with_param_tree(title, config, tree):
    html = [
        "<html><head><title>Model Architecture</title><style>",
        "body { font-family: Arial; padding: 2em; }",
        ".caret::before { content: '\\25B6'; display: inline-block; margin-right: 6px; }",
        ".caret { cursor: pointer; user-select: none; }",
        ".caret-down::before { transform: rotate(90deg); }",
        ".nested { display: none; list-style: none; margin-left: 20px; }",
        ".active { display: block; }",
        "table { border-collapse: collapse; margin-top: 20px; }",
        "td, th { border: 1px solid #ccc; padding: 8px; }",
        "th { background: #f2f2f2; }",
        "span[style*='color:green']::after { content: ' (activation)'; font-style: italic; font-size: 90%; color: darkgreen; }",
        "span[style*='color:blue']::after { content: ' (linear)'; font-style: italic; font-size: 90%; color: navy; }",
        "ul { list-style: none; padding-left: 0; }",
        "ul ul { padding-left: 20px; }"
        "</style></head><body>",
        f"<h1>{title}</h1><table>"
    ]
    for k, v in config.items():
        val = json.dumps(v, indent=2) if isinstance(v, dict) else v
        html.append(f"<tr><th>{k}</th><td><pre>{val}</pre></td></tr>")
    html.append("</table><h2>Model Tree (Total Parameters + Output Info)</h2><ul>")
    html.append(render_tree_html_enriched(tree))
    html.extend([
        "</ul><script>",
        "document.querySelectorAll('.caret').forEach(function(el) {",
        "   el.onclick = function() {",
        "       this.parentElement.querySelector('.nested').classList.toggle('active');",
        "       this.classList.toggle('caret-down');",
        "   };",
        "});",
        "</script></body></html>"
    ])
    return "\n".join(html)



def hf_login():
    config_json_path = "config/config.json"

# Step 1: Load config.json
    try:
        with open(config_json_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load {config_json_path}: {e}")
        exit(1)

    # Step 2: Read token from JSON
    hf_token = config.get("huggingface_token")

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


def get_model_details_from_csv(model_name, config_path=None):
    DEFAULT_CSV_PATH = "config/details.csv"
    """
    Searches for model details. 
    Checks config_path first (if provided); if not found, checks DEFAULT_CSV_PATH.
    """
    # Create a list of paths to check in order of priority
    search_paths = []
    
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
                            "model_commit_id": row.get('model_commit_id', '').strip()
                        }
        except KeyError as e:
            print(f"Warning: Missing column {e} in {current_path}. Skipping...")
            continue

    # If the loop finishes without returning, the model wasn't in ANY file
    print(f"Error: Model '{model_name}' not found in {[str(p) for p in search_paths]}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Update vLLM benchmark JSONs with dynamic parameters.")
    parser.add_argument("model", help="Model name/ID")
    parser.add_argument("results_path", help="Path to the store the results")

    args = parser.parse_args()

    hf_login()

    model_name = args.model

    details = get_model_details_from_csv(model_name)
    
    model_commit_id = details['model_commit_id']

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=model_commit_id)
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)
        model.to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")


        # prompt
        my_prompt = "Hey how are you doing?"

        print(my_prompt)


        inputs = tokenizer(my_prompt, return_tensors="pt")

        # Moving inputs to the same device as the model
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        sequence_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        print(f"Tokenized prompt input_ids shape: {input_ids.shape}")
        print(f"Tokenized prompt attention_mask shape: {attention_mask.shape}")
        print(f"Calculated batch_size: {batch_size}, sequence_length: {sequence_length}")


        hook_results.clear()

        # full module paths for consistent lookup
        for module_name, module_instance in model.named_modules():
            module_instance._full_module_path = module_name

        # Registering hooks
        hooks = []
        for module_name, module_instance in model.named_modules():
            if isinstance(module_instance, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                continue
            hooks.append(module_instance.register_forward_hook(capture_mem_and_shape_hook))


        # try:
        with torch.no_grad():
            start_time = time.time()
            outputs = model(input_ids=input_ids)
            end_time = time.time()
            print(f"Forward pass completed in {end_time - start_time:.4f} seconds.")

            logits = outputs.logits
            print(f"Output logits shape: {logits.shape}")


        for h in hooks:
            h.remove()

        if not hook_results:
            print("hook_results is empty!")
        else:
            if "" in hook_results:
                print(f"Path: \"\" (Root Module)")
                root_entry = hook_results[""][-1]
                print(f"  Entry: Input={root_entry.get('input_shape')}, Output={root_entry.get('output_shape')}, GPU={root_entry.get('gpu_mem_mb'):.2f}MB")

            count = 0
            for i, (path, data_list) in enumerate(hook_results.items()):
                if path == "":
                    continue
                if count >= 5:
                    break
                if data_list:
                    entry = data_list[-1]
                    print(f"Path: {path}")
                    print(f"  Entry: Input={entry.get('input_shape')}, Output={entry.get('output_shape')}, GPU={entry.get('gpu_mem_mb'):.2f}MB")
                    count += 1
                else:
                    print(f"Path: {path} (No data entries)")

        
        # --- Start tree building from the absolute root model ---
        initial_module = model
        initial_path = ""
        initial_module._full_module_path = initial_path

        tree = build_enriched_tree(initial_module, current_path=initial_path)

        result_model_name = args.model.replace("/", "_")

        html = build_html_with_param_tree(f"{model_name} Architecture", config.to_dict(), tree)
        output_path = f"{args.results_path}/{result_model_name}.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)


        print(f"Saved to: {output_path}")


        del model, input_ids, attention_mask, tokenizer, outputs, logits
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        print("\n architecture HTML file generated with prompt input. ")

    except:
        print("Issue while creating model tree")


if __name__ == "__main__":
    main()