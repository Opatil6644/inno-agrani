"""
This script is a performance profiling utility designed to benchmark Large Language Models (LLMs) 
using different torch.compile backends (e.g., 'inductor' or 'tensorrt' etc). It measures 
industry-standard metrics to evaluate the efficiency of the model.
"""

import torch
import torch_tensorrt
import time
import pandas as pd
import numpy as np
import os
import gc
from transformers import AutoModelForCausalLM
import argparse
import warnings
warnings.filterwarnings("ignore")
import json

BATCH_SIZE = 8
INPUT_LEN = 32          # vLLM: input_len
OUTPUT_LEN = 128        # vLLM: output_len
RUNS = 10               
DTYPE = torch.bfloat16
DEVICE = "cuda"

config_json_path = "config/user_eval_setting.json"

# Step 1: Load config.json
try:
    with open(config_json_path, "r") as f:
        config = json.load(f)
except Exception as e:
    print(f"Failed to load {config_json_path}: {e}")
    exit(1)

# Step 2: Read token from JSON
hf_token = config.get("hf_token")

HF_TOKEN = hf_token

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=DTYPE,              # torch_dtype deprecated → dtype
        device_map="cuda",
        token=HF_TOKEN,
        trust_remote_code=True,
    ).eval()

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    return model


# -------------------------------------------------
# BENCHMARK FUNCTION (vLLM-ALIGNED & SAFE)
# -------------------------------------------------
def benchmark_backend(model_name, backend_name, input_ids, attention_mask):
    print(f"  > Backend: {backend_name}")

    model = load_model(model_name)

    try:
        # -----------------------------
        # COMPILE
        # -----------------------------
        compile_start = time.time()
        optimized_model = torch.compile(model, backend=backend_name)

        # -----------------------------
        # WARM-UP (forces compilation)
        # -----------------------------
        with torch.no_grad():
            _ = optimized_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4,
                use_cache=True,
                do_sample=False,
            )
        torch.cuda.synchronize()
        compile_time = time.time() - compile_start

        # =====================================================
        # PREFILL + TTFT 
        # =====================================================
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            _ = optimized_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,   # MUST be >= 1 (HF restriction)
                use_cache=True,
                do_sample=False,
            )
        torch.cuda.synchronize()

        ttft = time.time() - start
        prefill_latency = ttft   # 
        prefill_peak_mem = torch.cuda.max_memory_allocated() / 1e9 # Bytes -> GB

        # -----------------------------
        # END-TO-END + DECODE METRICS
        # -----------------------------
        latencies = []
        torch.cuda.reset_peak_memory_stats()

        for _ in range(RUNS):
            torch.cuda.synchronize()
            start = time.time()

            with torch.no_grad():
                output = optimized_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=OUTPUT_LEN,
                    use_cache=True,
                    do_sample=False,
                )

            torch.cuda.synchronize()
            latencies.append(time.time() - start)

        latencies = np.array(latencies)

        total_latency = latencies.mean()
        p50 = np.percentile(latencies, 50)
        p99 = np.percentile(latencies, 99)

        # -----------------------------
        # DECODE THROUGHPUT
        # -----------------------------
        tokens_generated = (
            output.shape[1] - input_ids.shape[1]
        ) * BATCH_SIZE

        decode_time = total_latency - ttft
        decode_throughput = tokens_generated / decode_time

        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        return {
            "Model": model_name.split("/")[-1],
            "Backend": backend_name,
            "Batch": BATCH_SIZE,
            "Input Len": INPUT_LEN,
            "Output Len": OUTPUT_LEN,
            "Compile (s)": round(compile_time, 2),
            "Prefill / TTFT (ms)": round(prefill_latency * 1000, 2),
            "Decode Throughput (tok/s)": round(decode_throughput, 2),
            "End-to-End Lat (s)": round(total_latency, 3),
            "P50 Lat (s)": round(p50, 3),
            "P99 Lat (s)": round(p99, 3),
            "Prefill Peak VRAM (GB)": round(prefill_peak_mem, 2),
            "Peak VRAM (GB)": round(peak_mem, 2),
        }

    except Exception as e:
        print(f"    [!] Error: {e}")
        return {
            "Model": model_name,
            "Backend": backend_name,
            "Error": str(e)[:80],
        }

    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Torch_Tensorrt Inference Profiler")
    parser.add_argument("--model_name", required=True, help="HuggingFace model name")
    parser.add_argument("--backend",  required=True, help="Type of backend to use")
    parser.add_argument("--output_path", help="Path to the output csv file containing results")

    args = parser.parse_args()

    all_results = []

    # for model_name in models:
    print("\n" + "=" * 60)
    print(f"MODEL: {args.model_name}")
    print("=" * 60)

    input_ids = torch.randint(
        low=1000,
        high=10000,
        size=(BATCH_SIZE, INPUT_LEN),
        device=DEVICE,
    )
    attention_mask = torch.ones_like(input_ids)

    # for backend in backends:
    result = benchmark_backend(
        args.model_name,
        args.backend,
        input_ids,
        attention_mask,
    )
    all_results.append(result)

    # -------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 100)
    print(
        f"BENCHMARK | "
        f"{INPUT_LEN} in → {OUTPUT_LEN} out | Batch {BATCH_SIZE}"
    )
    print("=" * 100)
    print(df.to_string(index=False))

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_file = f"benchmark_B{BATCH_SIZE}_{ts}_{args.backend}.csv"
    path = args.output_path
    output_path = f'{path}_{out_file}'
    df.to_csv(output_path, index=False)

    print(f"\n[+] Results saved to: {out_file}")


# -------------------------------------------------
# EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    main()

