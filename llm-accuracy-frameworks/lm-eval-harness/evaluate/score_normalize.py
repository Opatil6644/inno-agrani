"""
Normalize raw evaluation scores from the lm-evaluation-harness to a 0-100 scale,
accounting for random guessing (lower bounds) based on benchmark-specific subtasks.

1. Normalizes BBH, MUSR, MMLU-PRO, GPQA, IFEval, and MATH scores.
2. Handles single-metric runs or full "leaderboard" aggregate runs.
3. Robust JSON IO with error checking for missing keys in raw data.
"""

import json
from pathlib import Path
import numpy as np
import argparse
import sys
import logging

# Setup logging to inherit configuration from the caller or run standalone
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Normalization function
def normalize_within_range(value, lower_bound=0, higher_bound=1):
    return (np.clip(value - lower_bound, 0, None)) / (higher_bound - lower_bound) * 100



def bbh_normalize(data):
    bbh_subtasks = {
    "sports_understanding": 2,
    "tracking_shuffled_objects_three_objects": 3,
    "navigate": 2,
    "snarks": 2,
    "date_understanding": 6,
    "reasoning_about_colored_objects": 18,
    "object_counting": 19,
    "logical_deduction_seven_objects": 7,
    "geometric_shapes": 11,
    "web_of_lies": 2,
    "movie_recommendation": 6,
    "logical_deduction_five_objects": 5,
    "salient_translation_error_detection": 6,
    "disambiguation_qa": 3,
    "temporal_sequences": 4,
    "hyperbaton": 2,
    "logical_deduction_three_objects": 3,
    "causal_judgement": 2,
    "formal_fallacies": 2,
    "tracking_shuffled_objects_seven_objects": 7,
    "ruin_names": 6,
    "penguins_in_a_table": 5,
    "boolean_expressions": 2,
    "tracking_shuffled_objects_five_objects": 5
}
    
    # Normalize BBH subtasks scores
    bbh_scores = []
    for subtask, num_choices in bbh_subtasks.items():
        subtask_key = f'leaderboard_bbh_{subtask}'
        if subtask_key in data['results']:
            bbh_raw_score = data['results'][subtask_key]['acc_norm,none']
            lower_bound = 1 / num_choices
            normalized_score = normalize_within_range(bbh_raw_score, lower_bound, 1.0)
            bbh_scores.append(normalized_score)

    # Average BBH score
    bbh_score = sum(bbh_scores) / len(bbh_scores)
    return bbh_score


def musr_normalize(data):
    musr_subtasks = {
    'murder_mysteries': 2,
    'object_placements': 5,
    'team_allocation': 3
}
    # Normalize MUSR scores
    musr_scores = []

    for subtask, num_choices in musr_subtasks.items():
        musr_raw_score = data['results'][f'leaderboard_musr_{subtask}']['acc_norm,none']
        lower_bound = 1 / num_choices
        normalized_score = normalize_within_range(musr_raw_score, lower_bound, 1.0)
        musr_scores.append(normalized_score)

    musr_score = sum(musr_scores) / len(musr_scores)
    return musr_score

def mmlu_normalize(data):
    # Normalize MMLU PRO scores
    mmlu_pro_raw_score = data['results']['leaderboard_mmlu_pro']['acc,none']
    mmlu_pro_score = normalize_within_range(mmlu_pro_raw_score, 0.1, 1.0)
    return mmlu_pro_score

def gpqa_normalize(data):
    # Normalize GPQA scores
    gpqa_raw_score = data['results']['leaderboard_gpqa']['acc_norm,none']
    gpqa_score = normalize_within_range(gpqa_raw_score, 0.25, 1.0)
    return gpqa_score

def ifeval_normalize(data):
    # Compute IFEval
    ifeval_inst_score = data['results']['leaderboard_ifeval']['inst_level_strict_acc,none'] * 100
    ifeval_prompt_score = data['results']['leaderboard_ifeval']['prompt_level_strict_acc,none'] * 100
    # Average IFEval scores
    ifeval_score = (ifeval_inst_score + ifeval_prompt_score) / 2
    return ifeval_score

def math_normalize(data):
    # Calculate the MATH score
    math_raw_score = data['results']['leaderboard_math_hard']['exact_match,none']
    math_score = normalize_within_range(math_raw_score, 0, 1.0)
    return math_score

def main():
    parser = argparse.ArgumentParser(description="Normalize raw scores from lm-eval-harness")
    parser.add_argument("raw_results_file", type=Path, help="Path to input JSON")
    parser.add_argument("output_dir", type=Path, help="Directory to save output")
    parser.add_argument("metric_type", help="Metric type (e.g. leaderboard_bbh or leaderboard)")
    parser.add_argument("model_name", help="Name of the model")
    args = parser.parse_args()

    # 1. Load Data
    if not args.raw_results_file.exists():
        logging.error(f"File not found: {args.raw_results_file}")
        sys.exit(1)

    with open(args.raw_results_file, 'r') as f:
        data = json.load(f)

    # 2. Extract Metadata
    precision = data.get('config', {}).get('model_dtype', 'unknown')
    
    # 3. Dispatch Map (Production-grade replacement for if/else blocks)
    metrics_map = {
        "leaderboard_ifeval": ("IFEval", ifeval_normalize),
        "leaderboard_bbh": ("BBH", bbh_normalize),
        "leaderboard_math_hard": ("MATH", math_normalize),
        "leaderboard_gpqa": ("GPQA", gpqa_normalize),
        "leaderboard_musr": ("MUSR", musr_normalize),
        "leaderboard_mmlu_pro": ("MMLU", mmlu_normalize),
    }

    final_results = {
        "Model name": args.model_name,
        "Precision": precision
    }

    # 4. Processing Logic
    if args.metric_type == "leaderboard":
        for label, func in metrics_map.values():
            final_results[label] = func(data)
    elif args.metric_type in metrics_map:
        label, func = metrics_map[args.metric_type]
        final_results[label] = func(data)
    else:
        logging.error(f"Unknown metric_type: {args.metric_type}")
        sys.exit(1)

    # 5. Save Output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / f"{args.metric_type}_normalized.json"
    
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=4)
    
    logging.info(f"Normalization complete. Results: {final_results}")

if __name__ == "__main__":
    main()