# LLM Evaluation and Scoring Pipeline
This repository contains a suite of Python scripts designed to automate the large-scale evaluation of Large Language Models (LLMs). It handles everything from executing the benchmarks to post-processing raw data into human-readable formats.

### File Structure
- run_eval.py: The primary orchestrator script that manages the evaluation workflow.

- score_normalize.py: A post-processing script that converts raw accuracy scores into normalized 0–100 scores, accounting for random guessing.

- consolidate_results.py: An aggregation utility that merges individual model JSON results into a single aggregated_results.csv file.