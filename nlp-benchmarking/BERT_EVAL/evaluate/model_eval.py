"""
BERT Model Evaluation Engine
----------------------------------------------
This script serves as the core execution engine for the BERT Evaluation and Profiling Pipeline. 
It automates the inference process for various Transformer-based models and benchmarks 
them against standard NLP datasets to generate performance metrics.
"""

import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline, BertTokenizer, BertForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification
import argparse
import json
import os
import sys
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")
import csv
from pathlib import Path

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


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


# Function for evaluating Question-Answering model
def qa_model_eval(model, dataset_in, output_path, model_commit_id):
    model_id = model
    try:
        if dataset_in == "squad_v2":
            # 1. Load the SQuAD validation set
            dataset = load_dataset(dataset_in, split="validation")

            # 2. Initialize the QA pipeline
            # Using the pipeline handles the sliding window (doc_stride) automatically
            qa_pipeline = pipeline(
                "question-answering", 
                model=model_id, 
                tokenizer=model_id, 
                revision=model_commit_id,
                device=0 if torch.cuda.is_available() else -1
            )
            # 3. Load the evaluation metric
            squad_metric = load(dataset_in)

            # 4. Generate Predictions
            # Note: For full replication, iterate through the dataset
            # 4. Generate Predictions
            predictions = []
            references = []

            for example in dataset:
                res = qa_pipeline(
                    question=example["question"],
                    context=example["context"],
                    handle_impossible_answer=True,
                    max_seq_len=384,
                    doc_stride=128
                )
                
                # SQuAD v2 requires 'no_answer_probability'
                # If the pipeline finds an answer, it returns a score for that answer.
                # We use 0.0 as a baseline or calculate it based on the 'null' score if available.
                predictions.append({
                    "prediction_text": res["answer"],
                    "id": example["id"],
                    "no_answer_probability": 0.0  # The metric requires this key to exist
                })
                
                references.append({
                    "answers": example["answers"],
                    "id": example["id"]
                })

        else:
            # 1. Load the SQuAD validation set
            dataset = load_dataset(dataset_in, split="validation")

            # 2. Initialize the QA pipeline
            # Using the pipeline handles the sliding window (doc_stride) automatically
            qa_pipeline = pipeline(
                "question-answering", 
                model=model_id, 
                tokenizer=model_id, 
                device=0 if torch.cuda.is_available() else -1
            )

            # 3. Load the evaluation metric
            squad_metric = load(dataset_in)

            predictions = []
            references = []

            for example in dataset:
                res = qa_pipeline(
                    question=example["question"],
                    context=example["context"],
                    handle_impossible_answer=False, # SQuAD v1.1 has no unanswerable questions
                    max_seq_len=384,
                    doc_stride=128
                )
                
                predictions.append({
                    "prediction_text": res["answer"],
                    "id": example["id"]
                })
                
                references.append({
                    "answers": example["answers"],
                    "id": example["id"]
                })


        # Compute results (This will now work!)
        results = squad_metric.compute(predictions=predictions, references=references)

        print("Results:", results)

        # Export
        os.makedirs(output_path, exist_ok=True)
        for name, data in [("results", results)]:
            path = os.path.join(output_path, f"{name}.json")
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Evaluation completed and results saved to {path}")

    except Exception as e:
        print(f"Initialization Error: {e}")
        sys.exit(1)

# ==================== DATASET LOADING ====================
def load_conll2003_dataset(split="validation"):
    print(f"Loading CoNLL-2003 {split} set...")
    try:
        dataset = load_dataset("conll2003", trust_remote_code=True)
        return dataset[split]
    except Exception as e:
        print(f"Standard load failed, trying alternative: {e}")
        dataset = load_dataset("eriktks/conll2003")
        return dataset[split]

# Function for evaluating token-classification model
def token_classfification_model_eval(model, output_path, model_commit_id):
    try:
        # ==================== CONFIGURATION ====================
        MODEL_NAME = model
        BATCH_SIZE = 16
        MAX_SAMPLES = None

        # The CoNLL-2003 dataset uses this specific index order
        DATASET_LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

        # ==================== MAIN EVALUATION ====================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"1. Loading model and tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=model_commit_id)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, revision=model_commit_id).to(device)
        model.eval()

        # CRITICAL: Get the model's internal mapping
        # dslim/bert-base-NER maps 3 -> B-PER, but CoNLL dataset maps 1 -> B-PER
        id2label_model = model.config.id2label

        print(f"2. Loading dataset")
        test_data = load_conll2003_dataset("validation") 

        if MAX_SAMPLES:
            test_data = test_data.select(range(min(MAX_SAMPLES, len(test_data))))

        all_predictions = []
        all_labels = []

        print(f"3. Running evaluation on {device}...")
        num_batches = (len(test_data) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(test_data))
            
            batch = [test_data[i] for i in range(start_idx, end_idx)]
            batch_tokens = [x['tokens'] for x in batch]
            batch_tags = [x['ner_tags'] for x in batch]
            
            # Tokenize with alignment
            tokenized = tokenizer(
                batch_tokens,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**tokenized)
            
            predictions = torch.argmax(outputs.logits, dim=2).cpu().numpy()

            
            # Process each sequence in batch
            for i in range(len(batch)):
                word_ids = tokenized.word_ids(batch_index=i)
                pred_labels = []
                true_labels = []
                
                previous_word_idx = None
                for j, word_idx in enumerate(word_ids):
                    # 1. Ignore special tokens (None)
                    # 2. Only take the first token of a split word (word_idx != previous_word_idx)
                    if word_idx is not None and word_idx != previous_word_idx:
                        # Map prediction ID to string using MODEL config
                        p_idx = predictions[i][j]
                        pred_labels.append(id2label_model[p_idx])
                        
                        
                        # Map truth ID to string using DATASET labels
                        t_idx = batch_tags[i][word_idx]
                        true_labels.append(DATASET_LABELS[t_idx])
                        
                    previous_word_idx = word_idx

                all_predictions.append(pred_labels)
                all_labels.append(true_labels)

        # ==================== RESULTS ====================
        print("\n" + "="*70)
        print("EVALUATION RESULTS".center(70))
        print("="*70)

        # Calculate metrics
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        acc_score = accuracy_score(all_labels, all_predictions)

        print(f"\nAccuracy:, {acc_score:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        results = {
            "Accuracy": round(acc_score, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4)
        }

        # Export
        os.makedirs(output_path, exist_ok=True)
        for name, data in [("results", results)]:
            path = os.path.join(output_path, f"{name}.json")
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Evaluation completed and results saved to {path}")

    except Exception as e:
        print(f"Initialization Error: {e}")
        sys.exit(1)

# Function for evaluating Masked-language model
def mlm_model_eval(model, output_path, model_commit_id):

    max_samples=3500
    model_id = model
    try:
        # 1. Initialize Pipeline
        # We use 'fill-mask' but we will access the underlying model for loss
        pipe = pipeline('fill-mask', revision=model_commit_id, model=model_id, device=0 if torch.cuda.is_available() else -1)
        model = pipe.model
        tokenizer = pipe.tokenizer
        device = model.device
        model.eval()

        # 2. Load Dataset
        test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        test_texts = [text for text in test_dataset["text"] if len(text.strip()) > 30]

        total_loss = 0
        total_tokens = 0
        mask_prob = 0.15

        print(f"Evaluating Perplexity using {model_id} on {device}...")

        # 3. Evaluation Loop
        for text in tqdm(test_texts[:max_samples]):
            if len(text) < 30:  # skip very short / empty lines
                continue
            text = text[:1000]
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            labels = inputs['input_ids'].clone()
            
            # --- Masking Logic ---
            # Create a mask for special tokens
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)
            
            # Select tokens to mask
            probability_matrix = torch.full(labels.shape, mask_prob, device=device)
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            # Only calculate loss on masked tokens
            labels[~masked_indices] = -100 
            
            # Replace masked indices with [MASK] in input_ids
            inputs['input_ids'][masked_indices] = tokenizer.mask_token_id

            # 4. Forward Pass
            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss 
            
            # 5. Accumulate Statistics
            num_masked = masked_indices.sum().item()
            if num_masked > 0:
                total_loss += loss.item() * num_masked
                total_tokens += num_masked

        # 6. Final Metrics
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = np.exp(avg_loss)
            results = {"perplexity": round(perplexity, 2)}
            print(f"\nFinal Perplexity: {results['perplexity']}")
        else:
            results = {"error": "No tokens were masked."}
            print("No tokens masked.")

        # Export
        os.makedirs(output_path, exist_ok=True)
        path = os.path.join(output_path, "results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
            
    except Exception as e:
        print(f"Error during evaluation: {e}")


def main():
    parser = argparse.ArgumentParser(description="BERT Model Inference Profiler")
    parser.add_argument("--model_name", required=True, help="HuggingFace model name")
    parser.add_argument("--type",  required=True, help="Type of model(fill mask, question answering etc)")
    parser.add_argument("--output_path", type=str, default=".", help="Directory for JSON results")

    args = parser.parse_args()
    df = pd.read_csv("config/dataset_names.csv")

    details = get_model_details_from_csv(args.model_name)
    
    model_commit_id = details['model_commit_id']

    try:
        if args.type == "Question-Answering":
            # Check if model exists
            if args.model_name in df['model'].values:
                dataset_name = df.loc[
                    df['model'] == args.model_name, 'dataset'
                ].values[0]
                print("Using dataset:", dataset_name)
                qa_model_eval(args.model_name, dataset_name, args.output_path, model_commit_id)
            else:
                print("Model not found in CSV")

        elif args.type == "Fill-mask":
            mlm_model_eval(args.model_name, args.output_path, model_commit_id)

        elif args.type == "Token-Classification":
            token_classfification_model_eval(args.model_name, args.output_path, model_commit_id)
    except:
        print("Some issue with the inference pipeline")

if __name__ == "__main__":
    main()