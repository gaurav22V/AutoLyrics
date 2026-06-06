# src/evaluation/metrics.py
import sys
import os
import json
import evaluate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import *

def print_metrics(results_list, model_name):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    refs = [item["reference"] for item in results_list]
    preds = [item["prediction"] for item in results_list]
    avg_latency = sum(item["latency_seconds"] for item in results_list) / len(results_list)

    # Calculate error metrics
    wer = 100 * wer_metric.compute(predictions=preds, references=refs)
    cer = 100 * cer_metric.compute(predictions=preds, references=refs)

    print(f"\n{model_name} METRICS: ")
    print(f"Word Error Rate (WER): {wer:.2f}%")
    print(f"Character Error Rate (CER): {cer:.2f}%")
    print(f"Average Inference Latency: {avg_latency:.2f} seconds per clip")
    return wer

def main():
    print(f"Reading inference logs from: {PREDICTIONS_FILE}...")
    
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"Error: File not found at {PREDICTIONS_FILE}")
        print("Please execute src/inference/predict.py before running the metric analyzer.")
        return

    with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_wer = print_metrics(data["baseline"], "BASELINE")
    lora_wer = print_metrics(data["lora"], "AUTOLYRICS LoRA")

    # Benchmarking Delta
    absolute_improvement = base_wer - lora_wer
    relative_reduction = (absolute_improvement / base_wer) * 100 if base_wer > 0 else 0

    print("\nComparing Baseline and LoRA")
    print(f"Absolute WER Improvement: {absolute_improvement:.2f}%")
    print(f"Relative WER Reduction: {relative_reduction:.2f}%")

    # Constraint Evaluation
    if lora_wer < base_wer and relative_reduction >= 15.0:
        print(" Success: Cleared target constraint of greater than 15% reduction.")
    elif lora_wer < base_wer:
        print("Improved, but didn't hit the greater than 15% threshold.")
    else:
        print("Model performed worse than the Baseline")

if __name__ == "__main__":
    main()