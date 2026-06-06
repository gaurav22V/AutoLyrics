# src/inference/predict.py
import sys
import os
import json
import torch
import time
from tqdm import tqdm
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import *

def generate_predictions(model, processor, test_data, model_name):
    print(f"\n Inference on {model_name}:")
    model.eval()
    results = []

    for batch in tqdm(test_data):
        input_features = torch.tensor([batch["input_features"]]).to("cuda" if torch.cuda.is_available() else "cpu")
        labels = batch["labels"]

        start_time = time.time()
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_new_tokens=400)
        end_time = time.time()

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        reference = processor.decode(labels, skip_special_tokens=True)

        results.append({
            "reference": reference,
            "prediction": transcription,
            "latency_seconds": end_time - start_time
        })
    return results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading compiled test split from {PROCESSED_DATA_DIR}...")
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    test_data = dataset["test"]
    processor = WhisperProcessor.from_pretrained(MODEL_ID)

    # Base: Whisper Model
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    base_results = generate_predictions(base_model, processor, test_data, "ZERO-SHOT BASELINE")

    del base_model
    torch.cuda.empty_cache()

    # LoRA Model
    print("\n Loading the Fine-tuned LoRA model: ")
    base_model_reload = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    try:
        lora_model = PeftModel.from_pretrained(base_model_reload, OUTPUT_DIR).to(device)
        lora_results = generate_predictions(lora_model, processor, test_data, "AUTOLYRICS LORA")
    except Exception as e:
        print(f"Error loading LoRA adapter from {OUTPUT_DIR}: {e}")
        return

    # Save to JSON
    output_data = {"baseline": base_results, "lora": lora_results}
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
    print(f"\n Saving all predictions to {PREDICTIONS_FILE}")

if __name__ == "__main__":
    main()