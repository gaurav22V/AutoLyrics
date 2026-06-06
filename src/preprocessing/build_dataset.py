# src/preprocessing/build_dataset.py
import sys
import os
import json
from datasets import Dataset, Audio
from transformers import WhisperProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import *

def main():
    print("Starting Dataset Compilation...")
    
    if not os.path.exists(TRANSCRIPTS_FILE):
        print(f"Error: {TRANSCRIPTS_FILE} not found. Run transcribe_audio.py first!")
        return

    with open(TRANSCRIPTS_FILE, "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    raw_data = {"audio_path": [], "lyrics": []}
    
    for wav_path, lyric_text in transcripts.items():
        if os.path.exists(wav_path):
            raw_data["audio_path"].append(wav_path)
            raw_data["lyrics"].append(lyric_text)

    print(f"Successfully verified {len(raw_data['audio_path'])} audio-text pairs.")
    
    dataset = Dataset.from_dict({
        "audio": raw_data["audio_path"],
        "sentence": raw_data["lyrics"]
    })
    
    # Resample the frequency to 16,000 Hz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    
    # Splitting dataset: 80% - training, 20% - test
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    
    # Feature Extraction & Tokenization Mapping 
    def prepare_dataset(batch):
        audio = batch["audio"]
        
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        # Tokenization of flat English words
        batch["labels"] = processor.tokenizer(
            batch["sentence"],
            max_length=400,
            truncation=True
        ).input_ids
        return batch

    print("Converting audio waves to spectrograms and processing tokens")
    processed_dataset = split_dataset.map(prepare_dataset, remove_columns=["audio", "sentence"], num_proc=1)
    
    os.makedirs(os.path.dirname(PROCESSED_DATA_DIR), exist_ok=True)
    processed_dataset.save_to_disk(PROCESSED_DATA_DIR)
    print(f"Dataset compiled. Saved to: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()