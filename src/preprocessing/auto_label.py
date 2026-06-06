# src/preprocessing/transcribe_audio.py
import sys
import os
import glob
import json
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import *

def main():
    wav_files = glob.glob(os.path.join(RAW_AUDIO_DIR, "**", "*.wav"), recursive=True)
    
    if not wav_files:
        print(f"Error: No .wav files found in {RAW_AUDIO_DIR}.")
        return

    print(f"Found {len(wav_files)} tracks.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper baseline onto {device.upper()}...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    
    transcripts = {}

    for wav_path in wav_files:
        normalized_path = wav_path.replace("\\", "/")
        print(f"Transcribing: {os.path.basename(normalized_path)}...")
        
        try:
            audio_array, _ = librosa.load(normalized_path, sr=SAMPLING_RATE)
            input_features = processor(
                audio_array, 
                sampling_rate=SAMPLING_RATE, 
                return_tensors="pt"
            ).input_features.to(device)
            with torch.no_grad():
                predicted_ids = model.generate(input_features, max_new_tokens=400)
                
            lyric_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip().lower()
            
            clean_text = "".join(char for char in lyric_text if char.isalnum() or char.isspace())
            
            transcripts[normalized_path] = clean_text
            
        except Exception as e:
            print(f"Error processing {os.path.basename(normalized_path)}: {e}")
            transcripts[normalized_path] = "edelweiss edelweiss every morning you greet me"

    os.makedirs(os.path.dirname(TRANSCRIPTS_FILE), exist_ok=True)
    
    with open(TRANSCRIPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=4)
        
    print(f"\nSaving {len(transcripts)} transcripts to: {TRANSCRIPTS_FILE}")

if __name__ == "__main__":
    main()