import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()

# Inputs
RAW_AUDIO_DIR = str(PROJECT_ROOT / "data" / "NUS_48E")
TRANSCRIPTS_FILE = str(PROJECT_ROOT / "data" / "transcripts.json")

# Outputs
PROCESSED_DATA_DIR = str(PROJECT_ROOT / "processed_dataset")
OUTPUT_DIR = str(PROJECT_ROOT / "models" / "autolyrics_lora")
PREDICTIONS_FILE = str(PROJECT_ROOT / "src" / "predictions.json")

MODEL_ID = "openai/whisper-small" 
SAMPLING_RATE = 16000
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_STEPS = 500