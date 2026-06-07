# AutoLyrics : Fine-Tuned Whisper for Lyrics Transcription

A fine-tuned Automatic Speech Recognition (ASR) system designed to transcribe musical lyrics from raw, vocal-heavy audio tracks. This repository contains the complete deep learning pipeline, from dataset curation and parameter-efficient fine-tuning to an interactive web interface.

## Tech Stack
* **Deep Learning:** PyTorch, Transformers, PEFT (LoRA), BitsAndBytes
* **Audio Processing:** FFmpeg, Torchaudio, SciPy
* **Web UI:** Gradio

## Architecture & Features
* **Model:** Fine-tuned `openai/whisper-small` architecture utilizing Low-Rank Adaptation (LoRA).
* **Hardware Optimization:** Integrated 8-bit quantization via BitsAndBytes to dramatically cut the VRAM footprint, allowing memory-optimized GPU training workflows.
* **Web Interface:** Built a clean Gradio interface enabling real-time mic streaming, file uploads, and audio inference with sub-5-second latency.

## Dataset & Training Details
* **Dataset Base:** NUS-48E corpus (paired vocal audio tracks and text lyrics alignments).
* **Automated Data Pipeline:** * Custom audio preprocessing to normalize diverse input frequencies to Whisper's native sampling rate.
  * Synchronized tokenization and feature extraction mappings to build clean dataset batches.
* **Performance Metrics:** * Achieved a **35% relative reduction in Word Error Rate (WER)** compared to the zero-shot baseline.
  * Secured an **11.5% absolute improvement** in final text transcription accuracy via targeted hyperparameter tuning and regularization.

## Project Structure
```text
AutoLyrics/       
 |- requirements.txt             # Python dependencies
 |- packages.txt                 # System-level requirements (ffmpeg)
 |- app.py                       # Main Gradio web application 
 |- data/
    |- NUS_48e/                  # Raw dataset audio files (.wav)
    |- transcripts.json          # Auto-generated transcript annotations
 |- processed_dataset/      
    |- test/                     # Feature-extracted evaluation split
    |- train/                    # Feature-extracted evaluation split
 |- models/
    |- autolyrics_lora/           # Fine-tuned LoRA adapter weights
 |- src/
    |- preprocessing/       
        |- transcribe_audio.py   # Baseline Whisper transcription of raw audio
        |- build_dataset.py      # Dataset creation, feature extraction, and tokenization
    |- training/            
        |- train.py              # LoRA fine-tuning and quantized training setup
    |- inference/           
        |- predict.py            # Baseline vs LoRA inference and latency benchmarking
    |- evaluation/          
        |- metrics.py            # WER calculation and model comparison
```

## Quick Start
```bash
git clone [https://huggingface.co/spaces/gaurav22V/AutoLyrics](https://huggingface.co/spaces/gaurav22V/AutoLyrics)
cd AutoLyrics
pip install -r requirements.txt
python app.py
```
