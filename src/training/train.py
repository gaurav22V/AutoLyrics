# src/training/train.py
import sys
import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import *

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

def main():
    print("Initializing Fine-Tuning Engine...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computing Hardware Detected: {device.upper()}")

    # Load compiled dataset
    if not os.path.exists(PROCESSED_DATA_DIR):
        print(f"Error: Compiled tensors not found at {PROCESSED_DATA_DIR}. Run build_dataset.py first!")
        return
        
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    train_dataset = dataset["train"]
    
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # Inject LoRA Parameters 
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.to(device)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        warmup_steps=50,
        max_steps=NUM_STEPS,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="no",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        report_to=["none"],
        remove_unused_columns=False,
        label_names=["labels"]
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    print("Parameters updating")
    trainer.train()

    print(f" Saving adapter weights to: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(" Training completed.")

if __name__ == "__main__":
    main()