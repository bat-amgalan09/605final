# train_deepspeed.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import deepspeed
import time
import os
import psutil
import GPUtil
from model import ChatbotModel
from data import load_dataset, collate_fn

def get_gpu_memory():
    gpus = GPUtil.getGPUs()
    mem_used = sum([gpu.memoryUsed for gpu in gpus])
    mem_total = sum([gpu.memoryTotal for gpu in gpus])
    return mem_used, mem_total

def main():
    # Training hyperparameters
    epochs = 10
    batch_size = 8  # Per-GPU batch size
    lr = 5e-5
    save_dir = "checkpoints_deepspeed"
    os.makedirs(save_dir, exist_ok=True)

    # Model and dataset
    model = ChatbotModel()
    model = model.cuda()

    train_dataset = load_dataset(split="train", limit=3000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=lr)

    # DeepSpeed config inline (or load from ds_config.json if you prefer)
    ds_config = {
        "train_batch_size": batch_size * torch.cuda.device_count(),
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 2},
        "checkpoint": {
            "tag": "global_step1",
            "save": True,
            "load": False
        }
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=None,
        model=model,
        optimizer=optimizer,
        config=ds_config
    )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * epochs
    )

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model_engine.train()
        total_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            model_engine.backward(loss)
            model_engine.step()
            lr_scheduler.step()

            total_loss += loss.item()

        end_time = time.time()
        mem_used, mem_total = get_gpu_memory()
        print(f"[Epoch {epoch + 1}] Loss: {total_loss / len(train_loader):.4f} | "
              f"Time: {end_time - start_time:.2f}s | "
              f"GPU Mem: {mem_used:.2f}/{mem_total:.2f} MB")

        # Save checkpoint
        model_engine.save_checkpoint(save_dir, tag=f"epoch_{epoch + 1}")

if __name__ == "__main__":
    main()
