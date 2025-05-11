# train_deepspeed.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
import deepspeed
from model import ChatbotModel
from data import load_dataset, collate_fn
import os

def main():
    # Hyperparameters
    epochs = 10
    lr = 5e-5
    batch_size = 8
    model = ChatbotModel()
    model = model.cuda()

    # Load data
    train_dataset = load_dataset(split='train', limit=5000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=lr)

    # DeepSpeed configuration
    ds_config = {
        "train_batch_size": batch_size * torch.cuda.device_count(),
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 2}
    }

    # Initialize DeepSpeed
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
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()

            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            model_engine.backward(loss)
            model_engine.step()
            lr_scheduler.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch + 1}] Loss: {total_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    main()
