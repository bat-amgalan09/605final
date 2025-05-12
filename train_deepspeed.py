import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import psutil
from dataload import prepare_data
from model import ChatbotModel


def train_with_deepspeed(
    limit=3000,
    batch_size=32,
    epochs=10,
    save_dir='checkpoints/deepspeed'
):
    os.makedirs(save_dir, exist_ok=True)

    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["NCCL_IB_DISABLE"] = "1"

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, vocab_size, tokenizer = prepare_data(batch_size=batch_size, limit=limit)
    model = ChatbotModel(vocab_size)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    ds_config = {
        "train_batch_size": batch_size * torch.cuda.device_count(),
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 1},
        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )

    for epoch in range(1, epochs + 1):
        model_engine.train()
        epoch_start = time.time()

        total_loss, total_tokens = 0, 0

        for input_ids, _ in train_loader:
            input_ids = input_ids.to(device)
            labels = input_ids[:, 1:].clone()
            input_ids = input_ids[:, :-1]

            outputs = model_engine(input_ids)

            # Align lengths safely
            min_len = min(outputs.size(1), labels.size(1))
            outputs = outputs[:, :min_len, :]
            labels = labels[:, :min_len]

            labels = labels.clamp(min=0, max=vocab_size - 1)

            loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
            token_count = (labels != tokenizer.pad_token_id).sum().item()
            model_engine.backward(loss / token_count)
            model_engine.step()

            total_loss += loss.item()
            total_tokens += token_count

        epoch_time = time.time() - epoch_start
        avg_train_loss = total_loss / total_tokens

        print(f"[DeepSpeed] Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Time = {epoch_time:.2f}s")

        # Evaluation
        model_engine.eval()
        test_loss, test_tokens = 0, 0
        total_correct, total_label_tokens = 0, 0

        with torch.no_grad():
            for input_ids, _ in test_loader:
                input_ids = input_ids.to(device)
                labels = input_ids[:, 1:].clone()
                input_ids = input_ids[:, :-1]

                outputs = model_engine(input_ids)

                min_len = min(outputs.size(1), labels.size(1))
                outputs = outputs[:, :min_len, :]
                labels = labels[:, :min_len]

                labels = labels.clamp(min=0, max=vocab_size - 1)

                loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
                test_loss += loss.item()
                test_tokens += (labels != tokenizer.pad_token_id).sum().item()

                pred = outputs.argmax(dim=-1)
                correct = ((pred == labels) & (labels != tokenizer.pad_token_id)).sum().item()
                total = (labels != tokenizer.pad_token_id).sum().item()
                total_correct += correct
                total_label_tokens += total

        avg_test_loss = test_loss / test_tokens
        accuracy = total_correct / total_label_tokens if total_label_tokens > 0 else 0.0

        print(f"[DeepSpeed] Epoch {epoch}: Test Loss = {avg_test_loss:.4f}, Accuracy = {accuracy:.4f}")

        if avg_test_loss < 1e9:
            model_engine.save_checkpoint(save_dir, tag=f"epoch{epoch}")


if __name__ == "__main__":
    train_with_deepspeed()
