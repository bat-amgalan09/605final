import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import psutil
from dataload import prepare_data
from model import ChatbotModel
from transformers import get_scheduler


def train_with_deepspeed(
    limit=5000,
    batch_size=32,
    epochs=10,
    save_dir='checkpoints/deepspeed'
):
    os.makedirs(save_dir, exist_ok=True)

    # ✅ Environment config to debug NCCL timeout issues
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

    train_losses, test_losses, times, mem_usage, throughputs, energies, grad_times, accuracies = [], [], [], [], [], [], [], []
    best_test_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model_engine.train()
        epoch_start = time.time()
        grad_start = time.time()

        total_loss, total_tokens = 0, 0
        cpu_before = psutil.cpu_percent(interval=None)

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model_engine(input_ids, labels)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            token_count = (labels != tokenizer.pad_token_id).sum().item()
            model_engine.backward(loss / token_count)
            model_engine.step()

            total_loss += loss.item()
            total_tokens += token_count

        grad_times.append(time.time() - grad_start)
        epoch_time = time.time() - epoch_start
        times.append(epoch_time)

        mem = torch.cuda.memory_allocated(device) / 1e6
        mem_usage.append(mem)
        cpu_after = psutil.cpu_percent(interval=None)
        energies.append(((cpu_before + cpu_after) / 2) * epoch_time)

        throughput = len(train_loader.dataset) / epoch_time
        throughputs.append(throughput)
        avg_train_loss = total_loss / total_tokens
        train_losses.append(avg_train_loss)

        # Eval
        model_engine.eval()
        test_loss, test_tokens = 0, 0
        total_correct, total_label_tokens = 0, 0
        with torch.no_grad():
            for input_ids, labels in test_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits = model_engine(input_ids, labels)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                test_loss += loss.item()
                test_tokens += (labels != tokenizer.pad_token_id).sum().item()

                pred = logits.argmax(dim=-1)
                correct = ((pred == labels) & (labels != tokenizer.pad_token_id)).sum().item()
                total = (labels != tokenizer.pad_token_id).sum().item()
                total_correct += correct
                total_label_tokens += total

        avg_test_loss = test_loss / test_tokens
        test_losses.append(avg_test_loss)
        accuracy = total_correct / total_label_tokens if total_label_tokens > 0 else 0.0
        accuracies.append(accuracy)

        if model_engine.is_gradient_accumulation_boundary():
            print(f"[DeepSpeed] Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Time = {epoch_time:.2f}s, "
                  f"CPU = {energies[-1]/epoch_time:.2f}%, Mem = {mem:.2f} MB, Throughput = {throughput:.2f} samples/s")
            print(f"[DeepSpeed] Epoch {epoch}: Test Loss = {avg_test_loss:.4f}, Accuracy = {accuracy:.4f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            model_engine.save_checkpoint(save_dir, tag=f"best_epoch{epoch}")

    return train_losses, test_losses, times, mem_usage, throughputs, energies, grad_times, accuracies


if __name__ == "__main__":
    train_with_deepspeed()
