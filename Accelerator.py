import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
from accelerate import Accelerator
from dataload import prepare_data
from model import ChatbotModel
from typing import List, Tuple

def train_with_accelerator(
    limit: int = 3000,
    batch_size: int = 64,
    epochs: int = 10,
    save_dir: str = "checkpoints/accelerator"
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:

    os.makedirs(save_dir, exist_ok=True)
    accelerator = Accelerator()

    device = accelerator.device
    train_loader, test_loader, vocab_size, tokenizer = prepare_data(batch_size=batch_size, limit=limit)
    model = ChatbotModel(vocab_size)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    train_losses, test_losses = [], []
    times, mem_usage, energies, grad_times, accuracies = [], [], [], [], []
    best_test_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_tokens = 0, 0

        cpu_percent_before = psutil.cpu_percent(interval=None)
        epoch_start = time.time()
        grad_start = time.time()

        for input_ids, labels in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids, labels)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            tokens = (labels != tokenizer.pad_token_id).sum().item()
            accelerator.backward(loss / tokens)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += tokens

        grad_times.append(time.time() - grad_start)
        epoch_time = time.time() - epoch_start
        times.append(epoch_time)

        mem = torch.cuda.memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0
        mem_usage.append(mem)
        cpu_percent_after = psutil.cpu_percent(interval=None)
        avg_cpu_percent = (cpu_percent_before + cpu_percent_after) / 2
        energies.append(avg_cpu_percent * epoch_time)

        throughputs = len(train_loader.dataset) / epoch_time

        avg_train_loss = total_loss / total_tokens
        train_losses.append(avg_train_loss)

        model.eval()
        test_loss, test_tokens = 0, 0
        with torch.no_grad():
            for input_ids, labels in test_loader:
                logits = model(input_ids, labels)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                test_loss += loss.item()
                test_tokens += (labels != tokenizer.pad_token_id).sum().item()

        avg_test_loss = test_loss / test_tokens
        test_losses.append(avg_test_loss)
        accuracy = 1 / (1 + avg_test_loss)
        accuracies.append(accuracy)

        if avg_test_loss < best_test_loss and accelerator.is_main_process:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

        if accelerator.is_main_process:
            print(f"Epoch {epoch:2d}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, "
                  f"Time = {epoch_time:.2f}s, Energy = {energies[-1]:.2f}, Mem = {mem:.2f} MB")

    return train_losses, test_losses, times, mem_usage, energies, grad_times, accuracies
