import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import psutil
from typing import List, Tuple

def train_model(
    model,
    train_loader,
    test_loader,
    tokenizer,
    device,
    epochs=10,
    lr=0.001,
    grad_clip=1.0,
    save_dir='checkpoints',
    verbose=True
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:

    print("🧠 Using train.py (single-core or GPU)...")
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    times, mem_usage, energies, throughputs, grad_times, accuracies = [], [], [], [], [], []
    best_test_loss = float('inf')
    process = psutil.Process()

    for epoch in range(1, epochs + 1):
        torch.autograd.set_detect_anomaly(True)
        model.train()
        total_loss, total_tokens = 0, 0
        epoch_start_time = time.time()
        grad_start_time = time.time()
        cpu_percent_before = psutil.cpu_percent(interval=None)

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, labels)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            token_count = (labels != tokenizer.pad_token_id).sum().item()
            (loss / token_count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += token_count

        grad_times.append(time.time() - grad_start_time)
        epoch_time = time.time() - epoch_start_time
        times.append(epoch_time)

        mem_mb = torch.cuda.memory_allocated(device) / 1e6 if torch.cuda.is_available() else process.memory_info().rss / 1024 ** 2
        mem_usage.append(mem_mb)

        cpu_percent_after = psutil.cpu_percent(interval=None)
        avg_cpu = (cpu_percent_before + cpu_percent_after) / 2
        energies.append(avg_cpu * epoch_time)
        throughputs.append(len(train_loader.dataset) / epoch_time)

        avg_train_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        train_losses.append(avg_train_loss)

        # Evaluation
        model.eval()
        test_loss, test_tokens = 0, 0
        with torch.no_grad():
            for input_ids, labels in test_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits = model(input_ids, labels)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                token_count = (labels != tokenizer.pad_token_id).sum().item()
                test_loss += loss.item()
                test_tokens += token_count

        avg_test_loss = test_loss / test_tokens if test_tokens > 0 else float('inf')
        test_losses.append(avg_test_loss)
        accuracies.append(1 - avg_test_loss if avg_test_loss != float('inf') else 0)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

        if verbose:
            print(f"Epoch {epoch:2d}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, "
                  f"Time = {epoch_time:.2f}s, Energy = {energies[-1]:.2f}, Mem = {mem_usage[-1]:.2f} MB")

    return train_losses, test_losses, times, mem_usage, throughputs, energies, grad_times, accuracies
