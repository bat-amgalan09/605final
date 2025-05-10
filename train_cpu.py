import torch
import torch.nn as nn
import time
import psutil
from typing import List

def train_model(
    rank,
    model,
    dataloader,
    criterion,
    optimizer,
    epochs,
    num_workers,
    queue
):
    print(f"🧠 Using train_cpu.py - process {rank}...")
    process = psutil.Process()
    times, mem_usage, throughputs, energies, grad_times, accuracies = [], [], [], [], [], []

    for epoch in range(epochs):
        torch.autograd.set_detect_anomaly(True)
        model.train()
        total_loss, total_tokens = 0, 0

        cpu_before = psutil.cpu_percent(interval=None)
        epoch_start = time.time()
        grad_start = time.time()

        for input_ids, labels in dataloader:
            input_ids = input_ids.to("cpu")
            labels = labels.to("cpu")

            optimizer.zero_grad()
            logits = model(input_ids, labels)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            token_count = (labels != -100).sum().item()
            (loss / token_count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += token_count

        grad_times.append(time.time() - grad_start)
        epoch_time = time.time() - epoch_start
        times.append(epoch_time)

        mem_mb = process.memory_info().rss / 1024 ** 2
        mem_usage.append(mem_mb)
        cpu_after = psutil.cpu_percent(interval=None)
        avg_cpu = (cpu_before + cpu_after) / 2
        energies.append(avg_cpu * epoch_time)
        throughputs.append(len(dataloader.dataset) / epoch_time)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        accuracy = torch.exp(-torch.tensor(avg_test_loss)).item()
        accuracies.append(accuracy)


    queue.put((rank, times, mem_usage, throughputs, energies, grad_times, accuracies))


""" def train_model_threaded(
    thread_id,
    model,
    dataloader,
    criterion,
    optimizer,
    epochs,
    num_threads=4
):
    import threading
    from queue import Queue

    print(f"🧵 Using threaded training - thread ID {thread_id}...")
    process = psutil.Process()
    lock = threading.Lock()

    times, mem_usage, throughputs, energies, grad_times, accuracies = [], [], [], [], [], []

    def train_epoch():
        nonlocal total_loss, total_tokens
        for input_ids, labels in dataloader:
            input_ids = input_ids.to("cpu")
            labels = labels.to("cpu")

            optimizer.zero_grad()
            logits = model(input_ids, labels)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            token_count = (labels != -100).sum().item()
            (loss / token_count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with lock:
                total_loss += loss.item()
                total_tokens += token_count

    for epoch in range(epochs):
        torch.autograd.set_detect_anomaly(True)
        model.train()
        total_loss, total_tokens = 0, 0
        cpu_before = psutil.cpu_percent(interval=None)
        epoch_start = time.time()
        grad_start = time.time()

        threads = [threading.Thread(target=train_epoch) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        grad_times.append(time.time() - grad_start)
        epoch_time = time.time() - epoch_start
        times.append(epoch_time)

        mem_mb = process.memory_info().rss / 1024 ** 2
        mem_usage.append(mem_mb)
        cpu_after = psutil.cpu_percent(interval=None)
        avg_cpu = (cpu_before + cpu_after) / 2
        energies.append(avg_cpu * epoch_time)
        throughputs.append(len(dataloader.dataset) / epoch_time)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        accuracies.append(1 - avg_loss if avg_loss != float('inf') else 0)

    return times, mem_usage, throughputs, energies, grad_times, accuracies
 """
