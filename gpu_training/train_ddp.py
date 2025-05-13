import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataload import prepare_data
from gpt2_utils import load_gpt2_model_and_tokenizer
import time
import psutil


def train_ddp(rank, limit, batch_size, epochs, queue):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=torch.cuda.device_count())

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    train_loader, test_loader, _, tokenizer = prepare_data(batch_size=batch_size, limit=limit)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer, model = load_gpt2_model_and_tokenizer()
    model = model.to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    train_losses = []
    test_losses = []
    times = []
    mem_usage = []
    energies = []
    grad_times = []
    accuracies = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        grad_start = time.time()

        total_loss, total_tokens = 0, 0
        cpu_percent_before = psutil.cpu_percent(interval=None)

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            tokens = (labels != tokenizer.pad_token_id).sum().item()
            (loss / tokens).backward()

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

        avg_train_loss = total_loss / total_tokens
        train_losses.append(avg_train_loss)

        # Evaluation
        model.eval()
        test_loss, test_tokens = 0, 0
        total_correct = 0
        total_label_tokens = 0
        pad_token_id = tokenizer.pad_token_id

        with torch.no_grad():
            for input_ids, labels in test_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                test_loss += loss.item()
                test_tokens += (labels != pad_token_id).sum().item()

                ## Accuracy
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                mask = shift_labels != pad_token_id
                pred = shift_logits.argmax(dim=-1)
                correct = ((pred == shift_labels) & mask).sum().item()
                total = mask.sum().item()
                total_correct += correct
                total_label_tokens += total

        avg_test_loss = test_loss / test_tokens
        test_losses.append(avg_test_loss)
        accuracy = total_correct / total_label_tokens if total_label_tokens > 0 else 0.0
        accuracies.append(accuracy)

        print(f"[GPU {rank}] Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, "
              f"Accuracy = {accuracy:.4f}, Mem = {mem:.2f} MB")

        if avg_test_loss < best_loss and rank == 0:
            best_loss = avg_test_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/ddp_best_model.pt")

    if rank == 0 and queue is not None:
        queue.put((train_losses, test_losses, times, mem_usage, energies, grad_times, accuracies))

    dist.destroy_process_group()
