import os
import time
import torch
import torch.optim as optim
import psutil
from accelerate import Accelerator
from Chatbot_model.dataload import prepare_data
from Chatbot_model.gpt2_utils import load_gpt2_model_and_tokenizer
from typing import List, Tuple

def train_with_accelerator(
    limit: int = 10000,
    batch_size: int = 64,
    epochs: int = 10,
    save_dir: str = "checkpoints/accelerator"
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:

    os.makedirs(save_dir, exist_ok=True)
    accelerator = Accelerator()
    device = accelerator.device

    train_loader, test_loader, _, tokenizer = prepare_data(batch_size=batch_size, limit=limit)

    # GPT-2 model
    tokenizer, model = load_gpt2_model_and_tokenizer()
    model = model.to(device)
    # Optimizer
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
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

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

        # Evaluation
        model.eval()
        test_loss, test_tokens = 0, 0
        total_correct = 0
        total_label_tokens = 0
        pad_token_id = tokenizer.pad_token_id

        with torch.no_grad():
            for input_ids, labels in test_loader:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                test_loss += loss.item()
                test_tokens += (labels != pad_token_id).sum().item()

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                pred = shift_logits.argmax(dim=-1)
                mask = (shift_labels != pad_token_id)
                correct = ((pred == shift_labels) & mask).sum().item()
                total = mask.sum().item()
                total_correct += correct
                total_label_tokens += total

        avg_test_loss = test_loss / test_tokens
        test_losses.append(avg_test_loss)
        accuracy = total_correct / total_label_tokens if total_label_tokens > 0 else 0.0
        accuracies.append(accuracy)

        if avg_test_loss < best_test_loss and accelerator.is_main_process:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

        if accelerator.is_main_process:
            print(f"Epoch {epoch:2d}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, "
                  f"Accuracy = {accuracy:.4f}, Time = {epoch_time:.2f}s, Energy = {energies[-1]:.2f}, Mem = {mem:.2f} MB")

    return train_losses, test_losses, times, mem_usage, energies, grad_times, accuracies
