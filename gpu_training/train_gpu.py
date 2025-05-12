import torch
import torch.optim as optim
import time
import psutil
import os
from typing import List, Tuple

def train_model_gpu(
    model,
    train_loader,
    test_loader,
    tokenizer,
    device,
    epochs=10,
    lr=0.001,
    grad_clip=1.0,
    save_dir='checkpoints/gpu_gpt2',
    verbose=True
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:

    os.makedirs(save_dir, exist_ok=True)
    #Using Adama optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses, times, mem_usage, throughputs, energies, grad_times, accuracies = [], [], [], [], [], [], [], []
    best_test_loss = float('inf')
    #Starting the training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        grad_start = time.time()

        total_loss = 0
        total_tokens = 0
        cpu_percent_before = psutil.cpu_percent(interval=None)

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            token_count = (labels != tokenizer.pad_token_id).sum().item()
            (loss / token_count).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += token_count
        #Time
        grad_times.append(time.time() - grad_start)
        epoch_time = time.time() - epoch_start
        times.append(epoch_time)
        #Memory
        mem = torch.cuda.memory_allocated(device) / 1e6 if torch.cuda.is_available() else 0
        mem_usage.append(mem)
        cpu_percent_after = psutil.cpu_percent(interval=None)
        avg_cpu_percent = (cpu_percent_before + cpu_percent_after) / 2
        energies.append(avg_cpu_percent * epoch_time)
        #Throughput
        throughput = len(train_loader.dataset) / epoch_time
        throughputs.append(throughput)
    
        avg_train_loss = total_loss / total_tokens
        train_losses.append(avg_train_loss)

        # Evaluation
        model.eval()
        test_loss = 0
        test_tokens = 0
        total_correct = 0
        total_label_tokens = 0
        pad_token_id = tokenizer.pad_token_id
        # Accuracy calculation
        with torch.no_grad():
            for input_ids, labels in test_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                test_loss += loss.item()
                test_tokens += (labels != pad_token_id).sum().item()

                #Accuracy prediction
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

        print(f" Epoch {epoch}: avg_test_loss={avg_test_loss:.4f}, accuracy={accuracy:.4f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

        if verbose:
            print(f"Epoch {epoch:2d}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, "
                  f"Time = {epoch_time:.2f}s, Energy = {energies[-1]:.2f}, Mem = {mem:.2f} MB")

    return train_losses, test_losses, times, mem_usage, throughputs, energies, grad_times, accuracies
