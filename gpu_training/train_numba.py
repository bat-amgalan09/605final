import numpy as np
import time
import torch
import torch.optim as optim
from numba import cuda, float32
from Chatbot_model.dataload import prepare_data
from Chatbot_model.gpt2_utils import load_gpt2_model_and_tokenizer 
import os
#Limiting the training batch to 10000 so that it can be faster, and for better accuracy, use 20-50 epochs. Saves the model 
def train_with_numba(limit=10000, batch_size=64, epochs=10, save_dir='checkpoints/numba_gpu'):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, test_loader, _, tokenizer = prepare_data(batch_size=batch_size, limit=limit)
    tokenizer, model = load_gpt2_model_and_tokenizer()
    model = model.to(device)
    #Training optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, test_losses, times, mem_usage, energies, grad_times, accuracies = [], [], [], [], [], [], []
    best_test_loss = float('inf')
    #Epoch Start
    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()
        grad_start = time.time()

        total_loss, total_tokens = 0, 0
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
        epoch_time = time.time() - start_time
        times.append(epoch_time)
        #Memory
        mem = torch.cuda.memory_allocated(device) / 1e6
        mem_usage.append(mem)
        #CPU Usage
        cpu_percent = os.getloadavg()[0]
        energies.append(cpu_percent * epoch_time)
        #Throughput
        throughput = len(train_loader.dataset) / epoch_time
        avg_train_loss = total_loss / total_tokens
        train_losses.append(avg_train_loss)
        throughputs = [throughput]

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
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                #Token Prediction for loss calculation
                pred = shift_logits.argmax(dim=-1)
                mask = shift_labels != pad_token_id
                correct = ((pred == shift_labels) & mask).sum().item()
                total = mask.sum().item()
                accuracy = correct / total if total > 0 else 0.0

                total = (labels != pad_token_id).sum().item()
                total_correct += correct
                total_label_tokens += total

        avg_test_loss = test_loss / test_tokens
        test_losses.append(avg_test_loss)
        accuracy = total_correct / total_label_tokens if total_label_tokens > 0 else 0.0
        accuracies.append(accuracy)

        print(f"Epoch {epoch:2d}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, "
              f"Accuracy = {accuracy:.4f}, Time = {epoch_time:.2f}s, Energy = {energies[-1]:.2f}, Mem = {mem:.2f} MB")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

    return train_losses, test_losses, times, mem_usage, energies, grad_times, accuracies
