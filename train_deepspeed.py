import os
import time
import torch
import torch.optim as optim
import psutil
import deepspeed
from dataload import prepare_data
from gpt2_utils import load_gpt2_model_and_tokenizer
import torch.distributed as dist

def train_with_deepspeed(
    batch_size=8,
    epochs=10,
    limit=1000,
    save_dir="checkpoints/deepspeed"
):
    os.makedirs(save_dir, exist_ok=True)
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    # Load GPT-2 tokenizer and model
    tokenizer, model = load_gpt2_model_and_tokenizer()

    # Prepare data
    train_loader, test_loader, _, tokenizer = prepare_data(batch_size=batch_size, limit=limit)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # Inline DeepSpeed config
    ds_config = {
        "train_batch_size": batch_size * torch.cuda.device_count(),
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 1},
        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )

    best_test_loss = float("inf")
    pad_token_id = tokenizer.pad_token_id

    for epoch in range(1, epochs + 1):
        model_engine.train()
        total_loss, total_tokens = 0, 0
        start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=None)

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model_engine(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

            token_count = (labels != pad_token_id).sum().item()
            total_loss += loss.item()
            total_tokens += token_count

        epoch_time = time.time() - start_time
        cpu_after = psutil.cpu_percent(interval=None)
        avg_cpu = (cpu_before + cpu_after) / 2
        mem = torch.cuda.memory_allocated(device) / 1e6
        throughput = len(train_loader.dataset) / epoch_time
        avg_loss = total_loss / total_tokens

        print(f"[DeepSpeed] Epoch {epoch}: Train Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s, "
              f"CPU = {avg_cpu:.2f}%, Mem = {mem:.2f} MB, Throughput = {throughput:.2f} samples/s")

        # Evaluation
        model_engine.eval()
        test_loss, test_tokens = 0, 0
        total_correct, total_label_tokens = 0, 0

        with torch.no_grad():
            for input_ids, labels in test_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                outputs = model_engine(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                test_loss += loss.item()
                test_tokens += (labels != pad_token_id).sum().item()

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                preds = shift_logits.argmax(dim=-1)
                mask = (shift_labels != pad_token_id)
                correct = ((preds == shift_labels) & mask).sum().item()
                total = mask.sum().item()
                total_correct += correct
                total_label_tokens += total

        avg_test_loss = test_loss / test_tokens
        accuracy = total_correct / total_label_tokens if total_label_tokens > 0 else 0.0

        print(f"[DeepSpeed] Epoch {epoch}: Test Loss = {avg_test_loss:.4f}, Accuracy = {accuracy:.4f}")

        # Save only the best checkpoint on rank 0
        if model_engine.global_rank == 0 and avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            model_engine.save_checkpoint(save_dir, tag=f"best_epoch{epoch}")


if __name__ == "__main__":
    train_with_deepspeed()
