import os
import time
import torch
import deepspeed
import torch.nn.functional as F
from dataload import prepare_data
from gpt2_utils import load_gpt2_model_and_tokenizer

# Training parameters
EPOCHS = 10
BATCH_SIZE = 64
MAX_LEN = 50
LIMIT = 10000

def loss_fn(logits, labels, pad_token_id):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=pad_token_id
    )
    return loss

def evaluate(model_engine, dataloader, pad_token_id):
    model_engine.eval()
    total_loss = 0
    correct = 0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(model_engine.device)
            labels = labels.to(model_engine.device)

            outputs = model_engine(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            total_loss += loss.item()

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            preds = shift_logits.argmax(dim=-1)

            mask = shift_labels != pad_token_id
            correct += (preds == shift_labels)[mask].sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy

def train():
    os.makedirs("checkpoints/deepspeed_gpt2", exist_ok=True)
    
    train_loader, test_loader, _, tokenizer = prepare_data(
        batch_size=BATCH_SIZE, max_len=MAX_LEN, limit=LIMIT
    )
    pad_token_id = tokenizer.pad_token_id

    tokenizer, model = load_gpt2_model_and_tokenizer()

    # Optional: Freeze lower layers if needed
    # for param in model.transformer.h[:6].parameters():
    #     param.requires_grad = False

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="ds_config.json"
    )

    torch.cuda.reset_peak_memory_stats()
    total_start = time.time()
    best_test_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model_engine.train()
        total_loss = 0
        total_tokens = 0
        start_time = time.time()

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(model_engine.device)
            labels = labels.to(model_engine.device)

            outputs = model_engine(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            tokens = (labels != pad_token_id).sum().item()
            model_engine.backward(loss / tokens)
            model_engine.step()

            total_loss += loss.item()
            total_tokens += tokens

        train_loss = total_loss / total_tokens
        test_loss, accuracy = evaluate(model_engine, test_loader, pad_token_id)
        epoch_time = time.time() - start_time

        print(f"[Epoch {epoch:2d}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
              f"Accuracy: {accuracy:.4f} | Time: {epoch_time:.2f}s")

        if test_loss < best_test_loss and model_engine.global_rank == 0:
            best_test_loss = test_loss
            torch.save(model_engine.module.state_dict(), "checkpoints/deepspeed_gpt2/best_model.pt")

    total_time = time.time() - total_start
    peak_mem = torch.cuda.max_memory_allocated(model_engine.device) / 1024**3
    print(f"\n[Summary] Total Time: {total_time:.2f}s | Peak GPU Memory: {peak_mem:.2f} GB")

if __name__ == "__main__":
    train()
