import time
import torch
import torch.nn as nn
import deepspeed
from model import ChatbotModel
from dataload import prepare_data

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 64
MAX_LEN = 64
LIMIT = 1000

def loss_fn(logits, labels, pad_token_id):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

def evaluate(model_engine, dataloader, pad_token_id):
    model_engine.eval()
    total_loss = 0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(model_engine.device)
            labels = labels.to(model_engine.device)

            logits = model_engine(input_ids, labels)
            loss = loss_fn(logits, labels, pad_token_id)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            mask = labels != pad_token_id
            correct += (preds == labels)[mask].sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_tokens if total_tokens > 0 else 0
    return avg_loss, accuracy

def train():
    # Prepare data and model
    train_loader, test_loader, vocab_size, tokenizer = prepare_data(
        max_len=MAX_LEN, batch_size=BATCH_SIZE, limit=LIMIT
    )
    pad_token_id = tokenizer.pad_token_id
    model = ChatbotModel(vocab_size)

    # Reset CUDA memory stats
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DeepSpeed config path
    ds_config_path = "ds_config.json"

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config_path
    )

    total_start_time = time.time()

    for epoch in range(EPOCHS):
        model_engine.train()
        epoch_loss = 0

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(model_engine.device)
            labels = labels.to(model_engine.device)

            logits = model_engine(input_ids, labels)
            loss = loss_fn(logits, labels, pad_token_id)

            model_engine.backward(loss)
            model_engine.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        test_loss, accuracy = evaluate(model_engine, test_loader, pad_token_id)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {accuracy:.4f}")

    total_time = time.time() - total_start_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"\n[Summary]")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Peak GPU Memory Usage: {peak_mem:.2f} GB")

if __name__ == "__main__":
    train()
