import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataload import prepare_data
from model import ChatbotModel


def train_ddp(rank, limit, batch_size, epochs):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=torch.cuda.device_count())

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    train_loader, test_loader, vocab_size, tokenizer = prepare_data(batch_size=batch_size, limit=limit)

    model = ChatbotModel(vocab_size).to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_tokens = 0, 0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, labels)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            tokens = (labels != tokenizer.pad_token_id).sum().item()
            (loss / tokens).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += tokens

        avg_train_loss = total_loss / total_tokens

        # Evaluation (optional)
        model.eval()
        test_loss, test_tokens = 0, 0
        with torch.no_grad():
            for input_ids, labels in test_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits = model(input_ids, labels)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                test_loss += loss.item()
                test_tokens += (labels != tokenizer.pad_token_id).sum().item()

        avg_test_loss = test_loss / test_tokens
        print(f"[GPU {rank}] Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

        if avg_test_loss < best_loss and rank == 0:
            best_loss = avg_test_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/ddp_best_model.pt")

    dist.destroy_process_group()
