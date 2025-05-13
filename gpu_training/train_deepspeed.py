import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import deepspeed
import logging
import psutil
from dataload import prepare_data
from gpt2_utils import load_gpt2_model_and_tokenizer

def setup_logging():
    """Configure logging with rank-specific output."""
    rank = deepspeed.comm.get_rank() if deepspeed.comm.is_initialized() else 0
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [Rank {rank}] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"deepspeed_metrics_rank_{rank}.log", mode="w"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Configuration
CONFIG = {
    "epochs": 20,
    "batch_size": 64,
    "max_len": 50,
    "limit": 100000,
    "checkpoint_dir": "checkpoints/deepspeed_gpt2",
    "seed": 42,
    "gpu_power_watts": 250  # Power draw per GPU (W), adjustable
}

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_peak_memory(device):
    """Get peak memory usage in GB for GPU or CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / 1024**3
    return psutil.virtual_memory().used / 1024**3

def loss_fn(logits, labels, pad_token_id):
    """Compute cross-entropy loss, ignoring pad tokens."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=pad_token_id
    )

def evaluate(model_engine, dataloader, pad_token_id):
    """Evaluate model on test dataset, returning average loss and accuracy."""
    model_engine.eval()
    epoch_test_loss = 0
    correct_preds = 0
    total_non_pad_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = batch
            input_ids = input_ids.to(model_engine.device)
            labels = labels.to(model_engine.device)

            outputs = model_engine(input_ids=input_ids, labels=labels)
            epoch_test_loss += outputs.loss.item()

            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            preds = shift_logits.argmax(dim=-1)

            mask = shift_labels != pad_token_id
            correct_preds += (preds == shift_labels)[mask].sum().item()
            total_non_pad_tokens += mask.sum().item()

    avg_loss = epoch_test_loss / len(dataloader)
    accuracy = correct_preds / total_non_pad_tokens if total_non_pad_tokens > 0 else 0.0
    return avg_loss, accuracy

def train_step(model_engine, batch, pad_token_id):
    """Perform one training step, returning loss and token count."""
    input_ids, labels = batch
    input_ids = input_ids.to(model_engine.device)
    labels = labels.to(model_engine.device)

    outputs = model_engine(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    non_pad_tokens = (labels != pad_token_id).sum().item()

    model_engine.backward(loss / non_pad_tokens)
    model_engine.step()

    return loss.item(), non_pad_tokens

def save_checkpoint(model_engine, epoch, test_loss, path):
    """Save model checkpoint if on rank 0."""
    if model_engine.global_rank == 0:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_engine.module.state_dict(),
            "test_loss": test_loss
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

def train(config, ds_config_path="ds_config.json"):
    """Train GPT-2 model with DeepSpeed ZeRO-2, tracking memory, time, test loss, energy, and throughput."""
    logger = setup_logging()
    set_seed(config["seed"])
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # Load data
    try:
        train_loader, test_loader, _, tokenizer = prepare_data(
            batch_size=config["batch_size"],
            max_len=config["max_len"],
            limit=config["limit"]
        )
        logger.info(f"Dataset: train={len(train_loader.dataset)}, test={len(test_loader.dataset)}, vocab_size={len(tokenizer)}")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise

    pad_token_id = tokenizer.pad_token_id
    total_samples = len(train_loader.dataset)

    # Load model
    try:
        tokenizer, model = load_gpt2_model_and_tokenizer()
        logger.info(f"Loaded GPT-2 model with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

    # Initialize DeepSpeed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config_path
        )
    except Exception as e:
        logger.error(f"DeepSpeed initialization failed: {e}")
        raise

    if device.type == "cuda":
        logger.info(f"Rank {model_engine.global_rank}: Assigned to GPU {torch.cuda.current_device()} ({torch.cuda.get_device_name()})")

    # Training loop
    total_start = time.perf_counter()
    best_test_loss = float("inf")
    metrics_history = []
    peak_memory_total = 0.0
    total_energy_joules = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config["epochs"] + 1):
        model_engine.train()
        epoch_train_loss = 0
        total_non_pad_tokens = 0
        epoch_start = time.perf_counter()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for batch in train_loader:
            loss, non_pad_tokens = train_step(model_engine, batch, pad_token_id)
            epoch_train_loss += loss
            total_non_pad_tokens += non_pad_tokens

        train_loss = epoch_train_loss / total_non_pad_tokens
        test_loss, accuracy = evaluate(model_engine, test_loader, pad_token_id)
        epoch_time = time.perf_counter() - epoch_start
        peak_memory = get_peak_memory(model_engine.device)
        epoch_throughput = total_samples / epoch_time
        peak_memory_total = max(peak_memory_total, peak_memory)

        # Estimate energy: power_draw (W) × 2 GPUs × time (s) = Joules
        energy_joules = config["gpu_power_watts"] * 2 * epoch_time
        total_energy_joules += energy_joules

        # Log metrics
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "accuracy": accuracy,
            "epoch_time_s": epoch_time,
            "peak_memory_gb": peak_memory,
            "energy_joules": energy_joules,
            "throughput_samples_sec": epoch_throughput
        }
        metrics_history.append(metrics)
        logger.info(
            f"[Epoch {epoch:2d}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
            f"Accuracy: {accuracy:.4f} | Time: {epoch_time:.2f}s | Memory: {peak_memory:.2f} GB | "
            f"Energy: {energy_joules:.2f} J | Throughput: {epoch_throughput:.2f} samples/sec"
        )

        # Save checkpoint if test loss improves
        checkpoint_path = os.path.join(config["checkpoint_dir"], f"epoch_{epoch}_loss_{test_loss:.4f}.pt")
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_checkpoint(model_engine, epoch, test_loss, checkpoint_path)

    # Summary
    total_time = time.perf_counter() - total_start
    total_throughput = total_samples / total_time

    logger.info(
        f"[Summary] Total Time: {total_time:.2f}s | Peak Memory: {peak_memory_total:.2f} GB | "
        f"Total Energy: {total_energy_joules:.2f} J | Throughput: {total_throughput:.2f} samples/sec | "
        f"Final Test Loss: {best_test_loss:.4f}"
    )

    return metrics_history

if __name__ == "__main__":
    metrics = train(CONFIG)
