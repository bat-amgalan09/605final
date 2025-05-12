import torch
import torch.multiprocessing as mp
from train_ddp import train_ddp
import os
from visuals import plot_metrics
if __name__ == '__main__':
    print(" Launching DDP Training Script")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    num_gpus = torch.cuda.device_count()

    print(f"Using {num_gpus} GPU(s) with DDP simulation")

    #  Create Queue inside __main__
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue() if num_gpus > 0 else None

    # ⬇Pass the queue as an argument to all processes (only rank 0 will use it)
    mp.spawn(
        train_ddp,
        args=(10000, 64, 10, queue),
        nprocs=num_gpus,
        join=True
    )

    if queue is not None:
        train_losses, test_losses, times, mem, energy, grad_times, accuracies = queue.get()
        print("Collected metrics from DDP rank 0:")
        print("Train Losses:", train_losses)
        print("Test Losses:", test_losses)
        print("Times:", times)
        print("Memory (MB):", mem)
        print("Energy:", energy)
        print("Gradient Times:", grad_times)
        print("Accuracies:", accuracies)
    metrics = train_with_deepspeed()
    plot_metrics(*metrics, save_path="gpu_metrics_deepspeed.png")
