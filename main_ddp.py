if __name__ == '__main__':
    import os
    import torch
    import numpy as np
    import torch.distributed as dist
    from dataload import prepare_data
    from model import ChatbotModel
    from train_ddp import train_ddp  # DDP logic
    from torch.multiprocessing import Queue

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("✅ Launching DDP Training Script")

    # Run DDP even on a single GPU (for simulation/testing purposes)
    world_size = torch.cuda.device_count()
    if world_size >= 1:
        print(f"Using {world_size} GPU(s) with DDP simulation")

        os.makedirs("checkpoints", exist_ok=True)
        queue = Queue()

        torch.multiprocessing.spawn(
            train_ddp,
            args=(100, 64, 10, queue),  # limit, batch_size, epochs, queue
            nprocs=world_size,
            join=True
        )

        # 📊 Collect metrics from queue
        if not queue.empty():
            metrics = queue.get()
            (train_losses, test_losses, times, mem, throughputs, energies, grad_times, accuracies) = metrics

            print("✅ Metrics collected from DDP:")
            print("Train Losses:", train_losses)
            print("Test Losses:", test_losses)
            print("Times:", times)
            print("Memory Usage:", mem)
            print("Throughput:", throughputs)
            print("Energy:", energies)
            print("Gradient Times:", grad_times)
            print("Accuracies:", accuracies)

            # Optionally save metrics
            np.savez("checkpoints/ddp_metrics.npz", 
                     train_losses=train_losses,
                     test_losses=test_losses,
                     times=times,
                     mem=mem,
                     throughputs=throughputs,
                     energies=energies,
                     grad_times=grad_times,
                     accuracies=accuracies)

    else:
        print("❌ No GPU available for DDP. Please use a GPU-enabled runtime.")
