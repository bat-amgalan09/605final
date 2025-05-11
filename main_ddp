if __name__ == '__main__':
    import os
    import torch
    import numpy as np
    import torch.distributed as dist
    from dataload import prepare_data
    from model import ChatbotModel
    from train_ddp import train_ddp  # DDP logic

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Launching DDP Training Script")

    # Run DDP even on a single GPU (for simulation/testing purposes)
    world_size = torch.cuda.device_count()
    if world_size >= 1:
        print(f"Using {world_size} GPU(s) with DDP simulation")
        torch.multiprocessing.spawn(
            train_ddp,
            args=(100, 64, 10),  # limit, batch_size, epochs
            nprocs=world_size,
            join=True
        )
    else:
        print("No GPU available for DDP. Please use a GPU-enabled runtime.")
