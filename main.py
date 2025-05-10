if __name__ == '__main__':
    import os
    import torch
    import numpy as np
    from dataload import prepare_data
    from model import ChatbotModel
    from train import train_model  # For single-core and GPU (MPS)
    from train_cpu import train_model as train_model_cpu  # For multi-core CPU
    from torch.utils.data import Subset
    from torch.multiprocessing import Process, Queue, set_start_method
    from visualize import plot_metrics

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Device Setup
    #device = torch.device("cpu")  # Force CPU mode for parallel testing
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ENABLE_PARALLEL = device.type == "cpu"
    print(f"Using device: {device}")

    # Initialize metric containers
    all_times = []
    all_mem = []
    all_throughputs = []
    all_energies = []
    all_grad_times = []
    all_accuracies = []
    all_train_losses = []
    all_test_losses = []
    labels = []

    # Data Preparation
    train_loader, test_loader, vocab_size, tokenizer = prepare_data(batch_size=64, limit=100)

    # Single-Core Training
    print("\n🔁 Starting Single-Core Training...")
    model = ChatbotModel(vocab_size).to(device)
    train_losses, test_losses, times, mem, throughputs, energies, grad_times, accuracies = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        epochs=10,
        save_dir="checkpoints/singlecore"
    )

    all_times.append(times)
    all_mem.append(mem)
    all_throughputs.append(throughputs)
    all_energies.append(energies)
    all_grad_times.append(grad_times)
    all_accuracies.append(accuracies)
    all_train_losses = train_losses
    all_test_losses = test_losses
    labels.append("Single-core")
# Multi-core CPU Training
if ENABLE_PARALLEL:
    print("\n🔁 Starting Multi-Core Training...")
    set_start_method("spawn", force=True)
    num_cores = 4

    queue = Queue()
    processes = []
    dataset = train_loader.dataset
    dataset_size = len(dataset)
    chunk_size = dataset_size // num_cores
    subsets = [Subset(dataset, range(i * chunk_size, (i + 1) * chunk_size)) for i in range(num_cores)]
    if dataset_size % num_cores != 0:
        subsets[-1] = Subset(dataset, range((num_cores - 1) * chunk_size, dataset_size))

    for rank in range(num_cores):
        model_copy = ChatbotModel(vocab_size).to(device)
        model_copy.load_state_dict(model.state_dict())
        optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=0.001)
        subset_loader = torch.utils.data.DataLoader(subsets[rank], batch_size=64, shuffle=True, num_workers=0)
        p = Process(target=train_model_cpu, args=(rank, model_copy, subset_loader, torch.nn.CrossEntropyLoss(), optimizer_copy, 10, num_cores, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # ✅ This part was wrongly unindented — now fixed:
    process_metrics = []
    for _ in range(num_cores):
        rank, times, mem, throughputs, energies, grad_times, accuracies = queue.get()
        process_metrics.append((rank, times, mem, throughputs, energies, grad_times, accuracies))

    process_metrics.sort(key=lambda x: x[0])

    def average_across_processes(metric_index):
        return [
            sum(proc[metric_index][i] for proc in process_metrics) / num_cores
            for i in range(len(process_metrics[0][metric_index]))
        ]

    all_times.append(average_across_processes(1))
    all_mem.append(average_across_processes(2))
    all_throughputs.append(average_across_processes(3))
    all_energies.append(average_across_processes(4))
    all_grad_times.append(average_across_processes(5))
    all_accuracies.append(average_across_processes(6))
    labels.append("Multi-core CPU")


    # Visualize Training Metrics
    print("\n📊 lengths:")
    print("throughputs:", len(all_throughputs))
    print("energies:", len(all_energies))
    print("grad_times:", len(all_grad_times))
    print("accuracies:", len(all_accuracies))
    print("labels:", len(labels))
    print("train_losses:", len(all_train_losses))
    print("test_losses:", len(all_test_losses))

    plot_metrics(
        times=all_times,
        mem_usages=all_mem,
        throughputs=all_throughputs,
        energies=all_energies,
        grad_times=all_grad_times,
        
        labels=labels,
        train_losses=all_train_losses,
        test_losses=all_test_losses
    )

    print("✅ All training and visualization completed.")