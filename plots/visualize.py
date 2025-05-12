import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(times, mem_usages, throughputs, energies, grad_times, 
                 labels, train_losses, test_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 12))

    plt.subplot(2, 3, 1)
    for i, (t, label) in enumerate(zip(times, labels)):
        plt.plot(epochs, t, label=label, marker='o')
    plt.title("Epoch Time")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    for i, (m, label) in enumerate(zip(mem_usages, labels)):
        plt.plot(epochs, m, label=label, marker='o')
    plt.title("Memory Usage (MB)")
    plt.xlabel("Epoch")
    plt.ylabel("MB")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 3)
    for i, (tp, label) in enumerate(zip(throughputs, labels)):
        plt.plot(epochs, tp, label=label, marker='o')
    plt.title("Throughput (samples/sec)")
    plt.xlabel("Epoch")
    plt.ylabel("Samples/s")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 4)
    for i, (e, label) in enumerate(zip(energies, labels)):
        plt.plot(epochs, e, label=label, marker='o')
    plt.title("Energy Proxy (CPU% x Time)")
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    for i, (gt, label) in enumerate(zip(grad_times, labels)):
        plt.plot(epochs, gt, label=label, marker='o')
    plt.title("Gradient Computation Time")
    plt.xlabel("Epoch")
    plt.ylabel("Seconds")
    plt.legend()
    plt.grid(True)

    """ plt.subplot(2, 3, 6)
    for i, (acc, label) in enumerate(zip(accuracies, labels)):
        plt.plot(epochs, [a * 100 for a in acc], label=label, marker='o')
    plt.title("Accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True) """

    plt.tight_layout()
    plt.savefig("training_metrics_comparison.png")
    plt.show()
