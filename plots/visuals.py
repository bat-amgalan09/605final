import matplotlib.pyplot as plt
import seaborn as sns

# Use a nice Seaborn style
sns.set(style="whitegrid", font_scale=1.2)

def plot_metrics(train_losses, test_losses, times, mem_usage, throughputs, accuracies, save_path="gpu_metrics.png"):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(16, 10))

    # 1. Loss plot
    plt.subplot(2, 2, 1)
    sns.lineplot(x=epochs, y=train_losses, label="Train Loss")
    sns.lineplot(x=epochs, y=test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    # 2. Time per epoch
    plt.subplot(2, 2, 2)
    sns.lineplot(x=epochs, y=times, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("Training Time Per Epoch")

    # 3. GPU memory usage
    plt.subplot(2, 2, 3)
    sns.lineplot(x=epochs, y=mem_usage, marker="o", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Memory (MB)")
    plt.title("GPU Memory Usage")

    # 4. Throughput + Accuracy
    plt.subplot(2, 2, 4)
    sns.lineplot(x=epochs, y=throughputs, label="Throughput (samples/s)")
    sns.lineplot(x=epochs, y=accuracies, label="Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.title("Throughput and Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Benchmark plot saved to: {save_path}")
