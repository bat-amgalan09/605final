from gpu_training.Accelerator import train_with_accelerator
from plots.visuals import plot_metrics

if __name__ == "__main__":
    train_with_accelerator(limit=10000)
