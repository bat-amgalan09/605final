from train_numba import train_with_numba
from visuals import plot_metrics

if __name__ == "__main__":
    train_with_numba(limit=10000)
    metrics = train_with_numba()
    plot_metrics(*metrics, save_path="gpu_metrics_numba.png")
