## Also tried numba cuda optimization, but it did not fit well with llms
from train_numba import train_with_numba
from visuals import plot_metrics

if __name__ == "__main__":
    train_with_numba(limit=10000)

