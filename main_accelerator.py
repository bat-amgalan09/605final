from Accelerator import train_with_accelerator

if __name__ == "__main__":
    train_with_accelerator(limit=10000)
    metrics = train_with_accelerator()
    plot_metrics(*metrics, save_path="hf_accelerator.png")
