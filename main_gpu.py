import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataload import prepare_data
from train_gpu import train_model_gpu
from gpt2_utils import load_gpt2_model_and_tokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = load_gpt2_model_and_tokenizer()
    model = model.to(device)

    train_loader, test_loader, _, tokenizer = prepare_data(batch_size=64, limit=10000)

    train_model_gpu(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        epochs=10,
        lr=5e-5,
        save_dir="checkpoints/gpu_gpt2"
    )

if __name__ == "__main__":
    main()
    metrics = train_with_deepspeed()
    plot_metrics(*metrics, save_path="gpu_metrics_GPU.png")
