import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from train_gpu import train_model_gpu
from gpt2_utils import load_gpt2_model_and_tokenizer

def collate_fn(batch):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    texts = [utter for sample in batch for utter in sample["dialog"]]
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")

    input_ids = encodings["input_ids"]
    labels = input_ids.clone()

    return input_ids, labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, model = load_gpt2_model_and_tokenizer()
    model = model.to(device)

    dataset = load_dataset("daily_dialog")
    train_data = dataset["train"].select(range(3000))
    test_data = dataset["test"].select(range(500))

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False,
                             collate_fn=collate_fn)

    train_model_gpu(
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
