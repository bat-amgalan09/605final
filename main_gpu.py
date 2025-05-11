import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from model import ChatbotModel
from train_gpu import train_model_gpu


def collate_fn(batch):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    texts = [item["dialog"][-1] for item in batch]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.clone()

    return input_ids, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    vocab_size = tokenizer.vocab_size
    model = ChatbotModel(vocab_size=vocab_size).to(device)

    dataset = load_dataset("daily_dialog")
    train_data = dataset["train"].select(range(3000))
    test_data = dataset["test"].select(range(500))

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=collate_fn)

    results = train_model_gpu(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        epochs=10,
        lr=5e-5,
        save_dir="checkpoints/gpu_baseline"
    )


if __name__ == "__main__":
    main()
