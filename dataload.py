import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Tuple

class ChatDataset(Dataset):
    def __init__(self, input_tokens, target_tokens):
        self.input_tokens = input_tokens
        self.target_tokens = target_tokens

    def __len__(self):
        return len(self.input_tokens)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_tokens[idx], dtype=torch.long),
            torch.tensor(self.target_tokens[idx], dtype=torch.long)
        )


def collate_fn(batch, pad_token_id):
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    input_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
    return input_padded, labels_padded


def prepare_data(tokenizer_name='gpt2', max_len=30, batch_size=64, limit=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('daily_dialog', split='train',trust_remote_code=True)
    dialog_pairs = []

    for dialog in dataset['dialog']:
        for i in range(len(dialog) - 1):
            input_text = dialog[i][:500]
            target_text = dialog[i + 1][:500]
            if input_text.strip() and target_text.strip():
                dialog_pairs.append((input_text, target_text))

    if limit:
        dialog_pairs = dialog_pairs[:limit]

    np.random.shuffle(dialog_pairs)
    input_tokens_list, target_tokens_list = [], []

    for input_text, target_text in dialog_pairs:
        input_tokens = tokenizer(input_text, truncation=True, padding='max_length', max_length=max_len)['input_ids']
        target_tokens = tokenizer(target_text, truncation=True, padding='max_length', max_length=max_len)['input_ids']
        input_tokens_list.append(input_tokens)
        target_tokens_list.append(target_tokens)

    split = int(0.8 * len(input_tokens_list))
    train_dataset = ChatDataset(input_tokens_list[:split], target_tokens_list[:split])
    test_dataset = ChatDataset(input_tokens_list[split:], target_tokens_list[split:])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id)
    )

    return train_loader, test_loader, tokenizer.vocab_size, tokenizer

