import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Tuple

class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, token_sequences):
        self.token_sequences = token_sequences

    def __len__(self):
        return len(self.token_sequences)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.token_sequences[idx], dtype=torch.long)
        return tokens, tokens.clone()  # GPT-2 uses input_ids and labels = same

def collate_fn(batch, pad_token_id):
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    input_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
    return input_padded, labels_padded

def prepare_data(tokenizer_name='gpt2', max_len=64, batch_size=64, limit=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('daily_dialog', split='train', trust_remote_code=True)
    dialog_pairs = []

    for dialog in dataset['dialog']:
        for i in range(len(dialog) - 1):
            input_text = dialog[i][:500]
            target_text = dialog[i + 1][:500]
            if input_text.strip() and target_text.strip():
                full_text = input_text + tokenizer.eos_token + target_text
                dialog_pairs.append(full_text)

    if limit:
        dialog_pairs = dialog_pairs[:limit]

    np.random.shuffle(dialog_pairs)

    token_sequences = [
        tokenizer(text, truncation=True, padding='max_length', max_length=max_len)["input_ids"]
        for text in dialog_pairs
    ]

    split = int(0.8 * len(token_sequences))
    train_dataset = ChatDataset(token_sequences[:split])
    test_dataset = ChatDataset(token_sequences[split:])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id)
    )

    return train_loader, test_loader, tokenizer.vocab_size, tokenizer
