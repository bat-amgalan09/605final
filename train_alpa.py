import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax
from transformers import AutoTokenizer
import alpa
from alpa import parallelize
from datasets import load_dataset
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    has_nvml = True
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except ImportError:
    has_nvml = False


# Utility to measure GPU memory usage
def get_gpu_memory():
    if has_nvml:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / 1e6  # MB
    return 0

# Simple model definition
class ChatMLP(nn.Module):
    hidden_size: int
    vocab_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Embed(self.vocab_size, self.hidden_size)(x)
        x = jnp.mean(x, axis=1)
        x = nn.Dense(self.vocab_size)(x)
        return x

# Training step decorated with Alpa parallelization
@parallelize
def train_step(state, batch_inputs, batch_labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch_inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_labels).mean()
        return loss
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

def compute_loss(state, batch_inputs, batch_labels):
    logits = state.apply_fn({'params': state.params}, batch_inputs)
    return optax.softmax_cross_entropy_with_integer_labels(logits, batch_labels).mean()

def prepare_data(limit=1000, max_len=30):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("daily_dialog", split="train")
    pairs = []
    for dialog in dataset["dialog"]:
        for i in range(len(dialog) - 1):
            input_text, target_text = dialog[i], dialog[i + 1]
            if input_text.strip() and target_text.strip():
                pairs.append((input_text, target_text))
    np.random.shuffle(pairs)
    pairs = pairs[:limit]

    inputs = []
    targets = []
    for inp, tgt in pairs:
        input_ids = tokenizer.encode(inp, max_length=max_len, truncation=True, padding="max_length")
        target_ids = tokenizer.encode(tgt, max_length=max_len, truncation=True, padding="max_length")
        inputs.append(input_ids)
        targets.append(target_ids[0])

    return jnp.array(inputs), jnp.array(targets), tokenizer.vocab_size

def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones((1, 30)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def main():
    inputs, targets, vocab_size = prepare_data()
    model = ChatMLP(hidden_size=256, vocab_size=vocab_size)

    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, learning_rate=1e-3)

    print("\n🚀 Starting Alpa Training")
    for epoch in range(1, 11):
        start = time.time()
        cpu_before = psutil.cpu_percent(interval=None)
        gpu_mem = get_gpu_memory()

        perm = np.random.permutation(len(inputs))
        inputs_shuffled = inputs[perm]
        targets_shuffled = targets[perm]

        for i in range(0, len(inputs), 64):
            batch_inputs = inputs_shuffled[i:i+64]
            batch_labels = targets_shuffled[i:i+64]
            state = train_step(state, batch_inputs, batch_labels)

        loss = compute_loss(state, inputs[:64], targets[:64])
        end = time.time()
        cpu_after = psutil.cpu_percent(interval=None)
        energy = ((cpu_before + cpu_after) / 2) * (end - start)

        print(f"Epoch {epoch:2d}: Train Loss = {loss:.4f}, Time = {end-start:.2f}s, Energy = {energy:.2f}, Mem = {gpu_mem:.2f} MB")

if __name__ == "__main__":
    main()
