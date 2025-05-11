# warmup_cache.py

from transformers import AutoTokenizer, AutoModel
import os

# Optional: use a shared Hugging Face cache directory (helpful for Vast.ai multi-GPU)
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
os.makedirs("/workspace/hf_cache", exist_ok=True)

model_name = "distilbert-base-uncased"

print(f"🔄 Downloading and caching tokenizer for '{model_name}'...")
AutoTokenizer.from_pretrained(model_name)

print(f"🔄 Downloading and caching model for '{model_name}'...")
AutoModel.from_pretrained(model_name)

print("✅ Hugging Face model and tokenizer are cached locally.")
