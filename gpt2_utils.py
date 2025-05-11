from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_gpt2_model_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model
