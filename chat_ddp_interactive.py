import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# === Load tokenizer ===
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# === Load GPT-2 architecture ===
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load your trained DDP weights ===
checkpoint_path = "checkpoints/ddp_best_model.pt"
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# === Interactive Chat Loop ===
print("🤖 Chatbot (DDP-trained GPT-2) is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    input_text = user_input + tokenizer.eos_token
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=128,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print("Bot:", response.strip())
