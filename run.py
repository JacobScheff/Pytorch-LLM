import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchtext.data.utils import get_tokenizer
from transformers import GPT2Tokenizer

max_token_length = 20
max_output_length = 50
input = "InsideAR was"

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({"pad_token": "<PAD>"}) # Add a PAD token
vocab_size = len(tokenizer)
print(f"Vocab size: {len(tokenizer)}")

def encode(line, truncate=True):
    tokens = tokenizer.tokenize(line)
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens += [tokenizer.pad_token_id] * (max_token_length - len(tokens))
    if truncate and len(tokens) > max_token_length:
        # Remove the first tokens
        tokens = tokens[-max_token_length:]
    return tokens

def decode(tokens):
    output = tokenizer.convert_ids_to_tokens(tokens)
    output = [token.replace("Ġ", " ") for token in output]
    return output

# Load the model
print("Loading model...")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed_size = 120
        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.positional_embedding = nn.Embedding(max_token_length, self.embed_size)
        self.f1 = nn.Linear(max_token_length * self.embed_size, 1_000)
        self.f2 = nn.Linear(1_000, 1_000)
        self.f3 = nn.Linear(1_000, vocab_size)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_size, num_heads=8, device=device)

    def forward(self, x):
        vocab_x = self.embedding(x)
        pos_x = self.positional_embedding(torch.arange(max_token_length).to(device))
        x = vocab_x + pos_x

        x = x.permute(1, 0, 2) # Change to (seq_len, batch, embed_size)
        attn_output, _ = self.attention(x, x, x)
    
        x = self.flatten(attn_output.permute(1,0,2))
        
        x = self.relu(self.f1(x))
        x = self.relu(self.f2(x))
        x = self.f3(x)
        return x

model = Net().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval() # Set the model to evaluation mode

# Run the model
print("Running...")
print(input, end="")

output_string = input
for _ in range(max_output_length):
    encoded_input = encode(output_string)

    output = model(torch.tensor([encoded_input]).to(device))[0]
    output = torch.softmax(output, dim=0)
    output = torch.argmax(output).item()

    output = decode([output])[0]

    print(output, end="")
    output_string += "" + output