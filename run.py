import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchtext.data.utils import get_tokenizer

max_token_length = 20
max_output_length = 50
input = "This model suggests that"

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def encode(line):
    tokens = tokenizer(line)
    tokens = [word_to_id.get(token, word_to_id["<UNK>"]) for token in tokens]
    tokens += [word_to_id["<PAD>"]] * (max_token_length - len(tokens))
    if len(tokens) > max_token_length:
        # Remove the first tokens
        tokens = tokens[-max_token_length:]
    return tokens

def decode(tokens):
    return [vocab[token] for token in tokens]

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = get_tokenizer("basic_english")
word_to_id = torch.load("word_to_id.pth")

# Load the vocab
print("Loading vocab...")
vocab = torch.load("vocab.pth")

# Load the model
print("Loading model...")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed_size = 120
        self.embedding = nn.Embedding(len(vocab), self.embed_size)
        self.positional_embedding = nn.Embedding(max_token_length, self.embed_size)
        self.f1 = nn.Linear(max_token_length * self.embed_size, 1_000)
        self.f2 = nn.Linear(1_000, 1_000)
        self.f3 = nn.Linear(1_000, len(vocab))
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
model.load_state_dict(torch.load("models/model_1.pth"))
model.eval() # Set the model to evaluation mode

# Run the model
print("Running...")
print(input, end=" ")

output_string = input
for _ in range(max_output_length):
    encoded_input = encode(output_string)

    output = model(torch.tensor([encoded_input]).to(device))[0]
    output = torch.softmax(output, dim=0)
    output = torch.argmax(output).item()

    output = vocab[output]

    print(output, end=" ")
    output_string += " " + output