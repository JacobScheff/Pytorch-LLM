import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchtext.data.utils import get_tokenizer

max_token_length = 9
input = "the"

def encode(line):
    tokens = tokenizer(line)
    tokens = [word_to_id.get(token, word_to_id["<UNK>"]) for token in tokens]
    tokens += [word_to_id["<PAD>"]] * (max_token_length - len(tokens))
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

# Encode the input
encoded_input = encode(input)

# Load the model
print("Loading model...")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed_size = 20
        self.embedding = nn.Embedding(len(vocab), self.embed_size)
        self.f1 = nn.Linear(max_token_length * self.embed_size, 500)
        self.f2 = nn.Linear(500, 500)
        self.f3 = nn.Linear(500, len(vocab))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)

        x = self.f1(x)
        x = self.relu(x)

        x = self.f2(x)
        x = self.relu(x)

        for _ in range(5):
            x = self.f2(x)
            x = self.relu(x)
        
        x = self.f3(x)
        return x

model = Net()
model.load_state_dict(torch.load("model.pth"))
model.eval() # Set the model to evaluation mode

# Run the model
print("Running...")
print(input, end=" ")
while True:
    output = model(torch.tensor([encoded_input]))[0]
    output = torch.softmax(output, dim=0)
    output = torch.argmax(output).item()

    print(vocab[output], end=" ")
    encoded_input = encoded_input[1:] + [output]

    if len(encoded_input) >= max_token_length:
        break