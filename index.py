import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchtext.data.utils import get_tokenizer

max_token_length = 9

train_data = [
    "this is a sentence",
    "two words",
    "three words here",
    "there are five words here",
    "the quick brown fox jumps over the lazy dog",
]

# Create a tokenizer based off of the training data
tokenizer = get_tokenizer("basic_english")
vocab = set()
for line in train_data:
    vocab.update(tokenizer(line))
vocab = list(vocab)

# Create a mapping from words to indices
word_to_idx = {word: i for i, word in enumerate(vocab)}

# Special tokens
word_to_idx["<PAD>"] = len(word_to_idx)
vocab.append("<PAD>")
word_to_idx["<UNK>"] = len(word_to_idx)
vocab.append("<UNK>")

def encode(line):
    tokens = tokenizer(line)
    tokens = [word_to_idx.get(token, word_to_idx["<UNK>"]) for token in tokens]
    tokens += [word_to_idx["<PAD>"]] * (max_token_length - len(tokens))
    return tokens

def decode(tokens):
    return [vocab[token] for token in tokens]

training_data = [encode(line) for line in train_data]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed_size = 10
        self.embedding = nn.Embedding(len(vocab), self.embed_size)
        self.f1 = nn.Linear(max_token_length * self.embed_size, 100)
        self.f2 = nn.Linear(100, 100)
        self.f3 = nn.Linear(100, len(vocab))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)

        x = self.f1(x)
        x = self.relu(x)

        x = self.f2(x)
        x = self.relu(x)

        x = self.f3(x)
        x = self.softmax(x)
        return x
    
net = Net()
print(net)