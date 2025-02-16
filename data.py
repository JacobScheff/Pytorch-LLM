import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchtext.data.utils import get_tokenizer
import json

max_token_length = 20

train_data = json.load(open("training_data.json", "r"))[:1000]

# Create a tokenizer based off of the training data
tokenizer = get_tokenizer("basic_english")
vocab = set()
for line in train_data:
    vocab.update(tokenizer(line))
vocab = sorted(list(vocab))
print("vocab size: " + str(len(vocab)))

# Create a mapping from words to indices
word_to_id = {word: i for i, word in enumerate(vocab)}

# Special tokens
word_to_id["<PAD>"] = len(word_to_id)
vocab.append("<PAD>")
word_to_id["<UNK>"] = len(word_to_id)
vocab.append("<UNK>")

# Save word_to_id and vocab
torch.save(word_to_id, "word_to_id.pth")
torch.save(vocab, "vocab.pth")

def encode(line, truncate=True):
    tokens = tokenizer(line)
    tokens = [word_to_id.get(token, word_to_id["<UNK>"]) for token in tokens]
    tokens += [word_to_id["<PAD>"]] * (max_token_length - len(tokens))
    if truncate and len(tokens) > max_token_length:
        # Remove the first tokens
        tokens = tokens[-max_token_length:]
    return tokens

def decode(tokens):
    return [vocab[token] for token in tokens]

# Encode the training data
print("Encoding training data...")
encoded_train_data = [encode(line, truncate=False) for line in train_data]

# Create X and y
print("Creating X and y...")
X, y = [], []
for line in encoded_train_data:
    for i in range(1, len(line)):
        X.append((line[:i] + [word_to_id["<PAD>"]] * max(max_token_length - i, 0))[-max_token_length:])
        y.append([line[i]])

# Convert to tensors
print("Converting to tensors...")
X = torch.tensor(X)
y = torch.tensor(y)

# Create a dataset and dataloader
print("Creating dataset and dataloader...")
dataset = torch.utils.data.TensorDataset(X, y)

# Save the dataset
print("Saving dataset...")
torch.save(dataset, "dataset.pth")

print("Done! Created {} training examples.".format(len(X)))