import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
import json
from transformers import GPT2Tokenizer
from tqdm.auto import tqdm

max_token_length = 20

# Load the training data
print("Loading training data...")
train_data = json.load(open("training_data.json", "r", encoding="utf-8"))[:1]

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({"pad_token": "<PAD>"}) # Add a PAD token
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
    return tokenizer.convert_ids_to_tokens(tokens)

# Encode the training data
print("Encoding training data...")
encoded_train_data = []
for line in tqdm(train_data):
    encoded_train_data.append(encode(line, truncate=False))

# Create X and y
print("Creating X and y...")
X, y = [], []
for line in tqdm(encoded_train_data):
    for i in range(1, len(line)):
        X.append((line[:i] + [tokenizer.pad_token_id] * max(max_token_length - i, 0))[-max_token_length:])
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