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
train_data = json.load(open("training_data.json", "r", encoding="utf-8"))

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({"pad_token": "<PAD>"}) # Add a PAD token
tokenizer.add_special_tokens({"bos_token": "<BOS>"}) # Add a BOS token (beginning of sentence)
tokenizer.add_special_tokens({"eos_token": "<EOS>"}) # Add a EOS token (end of sentence)
print(f"Vocab size: {len(tokenizer)}")

def encode(line):
    tokens = tokenizer.tokenize(line)
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
    tokens += [tokenizer.pad_token_id] * (max_token_length - len(tokens))
    return tokens

def decode(tokens):
    return tokenizer.convert_ids_to_tokens(tokens)

# Encode the training data
print("Encoding training data...")
encoded_train_data = []
for line in tqdm(train_data):
    encoded_train_data.append(encode(line))

# Create X and y
print("Creating X and y...")
X, y = [], []
for line in tqdm(encoded_train_data):
    # Calculate the number of tokens past the max_token_length
    extra_tokens = len(line) - max_token_length
    if extra_tokens <= 0:
        eos_index = line.index(tokenizer.eos_token_id)
        x_line = line[:eos_index]
        x_line += [tokenizer.pad_token_id] * (max_token_length - len(x_line))
        X.append(x_line)
        y_line = line[1:eos_index+1]
        y_line += [tokenizer.pad_token_id] * (max_token_length - len(y_line))
        y.append(y_line)
    else:
        # Line is longer than max_token
        for i in range(extra_tokens):
            X.append(line[i:i+max_token_length])
            y_line = line[i+1:i+max_token_length+1]
            y_line += [tokenizer.pad_token_id] * (max_token_length - len(y_line))
            y.append(y_line)

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