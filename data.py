import torch
from torch import nn
from torch.utils.data import DataLoader
import json
from transformers import GPT2Tokenizer
from tqdm.auto import tqdm

max_token_length = 200

# Load the training data
print("Loading training data...")
train_data = json.load(open("training_data.json", "r", encoding="utf-8"))[:10000]

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({"pad_token": "<PAD>"}) # Add a PAD token
tokenizer.add_special_tokens({"bos_token": "<BOS>"}) # Add a BOS token (beginning of sentence)
tokenizer.add_special_tokens({"eos_token": "<EOS>"}) # Add a EOS token (end of sentence)
print(f"Vocab size: {len(tokenizer)}")

def encode_old(line):
    tokens = tokenizer.tokenize(line)
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
    tokens += [tokenizer.pad_token_id] * (max_token_length - len(tokens))
    return tokens

def decode(tokens):
    return tokenizer.convert_ids_to_tokens(tokens)

# Encode the training data
print("Encoding training data...")
encoded_train_data_old = []
for line in tqdm(train_data):
    encoded_train_data_old.append(encode_old(line))

print("Preparing data...")
train_data_with_special_tokens = [tokenizer.bos_token + line + tokenizer.eos_token for line in train_data]
print("Encoding training data...")
encoding_batch_size = 128
encoded_train_data = []
for i in tqdm(range(0, len(train_data), encoding_batch_size)):
    encoded_train_data_batch = tokenizer(train_data_with_special_tokens[i:i+encoding_batch_size], padding="max_length", truncation=True, max_length=max_token_length, return_tensors="pt")
    for j in range(len(encoded_train_data_batch["input_ids"])):
        encoded_train_data.append(encoded_train_data_batch["input_ids"][j])

# encoded_train_data = tokenizer(train_data_with_special_tokens, padding="max_length", truncation=True, max_length=max_token_length, return_tensors="pt")

# Check if the new and old encodings are the same
# for i in range(len(encoded_train_data_old)):
#     old = encoded_train_data_old[i]
#     new = encoded_train_data["input_ids"][i]
#     if len(old) != len(new):
#         print("Different lengths!")
#     for j in range(len(old)):
#         if old[j] != new[j]:
#             print("Different values!")
#             break

# # Create X and y
# print("Creating X and y...")
# X, y = [], []
# for line in tqdm(encoded_train_data):
#     # Calculate the number of tokens past the max_token_length
#     extra_tokens = len(line) - max_token_length
#     if extra_tokens > 0:
#         x_line = line[:max_token_length]
#         X.append(x_line)
#     else:
#         x_line = line + [tokenizer.pad_token_id] * (max_token_length - len(line))
#         X.append(x_line)

#     eos_index = line.index(tokenizer.eos_token_id)
#     if eos_index >= max_token_length:
#         y_line = line[1:max_token_length + 1]
#         y.append(y_line)
#     else:
#         y_line = line[1:eos_index+1]
#         y_line += [tokenizer.pad_token_id] * (max_token_length - len(y_line))
#         y.append(y_line)

# # Convert to tensors
# print("Converting to tensors...")
# X = torch.tensor(X)
# y = torch.tensor(y)

# # Create the dataset
# dataset = torch.utils.data.TensorDataset(X, y)

# # Save the dataset
# print("Saving dataset...")
# torch.save(dataset, "dataset.pth")

# print("Done! Created {} training examples.".format(len(X)))