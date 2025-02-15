import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchtext.data.utils import get_tokenizer

max_token_length = 9
batch_size = 2

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
vocab = sorted(list(vocab))

# Create a mapping from words to indices
word_to_id = {word: i for i, word in enumerate(vocab)}

# Save word_to_id
print(word_to_id)

# # Special tokens
# word_to_id["<PAD>"] = len(word_to_id)
# vocab.append("<PAD>")
# word_to_id["<UNK>"] = len(word_to_id)
# vocab.append("<UNK>")

# def encode(line):
#     tokens = tokenizer(line)
#     tokens = [word_to_id.get(token, word_to_id["<UNK>"]) for token in tokens]
#     tokens += [word_to_id["<PAD>"]] * (max_token_length - len(tokens))
#     return tokens

# def decode(tokens):
#     return [vocab[token] for token in tokens]

# encoded_train_data = [encode(line) for line in train_data]

# X, y = [], []
# for line in encoded_train_data:
#     for i in range(1, len(line)):
#         # Stop at the first padding token
#         if line[i] == word_to_id["<PAD>"]:
#             break

#         X.append(line[:i] + [word_to_id["<PAD>"]] * (max_token_length - i))
#         y.append([line[i]])

# X = torch.tensor(X)
# y = torch.tensor(y)

# # Create a dataset and dataloader
# dataset = torch.utils.data.TensorDataset(X, y)
# dataloader = DataLoader(dataset, batch_size=2)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.embed_size = 10
#         self.embedding = nn.Embedding(len(vocab), self.embed_size)
#         self.f1 = nn.Linear(max_token_length * self.embed_size, 100)
#         self.f2 = nn.Linear(100, 100)
#         self.f3 = nn.Linear(100, len(vocab))
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)
#         self.flatten = nn.Flatten()

#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.flatten(x)

#         x = self.f1(x)
#         x = self.relu(x)

#         x = self.f2(x)
#         x = self.relu(x)

#         x = self.f3(x)
#         x = self.softmax(x)
#         return x
    
# net = Net()

# # Train the model
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# for epoch in range(100):
#     for X_batch, y_batch in dataloader:
#         optimizer.zero_grad()
#         output = net(X_batch)
#         loss = criterion(output, y_batch.squeeze())
#         loss.backward()
#         optimizer.step()
        
#     print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# # Save the model
# torch.save(net.state_dict(), "model.pth")