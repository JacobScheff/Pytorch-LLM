import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchtext.data.utils import get_tokenizer

torch.manual_seed(0) # Set seed for reproducibility

max_token_length = 9
batch_size = 2

# Load the training data
print("Loading training data...")
dataset = torch.load("dataset.pth", weights_only=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load the tokenizer and vocab
print("Loading tokenizer and vocab...")
tokenizer = get_tokenizer("basic_english")
word_to_id = torch.load("word_to_id.pth")
vocab = torch.load("vocab.pth")

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
    
net = Net()

# Train the model
criterion = nn.CrossEntropyLoss() # Automatically applies softmax
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(100):
    if epoch == 50:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output = net(X_batch)
        loss = criterion(output, y_batch.squeeze())
        loss.backward()
        optimizer.step()
            
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the model
torch.save(net.state_dict(), "model.pth")