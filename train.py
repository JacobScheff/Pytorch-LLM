import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchtext.data.utils import get_tokenizer
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer

torch.manual_seed(0) # Set seed for reproducibility

max_token_length = 20
batch_size = 64

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Load the training data
print("Loading training data...")
dataset = torch.load("dataset.pth", weights_only=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({"pad_token": "<PAD>"}) # Add a PAD token
vocab_size = len(tokenizer)

# Create the model
print("Creating model...")
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

net = Net().to(device)

# Print the total number of parameters
total_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {total_params:,}")

# Train the model
print("Training model...")
criterion = nn.CrossEntropyLoss() # Automatically applies softmax
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
for epoch in range(100):
    if epoch == 50:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    for X_batch, y_batch in (dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = net(X_batch)
        loss = criterion(output, y_batch.squeeze())
        loss.backward()
        optimizer.step()
    
    # Save the model every few epochs
    # if (epoch + 1) % 5 == 0:
    #     torch.save(net.state_dict(), f"models/model_{epoch + 1}.pth")
            
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the model
torch.save(net.state_dict(), "model.pth")