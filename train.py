import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer

torch.manual_seed(0) # Set seed for reproducibility

max_token_length = 20
batch_size = 256

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
if device == "cuda":
    print(f"Device ID: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

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
class AttentionBlock(nn.Module):
    def __init__(self, embed_size, device="cpu"):
        super(AttentionBlock, self).__init__()
        self.embed_size = embed_size

        self.multi_head_attention = nn.MultiheadAttention(embed_dim=self.embed_size, num_heads=8, device=device, batch_first=True) # outputs: (batch_size, seq_len, embed_size)
        self.normaliztion = nn.LayerNorm(self.embed_size) # outputs: (batch_size, seq_len, embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size * 4),
            nn.ReLU(),
            nn.Linear(self.embed_size * 4, self.embed_size)
        ) # outputs: (batch_size, seq_len, embed_size)

    def forward(self, x):    
        attn_output, _ = self.multi_head_attention(x, x, x)
        x = x + attn_output

        x = self.normaliztion(x)

        feed_forward_output = self.feed_forward(x)
        x = x + feed_forward_output

        x = self.normaliztion(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed_size = 192
        self.num_attention_blocks = 8

        self.token_embedding = nn.Embedding(vocab_size, self.embed_size)
        self.positional_embedding = nn.Embedding(max_token_length, self.embed_size)
        self.pos_indices = torch.arange(max_token_length).to(device)

        self.attention_blocks = nn.ModuleList([
            AttentionBlock(self.embed_size, device=device)
            for _ in range(self.num_attention_blocks)
        ])

        self.flatten = nn.Flatten() # outputs: (batch_size, seq_len * embed_size)
        self.linear = nn.Linear(max_token_length * self.embed_size, vocab_size) # outputs: (batch_size, vocab_size)

    def forward(self, x):
        token_x = self.token_embedding(x)
        pos_x = self.positional_embedding(self.pos_indices)
        x = token_x + pos_x

        # Iterate through the attention blocks
        for block in self.attention_blocks:
            x = block(x)

        x = self.flatten(x)
        x = self.linear(x)

        return x # Softmax is automatically applied in the loss function

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

    # Create a progress bar with a loss label
    # bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}", dynamic_ncols=True)

    # for i, (X_batch, y_batch) in bar:
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = net(X_batch)
        loss = criterion(output, y_batch.flatten())
        loss.backward()
        optimizer.step()
        # bar.set_postfix(loss=loss.item())

    # Save the model every few epochs
    # if (epoch + 1) % 1 == 0:
    #     torch.save(net.state_dict(), f"models/model_{epoch + 1}.pth")

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the model
print("Saving model...")
torch.save(net.state_dict(), "model.pth")