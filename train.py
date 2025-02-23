import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer
from torch.cuda.amp import autocast, GradScaler
import os

def run():
    torch.manual_seed(0) # Set seed for reproducibility

    max_token_length = 20
    batch_size = 256

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    # device = "cpu"
    print(f"Using {device} device")
    if device == "cuda":
        print(f"Device ID: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # Load the training data
    print("Loading training data...")
    dataset = torch.load("dataset.pth", weights_only=False)

    # Get the number of CPU cores, but don't use all of them
    num_cpus = os.cpu_count()
    num_workers = max(1, num_cpus - 4) if num_cpus else 0  # Leave a couple of cores free, handle None case
    print(f"Using {num_workers} workers for data loading.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({"pad_token": "<PAD>"}) # Add a PAD token
    tokenizer.add_special_tokens({"bos_token": "<BOS>"}) # Add a BOS token (beginning of sentence)
    tokenizer.add_special_tokens({"eos_token": "<EOS>"}) # Add a EOS token (end of sentence)
    vocab_size = len(tokenizer)

    # Create the model
    print("Creating model...")
    class AttentionBlock(nn.Module):
        def __init__(self, embed_size, device="cpu"):
            super(AttentionBlock, self).__init__()
            self.embed_size = embed_size

            self.multi_head_attention = nn.MultiheadAttention(embed_dim=self.embed_size, num_heads=8, device=device, batch_first=True) # outputs: (batch_size, seq_len, embed_size)
            self.normaliztion = nn.LayerNorm(self.embed_size, device=device) # outputs: (batch_size, seq_len, embed_size)
            self.feed_forward = nn.Sequential(
                nn.Linear(self.embed_size, self.embed_size * 4, device=device),
                nn.ReLU(),
                nn.Linear(self.embed_size * 4, self.embed_size, device=device)
            ) # outputs: (batch_size, seq_len, embed_size)

        def forward(self, x, mask):    
            attn_output, _ = self.multi_head_attention(x, x, x, key_padding_mask=mask) # outputs: (batch_size, seq_len, embed_size)
            x = x + attn_output

            x = self.normaliztion(x)

            feed_forward_output = self.feed_forward(x)
            x = x + feed_forward_output

            x = self.normaliztion(x)
            return x

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.embed_size = 256
            self.num_attention_blocks = 8

            self.token_embedding = nn.Embedding(vocab_size, self.embed_size, device=device)
            self.positional_embedding = nn.Embedding(max_token_length, self.embed_size, device=device)
            self.pos_indices = torch.arange(max_token_length, device=device)

            self.attention_blocks = nn.ModuleList([
                AttentionBlock(self.embed_size, device=device)
                for _ in range(self.num_attention_blocks)
            ])

            self.linear = nn.Linear(self.embed_size, vocab_size, device=device) # outputs: (batch_size, seq_len, vocab_size)

        def forward(self, x):
            # Create mask for padding tokens. This needs to be a byte tensor
            mask = (x == tokenizer.pad_token_id)

            token_x = self.token_embedding(x)
            pos_x = self.positional_embedding(self.pos_indices)
            x = token_x + pos_x

            # Iterate through the attention blocks
            for block in self.attention_blocks:
                x = block(x, mask)

            x = self.linear(x)

            return x # Softmax is automatically applied in the loss function

    net = Net().to(device)

    # Print the total number of parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")

    # TODO: Figure out how to get triton
    # net = torch.compile(net)

    # Train the model
    print("Training model...")
    net.train() # Set the model to training mode
    criterion = nn.CrossEntropyLoss() # Automatically applies softmax
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)#0.001
    scaler = GradScaler() # Mixed precision training
    for epoch in range(100):
        # if epoch == 50:
        #     optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

        # Create a progress bar with a loss label
        bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}", dynamic_ncols=True)

        for i, (X_batch, y_batch) in bar:
        # for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # Use autocast for mixed precision
            with autocast():
                output = net(X_batch)
                output = output.reshape(-1, vocab_size)
                loss = criterion(output, y_batch.flatten())

            # Scale the loss and call backward() on the scaled loss
            scaler.scale(loss).backward()

            # Step the optimizer and update the scaler
            scaler.step(optimizer)
            scaler.update()

            bar.set_postfix(loss=loss.item())

        # Save the model every few epochs
        if (epoch + 1) % 1 == 0:
            torch.save(net.state_dict(), f"models/model_{epoch + 1}.pth")

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Save the model
    print("Saving model...")
    torch.save(net.state_dict(), "model.pth")



if __name__ == "__main__":
    run()