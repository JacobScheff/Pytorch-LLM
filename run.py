import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import GPT2Tokenizer

max_token_length = 200
max_output_length = 500
input = "The substitution principle in sustainability is the maxim that processes, services and products should, wherever possible, be replaced with"
model_path = "models/model_2.pth"

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({"pad_token": "<PAD>"}) # Add a PAD token
tokenizer.add_special_tokens({"bos_token": "<BOS>"}) # Add a BOS token (beginning of sentence)
tokenizer.add_special_tokens({"eos_token": "<EOS>"}) # Add a EOS token (end of sentence)
vocab_size = len(tokenizer)
print(f"Vocab size: {len(tokenizer)}")

def encode(line):
    tokens = tokenizer.tokenize(line)
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens += [tokenizer.pad_token_id] * (max_token_length - len(tokens))
    return tokens

def decode(tokens):
    return tokenizer.convert_ids_to_tokens(tokens)

def get_token_count(line):
    tokens = tokenizer.tokenize(line)
    return len(tokens)

# Load the model
print("Loading model...")
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

        def forward(self, x, key_padding_mask, attn_mask):    
            attn_output, _ = self.multi_head_attention(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask) # outputs: (batch_size, seq_len, embed_size)
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
        self.attn_mask = torch.triu(torch.ones(max_token_length, max_token_length, device=device) == 1, diagonal=1)

        self.attention_blocks = nn.ModuleList([
            AttentionBlock(self.embed_size, device=device)
            for _ in range(self.num_attention_blocks)
        ])

        self.linear = nn.Linear(self.embed_size, vocab_size, device=device) # outputs: (batch_size, seq_len, vocab_size)

    def forward(self, x):
        # Create mask for padding tokens
        key_padding_mask = (x == tokenizer.pad_token_id)

        token_x = self.token_embedding(x)
        pos_x = self.positional_embedding(self.pos_indices)
        x = token_x + pos_x

        # Iterate through the attention blocks
        for block in self.attention_blocks:
            x = block(x, key_padding_mask, self.attn_mask)

        x = self.linear(x)

        return x # Softmax is automatically applied in the loss function

model = Net().to(device)
model.load_state_dict(torch.load(model_path))
model.eval() # Set the model to evaluation mode

# Run the model
print("Running...")
print(input, end="")

output_string = "<BOS>" + input
predicted_token_index = get_token_count(input)
if predicted_token_index >= max_token_length:
    predicted_token_index = max_token_length - 1
for _ in range(max_output_length):
    encoded_input = encode(output_string)
    if len(encoded_input) > max_token_length:
        # Keep only the last max_token_length tokens
        encoded_input = encoded_input[-max_token_length:]

    output = model(torch.tensor([encoded_input]).to(device))[0]
    output = output[predicted_token_index]
    output = torch.softmax(output, dim=0)
    output = torch.argmax(output).item()

    output = decode([output])[0]
    output = output.replace("Ġ", " ")

    print(output, end="")
    output_string += "" + output
    predicted_token_index += 1
    if predicted_token_index >= max_token_length:
        predicted_token_index = max_token_length - 1
    
    if output == "<EOS>":
        break