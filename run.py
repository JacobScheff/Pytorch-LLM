import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchtext.data.utils import get_tokenizer

# Load the tokenizer
tokenizer = get_tokenizer("basic_english")
word_to_id = torch.load("word_to_id.pth")