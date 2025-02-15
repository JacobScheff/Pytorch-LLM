import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchtext.data.utils import get_tokenizer

train_data = [
    "this is a sentence",
    "two words",
    "three words here",
    "there are five words here",
    "the quick brown fox jumps over the lazy dog",
]

tokenizer = get_tokenizer("basic_english")
train_data = [tokenizer(item) for item in train_data]