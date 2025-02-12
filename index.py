import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = [
    "this is a sentence",
    "two words",
    "three words here",
    "there are five words here",
    "the quick brown fox jumps over the lazy dog",
]

tokenizer = Tokenizer(train_data)