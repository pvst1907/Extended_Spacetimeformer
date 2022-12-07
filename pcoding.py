import torch.nn as nn
import torch
import math


class EmbeddingGenerator(nn.Module):
    def __init__(self):
        super().__init__()
″
    def forward(self, sequence):
        return sequence