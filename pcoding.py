import torch.nn as nn
import torch
import math


class PositionalEncoder(nn.Module):
    def __init__(self, max_seq_length, embedding_size):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size))
        self.pe = torch.zeros(1, max_seq_length, embedding_size)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)

    def forward(self, sequence):
        sequence = sequence + self.pe[:sequence.size(1)]
        return sequence