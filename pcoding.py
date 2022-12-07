import torch.nn as nn
import torch


class EmbeddingGenerator(nn.Module):
    def __init__(self, embedding_size_time, embedding_size_variable, input_size, max_seq_length):
        super().__init__()
        self.variable_emb_generator = VariableEmbedding(input_size, embedding_size_variable)

    def forward(self, sequence, time_index_sequence, variable_index_sequence):
        var_embedding = self.variable_emb_generator(torch.squeeze(variable_index_sequence)).transpose(2, 1)
        return torch.cat((sequence, var_embedding), 2)


class VariableEmbedding(nn.Module):

    def __init__(self, num_variables, embedding_size):
        super().__init__()
        self.embed = nn.Embedding(num_variables, embedding_size)

    def forward(self, sequence):
        return self.embed(sequence)
