import torch.nn as nn
from attention import LocalSelfAttention, GlobalSelfAttention
from pcoding import EmbeddingGenerator


class Encoder(nn.Module):
    def __init__(self, xformer):
        super().__init__()
        self.max_seq_length = xformer.max_seq_length  # N_w
        self.input_size = xformer.input_size
        self.embedding_size_time = xformer.embedding_size_time
        self.embedding_size_variable = xformer.embedding_size_variable
        self.embedding_size = xformer.embedding_size

        self.context_embedding = EmbeddingGenerator()
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm1 = nn.LayerNorm(self.embedding_size)
        # Self-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.local_attention_layer = LocalSelfAttention(xformer, self.input_size, self.embedding_size)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm2 = nn.LayerNorm(self.embedding_size)
        # Self-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.global_attention_layer = GlobalSelfAttention(xformer, self.embedding_size)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm3 = nn.LayerNorm(self.embedding_size)
        # FFN  (in: (N_w x M) out: (N_w x M))
        self.W1 = nn.Linear(self.embedding_size, self.embedding_size)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm4 = nn.LayerNorm(self.embedding_size)

    def forward(self, sequence):
        sequence = sequence.float()
        # TODO: Add Time & TS Index Series Input
        # TODO: Flattening & Create Index Series
        embedded_sequence = self.context_embedding(sequence, time_index_sequence, variable_index_sequence)
        normed_sequence = self.norm1(embedded_sequence)
        local_attention = self.local_attention_layer(normed_sequence)
        normed_local_attention = self.norm2(embedded_sequence + local_attention)
        global_attention = self.global_attention_layer(normed_local_attention)
        normed_global_attention = self.norm3(normed_local_attention + global_attention)
        encoder_out = nn.ReLU()(self.W1(normed_global_attention))
        normed_encoder_out = self.norm4(normed_global_attention + encoder_out)
        return normed_encoder_out
