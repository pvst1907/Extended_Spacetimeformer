import torch
import torch.nn as nn
from attention import LocalSelfAttention, GlobalSelfAttention, LocalCrossAttention, GlobalCrossAttention
from pcoding import EmbeddingGenerator


class Decoder(nn.Module):
    def __init__(self, xformer):
        super().__init__()
        self.max_seq_length = xformer.max_seq_length  # N_w
        self.input_size = xformer.input_size
        self.embedding_size_time = xformer.embedding_size_time
        self.embedding_size_variable = xformer.embedding_size_variable
        self.embedding_size = xformer.embedding_size

        self.target_embedding = EmbeddingGenerator(self.embedding_size_time, self.embedding_size_variable, self.input_size, self.max_seq_length)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm1 = nn.LayerNorm(self.embedding_size)
        # Self-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.local_attention_layer = LocalSelfAttention(xformer, self.input_size, self.embedding_size, masked=True)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm2 = nn.LayerNorm(self.embedding_size)
        # Self-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.global_attention_layer = GlobalSelfAttention(xformer, self.embedding_size, masked=True)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm3 = nn.LayerNorm(self.embedding_size)
        # Cross-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.local_cross_attention_layer = LocalCrossAttention(xformer, self.input_size, self.embedding_size, masked=True)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm4 = nn.LayerNorm(self.embedding_size)
        # Self-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.global_cross_attention_layer = GlobalCrossAttention(xformer, self.embedding_size, masked=True)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm6 = nn.LayerNorm(self.embedding_size)
        # FFN  (in: (N_w x M) out: (N_w x M))
        self.W1 = nn.Linear(self.embedding_size, self.embedding_size)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm7 = nn.LayerNorm(self.embedding_size)
        # Output Layer in: (N_w x M) out: (N_w x 1))
        self.output_layer = nn.Linear(in_features=xformer.embedding_size, out_features=xformer.output_size)

    def forward(self, sequence_trg, sequence_src):
        sequence_trg = sequence_trg.float()
        sequence_src = sequence_src.float()
        time_index_sequence = torch.flatten(torch.cumsum(torch.full(sequence_trg.size(), 1), 2), 1, 2)
        variable_index_sequence = torch.flatten(torch.cumsum(torch.tile(torch.full((sequence_trg.shape[2],), 1), (sequence_trg.shape[0], sequence_trg.shape[1], 1)), 1), 1, 2)
        # TODO: Add Target Padding
        embedded_sequence_trg = self.target_embedding(sequence_trg, time_index_sequence, variable_index_sequence)
        normed_sequence_trg = self.norm1(embedded_sequence_trg)
        local_attention_trg = self.local_attention_layer(normed_sequence_trg)
        normed_local_attention_trg = self.norm2(embedded_sequence_trg + local_attention_trg)
        global_attention_trg = self.global_attention_layer(normed_local_attention_trg)
        normed_global_attention_trg = self.norm3(normed_local_attention_trg + global_attention_trg)
        local_attention = self.local_cross_attention_layer(normed_global_attention_trg, sequence_src)
        normed_local_attention = self.norm4(normed_global_attention_trg + local_attention)
        global_attention = self.global_attention_layer(normed_local_attention, sequence_src)
        normed_global_attention = self.norm5(normed_local_attention + global_attention)
        decoder_out = nn.ReLU()(self.W1(normed_global_attention))
        normed_encoder_out = self.norm6(normed_global_attention + decoder_out)
        return self.output_layer(normed_encoder_out)
