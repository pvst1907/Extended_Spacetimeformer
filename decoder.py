import torch
import torch.nn as nn
from attention import LocalSelfAttention, GlobalSelfAttention, LocalCrossAttention, GlobalCrossAttention
from pcoding import EmbeddingGenerator


class Decoder(nn.Module):
    def __init__(self, xformer):
        super().__init__()
        self.trg_seq_length = xformer.trg_seq_length  # N_w
        self.src_seq_length = xformer.src_seq_length
        self.output_size = xformer.output_size
        self.embedding_size_time = xformer.embedding_size_time
        self.embedding_size_variable = xformer.embedding_size_variable
        self.embedding_size = xformer.embedding_size
        self.s_qkv = xformer.s_qkv

        self.target_embedding = EmbeddingGenerator(self.embedding_size_time, self.embedding_size_variable, self.output_size, self.trg_seq_length)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm1 = nn.LayerNorm(self.embedding_size)
        # Self-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.local_attention_layer = LocalSelfAttention(self.output_size, self.trg_seq_length, self.embedding_size, self.s_qkv, masked=True)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm2 = nn.LayerNorm(self.s_qkv)
        # Self-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.global_attention_layer = GlobalSelfAttention(self.trg_seq_length, self.embedding_size, self.s_qkv, masked=True)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm3 = nn.LayerNorm(self.s_qkv)
        # Cross-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.local_cross_attention_layer = LocalCrossAttention(self.output_size, self.src_seq_length, self.trg_seq_length, self.embedding_size, self.s_qkv, masked=True)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm4 = nn.LayerNorm(self.s_qkv)
        # Self-Attention Layer Local (in: (N_w x M) out: (N_w x M))
        self.global_cross_attention_layer = GlobalCrossAttention(self.src_seq_length, self.trg_seq_length, self.embedding_size, self.s_qkv, masked=True)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm5 = nn.LayerNorm(self.s_qkv)
        # FFN  (in: (N_w x M) out: (N_w x M))
        self.W1 = nn.Linear(self.s_qkv, self.s_qkv)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm6 = nn.LayerNorm(self.s_qkv)
        # Output Layer in: (N_w x M) out: (N_w x 1))
        self.output_layer = nn.Linear(in_features=xformer.s_qkv, out_features=xformer.output_size)

    def forward(self, sequence_trg, sequence_src):
        # Flattening
        sequence_trg_flat = torch.unsqueeze(torch.flatten(sequence_trg, 1, 2), dim=2)
        sequence_src_flat = torch.unsqueeze(torch.flatten(sequence_src, 1, 2), dim=2)

        # Time & Variable Encoding/Embedding
        time_index_sequence = torch.flatten(torch.cumsum(torch.full(sequence_trg.size(), 1), 2), 1, 2)
        variable_index_sequence = torch.flatten(torch.cumsum(torch.tile(torch.full((sequence_trg.shape[2],), 1), (sequence_trg.shape[0], sequence_trg.shape[1], 1)), 1), 1, 2)-1
        embedded_sequence_trg = self.target_embedding(sequence_trg_flat, time_index_sequence, variable_index_sequence)

        # Norm
        normed_sequence_trg = self.norm1(embedded_sequence_trg)

        # Local Self Attetion
        local_attention_trg = self.local_attention_layer(normed_sequence_trg)

        # Norm
        normed_local_attention_trg = self.norm2(embedded_sequence_trg + local_attention_trg)

        # Global Self Attention
        global_attention_trg = self.global_attention_layer(normed_local_attention_trg)

        # Norm
        normed_global_attention_trg = self.norm3(normed_local_attention_trg + global_attention_trg)

        # Local Cross Attention
        local_attention = self.local_cross_attention_layer(normed_global_attention_trg, sequence_src_flat)

        # Norm
        normed_local_attention = self.norm4(normed_global_attention_trg + local_attention)

        # Global Cross Attention
        global_attention = self.global_cross_attention_layer(normed_local_attention, sequence_src_flat)

        # Norm
        normed_global_attention = self.norm5(normed_local_attention + global_attention)

        # Linear Layer & ReLU
        decoder_out = nn.ReLU()(self.W1(normed_global_attention))

        # Norm
        normed_encoder_out = self.norm6(normed_global_attention + decoder_out)

        # Linear Layer
        return self.output_layer(normed_encoder_out)
