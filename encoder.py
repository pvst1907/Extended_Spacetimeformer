import torch.nn as nn
from attention import SelfAttention


class MasterEncoder(nn.Module):
    def __init__(self, xformer, num_basic_encoders, num_atten_heads):
        super().__init__()
        self.max_seq_length = xformer.max_seq_length  # N_w
        self.basic_encoder_arr = nn.ModuleList([BasicEncoder(xformer, num_atten_heads) for _ in range(num_basic_encoders)])

    def forward(self, sentence_tensor):
        out_tensor = sentence_tensor
        for i in range(len(self.basic_encoder_arr)):
            out_tensor = self.basic_encoder_arr[i](out_tensor)
        return out_tensor


class BasicEncoder(nn.Module):
    def __init__(self, xformer, num_atten_heads):
        super().__init__()
        self.embedding_size = xformer.embedding_size  # M
        self.max_seq_length = xformer.max_seq_length  # N_w
        self.num_atten_heads = num_atten_heads

        # Self-Attention Layer (in: (N_w x M) out: (N_w x M))
        self.self_attention_layer = SelfAttention(xformer, num_atten_heads)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm1 = nn.LayerNorm(self.embedding_size)
        # FFN 1  (in: (N_w x M) out: (N_w x M))
        self.W1 = nn.Linear(self.max_seq_length * self.embedding_size, self.max_seq_length * 2 * self.embedding_size)
        # FFN 2  (in: (N_w x M) out: (N_w x M))
        self.W2 = nn.Linear(self.max_seq_length * 2 * self.embedding_size, self.max_seq_length * self.embedding_size)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm2 = nn.LayerNorm(self.embedding_size)

    def forward(self, sentence_tensor):
        sentence_tensor = sentence_tensor.float()
        self_atten_out = self.self_attention_layer(sentence_tensor)
        normed_atten_out = self.norm1(self_atten_out + sentence_tensor)
        basic_encoder_out = nn.ReLU()(self.W1(normed_atten_out.view(sentence_tensor.shape[0], -1)))
        basic_encoder_out = self.W2(basic_encoder_out)
        basic_encoder_out = basic_encoder_out.view(sentence_tensor.shape[0], self.max_seq_length, self.embedding_size)
        basic_encoder_out = self.norm2(basic_encoder_out + normed_atten_out)
        return basic_encoder_out
