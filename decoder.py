import torch.nn as nn
from attention import SelfAttention, CrossAttention


class MasterDecoderWithMasking(nn.Module):
    def __init__(self, xformer, num_basic_decoders, num_atten_heads):
        super().__init__()
        self.max_seq_length = xformer.max_seq_length  # N_w
        self.embedding_size = xformer.embedding_size  # M

        self.input_layer = nn.Linear(in_features=xformer.input_size, out_features=xformer.embedding_size)
        self.basic_decoder_arr = nn.ModuleList([BasicDecoderWithMasking(xformer, num_atten_heads) for _ in range(num_basic_decoders)])
        self.output_layer = nn.Linear(in_features=xformer.embedding_size, out_features=xformer.output_size)

    def forward(self, sentence_tensor, final_encoder_out):
        out_tensor = sentence_tensor
        out_tensor = self.input_layer(out_tensor)
        for i in range(len(self.basic_decoder_arr)):
            out_tensor = self.basic_decoder_arr[i](out_tensor, final_encoder_out)
        return self.output_layer(out_tensor)


class BasicDecoderWithMasking(nn.Module):
    def __init__(self, xformer, num_atten_heads):
        super().__init__()
        self.embedding_size = xformer.embedding_size
        self.max_seq_length = xformer.max_seq_length
        self.num_atten_heads = num_atten_heads
        self.qkv_size = self.embedding_size // num_atten_heads

        # Self-Attention Layer (in: (N_w x M) out: (N_w x M))
        self.self_attention_layer = SelfAttention(xformer, num_atten_heads, masked=True)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm1 = nn.LayerNorm(self.embedding_size)
        # Cross-Attention Layer (in: (N_w x M) out: (N_w x M))
        self.cross_attn_layer = CrossAttention(xformer, num_atten_heads, masked=True)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm2 = nn.LayerNorm(self.embedding_size)
        # FFN 1  (in: (N_w x M) out: (N_w x M))
        self.W1 = nn.Linear(self.max_seq_length * self.embedding_size, self.max_seq_length * 2 * self.embedding_size)
        # FFN 1  (in: (N_w x M) out: (N_w x M))
        self.W2 = nn.Linear(self.max_seq_length * 2 * self.embedding_size, self.max_seq_length * self.embedding_size)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm3 = nn.LayerNorm(self.embedding_size)

    def forward(self, sentence_tensor, final_encoder_out):
        self_atten_out = self.self_attention_layer(sentence_tensor)
        normed_atten_out = self.norm1(self_atten_out + sentence_tensor)
        cross_atten_out = self.cross_attn_layer(normed_atten_out, final_encoder_out)
        normed_cross_atten_out = self.norm2(cross_atten_out)
        basic_decoder_out = nn.ReLU()(self.W1(normed_cross_atten_out.view(sentence_tensor.shape[0], -1)))
        basic_decoder_out = self.W2(basic_decoder_out)
        basic_decoder_out = basic_decoder_out.view(sentence_tensor.shape[0], self.max_seq_length, self.embedding_size)
        basic_decoder_out = self.norm3(basic_decoder_out + normed_cross_atten_out)
        return basic_decoder_out
