import torch
import torch.nn as nn


class MasterDecoderWithMasking(nn.Module):
    def __init__(self, xformer, num_basic_decoders, num_atten_heads):
        super().__init__()
        self.max_seq_length = xformer.max_seq_length
        self.embedding_size = xformer.embedding_size
        self.basic_decoder_arr = nn.ModuleList([BasicDecoderWithMasking(xformer, num_atten_heads) for _ in range(num_basic_decoders)])

    def forward(self, sentence_tensor, final_encoder_out):
        mask = torch.ones(1, dtype=int)
        prediction = torch.ones(sentence_tensor.shape[0], self.max_seq_length, dtype=torch.long)
        for mask_index in range(1, sentence_tensor.shape[1]):
            masked_target_sentence = self.apply_mask(sentence_tensor, mask, self.max_seq_length, self.embedding_size)
            out_tensor = masked_target_sentence
            for i in range(len(self.basic_decoder_arr)):
                out_tensor = self.basic_decoder_arr[i](out_tensor, final_encoder_out, mask)
            prediction[:, mask_index] = out_tensor[:,mask_index]
            mask = torch.cat((mask, torch.ones(1, dtype=int)))
        return prediction

    def apply_mask(self, sentence_tensor, mask, max_seq_length, embedding_size):
        out = torch.zeros_like(sentence_tensor).float()
        out[:, :len(mask), :] = sentence_tensor[:, :len(mask), :]
        return out


class BasicDecoderWithMasking(nn.Module):
    def __init__(self, xformer, num_atten_heads):
        super().__init__()
        self.embedding_size = xformer.embedding_size
        self.max_seq_length = xformer.max_seq_length
        self.num_atten_heads = num_atten_heads
        self.qkv_size = self.embedding_size // num_atten_heads

        # Self-Attention Layer (in: (N_w x M) out: (N_w x M))
        self.self_attention_layer = SelfAttention(xformer, num_atten_heads)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm1 = nn.LayerNorm(self.embedding_size)
        # Cross-Attention Layer (in: (N_w x M) out: (N_w x M))
        self.cross_attn_layer = CrossAttention(xformer, num_atten_heads)
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm2 = nn.LayerNorm(self.embedding_size)
        # FFN 1  (in: (N_w x M) out: (N_w x M))
        self.W1 = nn.Linear( self.max_seq_length * self.embedding_size, self.max_seq_length * 2 * self.embedding_size )
        # FFN 1  (in: (N_w x M) out: (N_w x M))
        self.W2 = nn.Linear( self.max_seq_length * 2 * self.embedding_size, self.max_seq_length * self.embedding_size )
        # Layer Norm  (in: (N_w x M) out: (N_w x M))
        self.norm3 = nn.LayerNorm(self.embedding_size)

    def forward(self, sentence_tensor, final_encoder_out, mask):
        masked_sentence_tensor = sentence_tensor
        if mask is not None:
            masked_sentence_tensor = self.apply_mask(sentence_tensor, mask, self.max_seq_length, self.embedding_size)

        self_atten_out = self.self_attention_layer(masked_sentence_tensor)
        normed_atten_out = self.norm1(self_atten_out + masked_sentence_tensor)
        cross_atten_out = self.cross_attn_layer(normed_atten_out, final_encoder_out)
        normed_cross_atten_out = self.norm2(cross_atten_out)
        basic_decoder_out = nn.ReLU()(self.W1(normed_cross_atten_out.view(sentence_tensor.shape[0], -1)))
        basic_decoder_out = self.W2(basic_decoder_out)
        basic_decoder_out = basic_decoder_out.view(sentence_tensor.shape[0], self.max_seq_length, self.embedding_size)
        basic_decoder_out = self.norm3(basic_decoder_out + normed_cross_atten_out)
        return basic_decoder_out

    def apply_mask(self, sentence_tensor, mask, max_seq_length, embedding_size):
        out = torch.zeros(sentence_tensor.shape[0], max_seq_length, embedding_size).float()
        out[:,:,:len(mask)] = sentence_tensor[:,:,:len(mask)]
        return out
