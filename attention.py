import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, xformer, num_atten_heads, masked=False):
        super().__init__()
        self.max_seq_length = xformer.max_seq_length  # N_w
        self.embedding_size = xformer.embedding_size  # M
        self.num_atten_heads = num_atten_heads
        self.qkv_size = self.embedding_size // num_atten_heads
        self.attention_heads_arr = nn.ModuleList([AttentionHead(self.max_seq_length, self.qkv_size, self.qkv_size, masked) for _ in range(num_atten_heads)])

    def forward(self, sentence_tensor):  # Dimension sentence_tensor: (N_w x M)
        concat_out_from_atten_heads = torch.zeros(sentence_tensor.shape[0], self.max_seq_length, self.num_atten_heads * self.qkv_size).float()
        # Cut Input According to Number of Attention Heads
        for i in range(self.num_atten_heads):
            # Dimensions sentence_embed_slice: (N_w x s_qkv)
            sentence_embed_slice = sentence_tensor[:, :, i * self.qkv_size: (i+1) * self.qkv_size]
            # Dimensions concat_out_from_atten_heads: (N_w x s_qkv)
            concat_out_from_atten_heads[:, :, i * self.qkv_size: (i+1) * self.qkv_size] = self.attention_heads_arr[i](sentence_embed_slice)
        return concat_out_from_atten_heads


class AttentionHead(nn.Module):
    def __init__(self, max_seq_length, emb_size, qkv_size, masked=False):
        super().__init__()
        self.qkv_size = qkv_size  # s_qkv
        self.emb_size = emb_size  # M
        self.max_seq_length = max_seq_length  # N_w
        self.masked = masked

        # Dimensions W_q: (M x s_qkv)
        # Dimensions W_Q: (N_w*M) x (N_w*s_qkv)
        self.WQ = nn.Linear(max_seq_length * self.emb_size, max_seq_length * self.qkv_size)

        # Dimensions W_k: (M x s_qkv)
        # Dimensions W_K: (N_w*M) x (N_w*s_qkv)
        self.WK = nn.Linear(max_seq_length * self.emb_size, max_seq_length * self.qkv_size)

        # Dimensions W_v: (M x s_qkv)
        # Dimensions W_V: (N_w*M) x (N_w*s_qkv)
        self.WV = nn.Linear(max_seq_length * self.emb_size, max_seq_length * self.qkv_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, sent_embed_slice):  # Dimension sent_embed_slice: (N_w x M)
        # Dimensions Q: (N_w x s_qkv)
        Q = self.WQ(sent_embed_slice.reshape(sent_embed_slice.shape[0], -1).float())
        # Dimensions K: (N_w x s_qkv)
        K = self.WK(sent_embed_slice.reshape(sent_embed_slice.shape[0], -1).float())
        # Dimensions V: (N_w x s_qkv)
        V = self.WV(sent_embed_slice.reshape(sent_embed_slice.shape[0], -1).float())

        Q = Q.view(sent_embed_slice.shape[0], self.max_seq_length, self.qkv_size)
        K = K.view(sent_embed_slice.shape[0], self.max_seq_length, self.qkv_size)
        V = V.view(sent_embed_slice.shape[0], self.max_seq_length, self.qkv_size)

        # Calculating Attention \frac{softmax(Q*K^T)}{\sqrt{M}}*V
        A = K.transpose(2, 1)  # (s_qkv x N_w)
        QK_dot_prod = Q @ A  # (N_w x N_w)
        if self.masked:
            QK_dot_prod += torch.triu(torch.full(QK_dot_prod.size(), -torch.inf), diagonal=1)
        rowwise_softmax_normalizations = self.softmax(QK_dot_prod)
        Z = rowwise_softmax_normalizations @ V  # (N_w x s_qkv)
        coeff = 1.0/torch.sqrt(torch.tensor([self.qkv_size]).float())
        Z = coeff * Z  # (N_w x s_qkv)
        return Z


class CrossAttention(nn.Module):
    def __init__(self, xformer, num_atten_heads, masked=False):
        super().__init__()
        self.max_seq_length = xformer.max_seq_length  # N_w
        self.embedding_size = xformer.embedding_size  # M
        self.num_atten_heads = num_atten_heads
        self.qkv_size = self.embedding_size // num_atten_heads
        self.attention_heads_arr = nn.ModuleList([CrossAttentionHead(self.max_seq_length, self.qkv_size, self.qkv_size, masked) for _ in range(num_atten_heads)])

    def forward(self, basic_decoder_out, final_encoder_out):  # Dimension basic_decoder_out: (N_i x M), Dimension final_encoder_out: (N_w x M)
        concat_out_from_atten_heads = torch.zeros(basic_decoder_out.shape[0], self.max_seq_length, self.num_atten_heads * self.qkv_size).float()
        # Cut Input According to Number of Attention Heads
        for i in range(self.num_atten_heads):
            # Dimensions basic_decoder_slice: (N_i x s_qkv)
            basic_decoder_slice = basic_decoder_out[:, :, i * self.qkv_size: (i+1) * self.qkv_size]
            # Dimensions final_encoder_slice: (N_w x s_qkv)
            final_encoder_slice = final_encoder_out[:, :, i * self.qkv_size: (i+1) * self.qkv_size]
            # Dimensions concat_out_from_atten_heads: (N_w x s_qkv)
            concat_out_from_atten_heads[:, :, i * self.qkv_size: (i+1) * self.qkv_size] = self.attention_heads_arr[i](basic_decoder_slice, final_encoder_slice)
        return concat_out_from_atten_heads


class CrossAttentionHead(nn.Module):

    def __init__(self, max_seq_length, qkv_size, emb_size, masked=False):
        super().__init__()
        self.qkv_size = qkv_size  # s_qkv
        self.emb_size = emb_size  # M
        self.max_seq_length = max_seq_length  # N_w
        self.masked = masked

        # Dimensions W_q: (M x s_qkv)
        # Dimensions W_Q: (N_w*M) x (N_w*s_qkv)
        self.WQ = nn.Linear(max_seq_length * self.qkv_size, max_seq_length * self.qkv_size)

        # Dimensions W_k: (M x s_qkv)
        # Dimensions W_K: (N_w*M) x (N_w*s_qkv)
        self.WK = nn.Linear(max_seq_length * self.qkv_size, max_seq_length * self.qkv_size)

        # Dimensions W_v: (M x s_qkv)
        # Dimensions W_V: (N_w*M) x (N_w*s_qkv)
        self.WV = nn.Linear(max_seq_length * self.qkv_size, max_seq_length * self.qkv_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, basic_decoder_slice, final_encoder_slice):  # Dimension basic_decoder_slice (source): (N_w x M), Dimension final_encoder_slice (target): (N_w x M)
        # Dimensions Q: (N_w x s_qkv)
        Q = self.WQ(basic_decoder_slice.reshape(final_encoder_slice.shape[0], -1).float())
        # Dimensions K: (N_w x s_qkv)
        K = self.WK(final_encoder_slice.reshape(final_encoder_slice.shape[0], -1).float())
        # Dimensions V: (N_w x s_qkv)
        V = self.WV(final_encoder_slice.reshape(final_encoder_slice.shape[0], -1).float())

        Q = Q.view(final_encoder_slice.shape[0], self.max_seq_length, self.qkv_size)
        K = K.view(final_encoder_slice.shape[0], self.max_seq_length, self.qkv_size)
        V = V.view(final_encoder_slice.shape[0], self.max_seq_length, self.qkv_size)

        # Calculating Cross-Attention \frac{softmax(Q*K^T)}{\sqrt{M}}*V
        A = K.transpose(2, 1)  # (s_qkv x N_w)
        QK_dot_prod = Q @ A  # (N_w x N_w)
        if self.masked:
            QK_dot_prod += torch.triu(torch.full(QK_dot_prod.size(), -torch.inf), diagonal=1)
        rowwise_softmax_normalizations = self.softmax(QK_dot_prod)
        Z = rowwise_softmax_normalizations @ V  # (N_w x s_qkv)
        coeff = 1.0 / torch.sqrt(torch.tensor([self.qkv_size]).float())
        Z = coeff * Z  # (N_w x s_qkv)
        return Z
