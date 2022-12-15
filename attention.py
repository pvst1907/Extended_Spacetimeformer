import torch
import torch.nn as nn


class SafeSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X_denom = torch.sum(torch.exp(X), 2) + 1e-5
        return torch.divide(torch.exp(X), torch.unsqueeze(X_denom, dim=2))


class LocalSelfAttention(nn.Module):
    def __init__(self, input_size, seq_length, embedding_size, qkv_size, masked=False):
        super().__init__()
        self.seq_length = seq_length  # N_w
        self.embedding_size = embedding_size  # M
        self.qkv_size = qkv_size  # s_qkv
        self.input_size = input_size
        self.seq_length_head = self.seq_length//input_size
        self.attention_heads = nn.ModuleList([
            AttentionHead(self.seq_length_head, self.embedding_size, qkv_size, masked) for _ in range(input_size)
        ])

    def forward(self, sequence):  # Dimension sentence_tensor: (N_w x M)
        concat_out_from_atten_heads = torch.zeros(sequence.shape[0], self.seq_length, self.qkv_size).float()
        # Cut Input According to Number of Input Time Series
        for i in range(self.input_size):
            # Dimensions sentence_embed_slice: (N_w x s_qkv)
            sequence_slice = sequence[:, i * self.seq_length_head: (i+1) * self.seq_length_head, :]
            # Dimensions concat_out_from_atten_heads: (N_w x s_qkv)
            concat_out_from_atten_heads[:, i * self.seq_length_head: (i+1) * self.seq_length_head, :] = self.attention_heads[i](sequence_slice)
        return concat_out_from_atten_heads


class GlobalSelfAttention(nn.Module):
    def __init__(self, seq_length, embedding_size, qkv_size, masked=False):
        super().__init__()
        self.seq_length = seq_length  # N_w
        self.embedding_size = embedding_size  # M
        self.qkv_size = qkv_size
        self.attention_head = AttentionHead(self.seq_length, self.embedding_size, self.qkv_size, masked)

    def forward(self, sequence):  # Dimension sentence_tensor: (N_w x M)
        return self.attention_head(sequence)


class AttentionHead(nn.Module):
    def __init__(self, seq_length, emb_size, qkv_size, masked=False):
        super().__init__()
        self.qkv_size = qkv_size  # s_qkv
        self.emb_size = emb_size  # M
        self.seq_length = seq_length  # N_w
        self.masked = masked

        # Dimensions W_q: (M x s_qkv)
        # Dimensions W_Q: (N_w*M) x (N_w*s_qkv)
        self.WQ = nn.Linear(seq_length * self.emb_size, seq_length * self.qkv_size)

        # Dimensions W_k: (M x s_qkv)
        # Dimensions W_K: (N_w*M) x (N_w*s_qkv)
        self.WK = nn.Linear(seq_length * self.emb_size, seq_length * self.qkv_size)

        # Dimensions W_v: (M x s_qkv)
        # Dimensions W_V: (N_w*M) x (N_w*s_qkv)
        self.WV = nn.Linear(seq_length * self.emb_size, seq_length * self.qkv_size)

        self.softmax = SafeSoftmax()

    def forward(self, sent_embed_slice):  # Dimension sent_embed_slice: (N_w x M)
        # Dimensions Q: (N_w x s_qkv)
        Q = self.WQ(sent_embed_slice.reshape(sent_embed_slice.shape[0], -1).float())
        # Dimensions K: (N_w x s_qkv)
        K = self.WK(sent_embed_slice.reshape(sent_embed_slice.shape[0], -1).float())
        # Dimensions V: (N_w x s_qkv)
        V = self.WV(sent_embed_slice.reshape(sent_embed_slice.shape[0], -1).float())

        Q = Q.view(sent_embed_slice.shape[0], self.seq_length, self.qkv_size)
        K = K.view(sent_embed_slice.shape[0], self.seq_length, self.qkv_size)
        V = V.view(sent_embed_slice.shape[0], self.seq_length, self.qkv_size)

        # Calculating Attention \frac{softmax(Q*K^T)}{\sqrt{M}}*V
        A = K.transpose(2, 1)  # (s_qkv x N_w)
        QK_dot_prod = Q @ A  # (N_w x N_w)
        if self.masked:
            QK_dot_prod += torch.triu(torch.full(QK_dot_prod.size(), -1e20), diagonal=1)
        rowwise_softmax_normalizations = self.softmax(QK_dot_prod)
        Z = rowwise_softmax_normalizations @ V  # (N_w x s_qkv)
        coeff = 1.0/torch.sqrt(torch.tensor([self.qkv_size]).float())
        Z = coeff * Z  # (N_w x s_qkv)
        return Z


class LocalCrossAttention(nn.Module):
    def __init__(self, output_size, input_size, src_seq_length, trg_seq_length, embedding_size, qkv_size, masked=False):
        super().__init__()
        self.src_seq_length = src_seq_length  # N_w
        self.trg_seq_length = trg_seq_length  # N_w
        self.embedding_size = embedding_size  # M
        self.qkv_size = qkv_size  # s_qkv
        self.output_size = output_size
        self.input_size = input_size
        self.trg_seq_length_head = trg_seq_length // output_size
        self.src_seq_length_head = self.src_seq_length // input_size
        self.attention_heads = nn.ModuleList([
            CrossAttentionHead(self.src_seq_length_head, self.trg_seq_length_head, self.embedding_size, self.qkv_size, masked) for _ in range(input_size * output_size)
        ])

    def forward(self, basic_decoder_out, final_encoder_out):  # Dimension basic_decoder_out: (N_i x M), Dimension final_encoder_out: (N_w x M)
        concat_out_from_atten_heads = torch.zeros(basic_decoder_out.shape[0], self.trg_seq_length, self.qkv_size).float()
        # Cut Input According to Number of Input Time Series
        for i in range(self.output_size):
            # Dimensions basic_decoder_slice: (N_i x s_qkv)
            basic_decoder_slice = basic_decoder_out[:, i * self.trg_seq_length_head: (i + 1) * self.trg_seq_length_head, :]
            for j in range(self.input_size):
                # Dimensions final_encoder_slice: (Nw/input_size x s_qkv)
                final_encoder_out_slice = final_encoder_out[:, j * self.src_seq_length_head: (j + 1) * self.src_seq_length_head, :]
                if j == 0:
                    # Dimensions concat_out_from_atten_heads: (N_w x s_qkv)
                    concat_out_from_atten_heads[:, i * self.trg_seq_length_head: (i + 1) * self.trg_seq_length_head, :] = self.attention_heads[i](basic_decoder_slice, final_encoder_out_slice)
                else:
                    concat_out_from_atten_heads[:, i * self.trg_seq_length_head: (i + 1) * self.trg_seq_length_head, :] += self.attention_heads[i](basic_decoder_slice, final_encoder_out_slice)

        return concat_out_from_atten_heads


class GlobalCrossAttention(nn.Module):
    def __init__(self, src_seq_length, trg_seq_length, embedding_size, qkv_size, masked=False):
        super().__init__()
        self.src_seq_length = src_seq_length  # N_w
        self.trg_seq_length = trg_seq_length
        self.embedding_size = embedding_size  # M
        self.qkv_size = qkv_size
        self.attention_head = CrossAttentionHead(self.src_seq_length, self.trg_seq_length, self.embedding_size, self.qkv_size, masked)

    def forward(self, basic_decoder_out, final_encoder_out):  # Dimension basic_decoder_out: (N_i x M), Dimension final_encoder_out: (N_w x M)
        return self.attention_head(basic_decoder_out, final_encoder_out)


class CrossAttentionHead(nn.Module):

    def __init__(self, src_seq_length, trg_seq_length, qkv_size, emb_size, masked=False):
        super().__init__()
        self.qkv_size = qkv_size  # s_qkv
        self.emb_size = emb_size  # M
        self.trg_seq_length = trg_seq_length
        self.src_seq_length = src_seq_length  # N_w
        self.masked = masked

        # Dimensions W_q: (M x s_qkv)
        # Dimensions W_Q: (N_w*M) x (N_w*s_qkv)
        self.WQ = nn.Linear(trg_seq_length * self.emb_size, trg_seq_length * self.qkv_size)

        # Dimensions W_k: (M x s_qkv)
        # Dimensions W_K: (N_w*M) x (N_w*s_qkv)
        self.WK = nn.Linear(src_seq_length * self.emb_size, src_seq_length * self.qkv_size)

        # Dimensions W_v: (M x s_qkv)
        # Dimensions W_V: (N_w*M) x (N_w*s_qkv)
        self.WV = nn.Linear(src_seq_length * self.emb_size, src_seq_length * self.qkv_size)

        self.softmax = SafeSoftmax()

    def forward(self, basic_decoder_slice, final_encoder_slice):  #Dimension basic_decoder_slice (source): (N_w x M), Dimension final_encoder_slice (target): (N_w x M)
        # Dimensions Q: (N_w x s_qkv)

        Q = self.WQ(basic_decoder_slice.reshape(final_encoder_slice.shape[0], -1).float())
        # Dimensions K: (N_w x s_qkv)
        K = self.WK(final_encoder_slice.reshape(final_encoder_slice.shape[0], -1).float())
        # Dimensions V: (N_w x s_qkv)
        V = self.WV(final_encoder_slice.reshape(final_encoder_slice.shape[0], -1).float())

        Q = Q.view(final_encoder_slice.shape[0], self.trg_seq_length, self.qkv_size)
        K = K.view(final_encoder_slice.shape[0], self.src_seq_length, self.qkv_size)
        V = V.view(final_encoder_slice.shape[0], self.src_seq_length, self.qkv_size)

        # Calculating Cross-Attention \frac{softmax(Q*K^T)}{\sqrt{M}}*V
        A = K.transpose(2, 1)  # (s_qkv x N_w)
        QK_dot_prod = Q @ A  # (N_w x N_w)
        if self.masked:
            QK_dot_prod += torch.triu(torch.full(QK_dot_prod.size(), -1e20), diagonal=1)
        rowwise_softmax_normalizations = self.softmax(QK_dot_prod)
        Z = rowwise_softmax_normalizations @ V  # (N_w x s_qkv)
        coeff = 1.0 / torch.sqrt(torch.tensor([self.qkv_size]).float())
        Z = coeff * Z  # (N_w x s_qkv)
        return Z
