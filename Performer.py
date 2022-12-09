import torch.nn as nn
import torch
import numpy as np

class Performer(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, K, Q, V , sent_embed_slice, max_seq_length, qkv_size):
        self.qkv_size = qkv_size
        self.max_seq_length = max_seq_length
        Performer = torch.zeros(sent_embed_slice.shape[0], *(self.max_seq_length, self.qkv_size))
        for batch in range(sent_embed_slice.shape[0]):
            for i in range(self.max_seq_length):
                if i > 0:
                    M = M + torch.matmul(self.phi(K[batch, i, :]).view(self.qkv_size, 1),
                                         V[batch, i, :].view(1, self.qkv_size))
                    N = N + self.phi(K[batch, i, :])
                else:
                    M = torch.matmul(self.phi(K[batch, i, :]).view(self.qkv_size, 1), V[batch, i, :].view(1, self.qkv_size))
                    N = self.phi(K[batch, i, :])

                new_x = torch.matmul(self.phi(Q[batch, i, :]), M.clone())
                r = torch.matmul(self.phi(Q[batch, i, :]), N.view(self.qkv_size, 1))
                sign = torch.tensor(np.sign(r.detach().numpy()))
                Performer[batch, i, :] = new_x / (r + sign * 1e-6)
        ### Vectorized version ###
        # Performer = torch.zeros(sent_embed_slice.shape[0], *(self.max_seq_length, self.qkv_size))
        # for i in range(self.max_seq_length):
        #     if i > 0:
        #         M = M + torch.matmul(self.phi(K[:, i, :]).view(sent_embed_slice.shape[0],self.qkv_size, 1),
        #                              V[:, i, :].view(sent_embed_slice.shape[0],1, self.qkv_size))
        #         N = N + self.phi(K[:, i, :])
        #     else:
        #         M = torch.matmul(self.phi(K[:, i, :]).view(sent_embed_slice.shape[0],self.qkv_size, 1), V[:, i, :].view(sent_embed_slice.shape[0],1, self.qkv_size))
        #         N = self.phi(K[:, i, :])
        #
        #     new_x = torch.matmul(self.phi(Q[:, i, :]), M.clone())
        #     r = torch.matmul(self.phi(Q[:, i, :]), N.view(sent_embed_slice.shape[0],self.qkv_size, 1))
        #     sign = torch.tensor([np.sign(r.detach().numpy())])
        #     Performer[:, i, :] = torch.div(new_x, r + sign * 1e-6).view(sent_embed_slice.shape[0], self.qkv_size)
        return Performer



    def phi(self, x):
        #x = x.view(x.shape[0], x.shape[1], 1)
        d = self.qkv_size  # N_w
        M = self.qkv_size  # s_qkv
        N = torch.randn(M, d)  # (s_qkv x Nw)
        mult = torch.matmul(N, x)  # (s_qkv x 1)
        exp = torch.exp(mult)  # (s_qkv x 1)
        norm = (1 / np.sqrt(M)) * torch.exp(-0.5 * torch.norm(x)) * exp  # (s_qkv x 1)
        #norm = norm.view(x.shape[0], 1, -1)  # (1 x s_qkv)
        norm = norm.view(1, -1)  # (1 x s_qkv)
        return norm