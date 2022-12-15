import torch.nn as nn
import torch
import numpy as np

'''Needs to be updated'''
'''Problem is that when target source legnth is greater than source length, the index is out of bounds'''
class Performer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, sent_embed_slice, qkv_size, masked=False):
        self.qkv_size = qkv_size
        self.masked = masked
        L_ = Q.shape[1]
        M_ = V.shape[1]
        Performer = torch.zeros(sent_embed_slice.shape[0], *(L_, self.qkv_size))
        if self.masked:
            for i in range(L_):
                if i > 0:
                    M = M + torch.matmul(self.phi(K[:, i, :]).view(sent_embed_slice.shape[0], self.qkv_size, 1),
                                         V[:, i, :].view(sent_embed_slice.shape[0], 1, self.qkv_size))
                    N = N + self.phi(K[:, i, :])
                else:
                    M = torch.matmul(self.phi(K[:, i, :]).view(sent_embed_slice.shape[0], self.qkv_size, 1),
                                     V[:, i, :].view(sent_embed_slice.shape[0], 1, self.qkv_size))
                    N = self.phi(K[:, i, :])
                new_x = torch.matmul(self.phi(Q[:, i, :]), M)
                r = torch.matmul(self.phi(Q[:, i, :]), N.view(sent_embed_slice.shape[0], self.qkv_size, 1))
                sign = torch.tensor(np.sign(r.detach().numpy()))
                Performer[:, i, :] = torch.div(new_x, (r + sign * 1e-6)).view(sent_embed_slice.shape[0], self.qkv_size)
        else:

            sum = torch.zeros(sent_embed_slice.shape[0], self.qkv_size, self.qkv_size)
            norm_sum = torch.zeros(sent_embed_slice.shape[0], self.qkv_size, self.qkv_size)
            for j in range(M_):
                sum += torch.matmul(self.phi(K[:, j, :]).view(sent_embed_slice.shape[0], self.qkv_size, 1),
                                    V[:, j, :].view(sent_embed_slice.shape[0], 1, self.qkv_size))
                norm_sum += self.phi(K[:, j, :]).view(sent_embed_slice.shape[0], self.qkv_size, 1)
            for i in range(L_):
                new_x = torch.matmul(self.phi(Q[:, i, :]).view(sent_embed_slice.shape[0], 1, self.qkv_size), sum)
                r = torch.matmul(self.phi(Q[:, i, :]).view(sent_embed_slice.shape[0], 1, self.qkv_size), norm_sum)
                Performer[:, i, :] = torch.div(new_x, r).view(sent_embed_slice.shape[0], self.qkv_size)

        return Performer

    def phi(self, x):
        x = x.view(x.shape[0], x.shape[1], 1)
        d = x.shape[1]  # N_w
        M = self.qkv_size  # s_qkv
        N = torch.randn(M, d)  # (s_qkv x Nw)
        mult = torch.matmul(N, x)  # (s_qkv x 1)
        exp = torch.exp(mult)  # (s_qkv x 1)
        norm = (1 / np.sqrt(M)) * torch.exp(-0.5 * torch.norm(x)) * exp  # (s_qkv x 1)
        norm = norm.view(x.shape[0], 1, -1)  # (1 x s_qkv)
        return norm