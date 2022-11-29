import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing


class TransformerFG(nn.Module):

    def __init__(self,
                 max_seq_length,
                 embedding_size,
                 num_warmup_steps,
                 optimizer_params):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_size = embedding_size
        self.num_warmup_steps = num_warmup_steps
        self.optimizer_params = optimizer_params

        def start_training(self,
                           y,
                           X,
                           loss,
                           batch_size,
                           standardize=False):

            if standardize:
                scaler = preprocessing.MinMaxScaler().fit(X)
                X_std = torch.from_numpy(scaler.transform(X)).float()
                scaler = preprocessing.MinMaxScaler().fit(y)
                y_std = torch.from_numpy(scaler.transform(y)).float()
            else:
                X_std = torch.from_numpy(X).float()
                y_std = torch.from_numpy(X).float()

            train_iter = load_sequence(X_std, y_std, self.max_seq_length, batch_size)

