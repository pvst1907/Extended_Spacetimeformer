import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_src_trg
from encoder import Encoder
from decoder import Decoder
from train import train_torch, ScheduledOptim
from sklearn import preprocessing


class SpaceTimeFormer(nn.Module):
    def __init__(self,
                 pred_offset,
                 input_size,
                 output_size,
                 seq_length,
                 embedding_size_time,
                 embedding_size_variable):
        super().__init__()
        self.pred_offset = pred_offset
        self.input_size = input_size
        self.output_size = output_size
        self.max_seq_length = seq_length*input_size
        self.embedding_size_time = embedding_size_time
        self.embedding_size_variable = embedding_size_variable
        self.embedding_size = 1 + embedding_size_time + embedding_size_variable
        self.scores = {}

        self.encoder = Encoder(self)
        self.decoder = Decoder(self)

    def forward(self, source, target):
        source = self.encoder(source)
        output = self.decoder(target, source)
        return output

    def predict(self, sequence, standardize=True):
        if standardize:
            scaler = preprocessing.MinMaxScaler().fit(sequence)
            source = torch.from_numpy(scaler.transform(sequence)).float()
        else:
            source = torch.from_numpy(sequence).float()
        target = torch.zeros_like(source)
        target[0, :] = source[-1, :]
        for i in range(1, self.pred_offset):
            pred = self.forward(torch.unsqueeze(source, dim=0), torch.unsqueeze(target, dim=0))
            target[i, :] = pred[0, i-1, :]
        return torch.squeeze(pred).detach().numpy()

    def start_training(self,
                       sequence,
                       loss,
                       metric,
                       epochs,
                       batch_size,
                       num_warmup_steps,
                       optimizer_params,
                       test_size=0.1,
                       standardize=True,
                       verbose=False,
                       plot=False):
        if standardize:
            scaler = preprocessing.MinMaxScaler().fit(sequence)
            sequence_std = torch.from_numpy(scaler.transform(sequence)).float()
        else:
            sequence_std = torch.from_numpy(sequence).float()

        # Generate Training Set
        split = int(len(sequence_std)*(1-test_size))
        train_iter = load_src_trg(sequence_std[:split], self.max_seq_length, self.pred_offset, batch_size)
        test_iter = load_src_trg(sequence_std[split:], self.max_seq_length, self.pred_offset, batch_size)

        beta1, beta2, epsilon = optimizer_params['beta1'], optimizer_params['beta2'], optimizer_params['epsilon']
        master_encoder_optimizer = ScheduledOptim(
            optim.Adam(self.master_encoder.parameters(), betas=(beta1, beta2), eps=epsilon),
            lr_mul=2,
            d_model=self.embedding_size,
            n_warmup_steps=num_warmup_steps)

        master_decoder_optimizer = ScheduledOptim(
            optim.Adam(self.master_decoder.parameters(), betas=(beta1, beta2), eps=epsilon),
            lr_mul=2,
            d_model=self.embedding_size,
            n_warmup_steps=num_warmup_steps)
        self.scores['Train'], self.scores['Evaluation'] = train_torch(self, train_iter, test_iter, loss, metric, epochs, master_encoder_optimizer, master_decoder_optimizer, verbose=verbose, plot=plot)
