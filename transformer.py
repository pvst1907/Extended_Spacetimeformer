import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_src_trg
from encoder import MasterEncoder
from decoder import MasterDecoderWithMasking
from train import train_torch, ScheduledOptim
from sklearn import preprocessing


class Transformer(nn.Module):
    def __init__(self,
                 pred_offset,
                 input_size,
                 output_size,
                 max_seq_length,
                 embedding_size,
                 num_basic_encoders,
                 num_atten_heads,
                 num_basic_decoders,
                 positional_ecoding=True):
        super().__init__()
        self.pred_offset = pred_offset
        self.input_size = input_size
        self.output_size = output_size
        self.max_seq_length = max_seq_length
        self.embedding_size = embedding_size
        self.positional_ecoding = positional_ecoding
        self.scores = {}

        self.master_encoder = MasterEncoder(self, num_basic_encoders, num_atten_heads, self.positional_ecoding)
        self.master_decoder = MasterDecoderWithMasking(self, num_basic_decoders, num_atten_heads, self.positional_ecoding)

    def forward(self, source, target):
        source = self. master_encoder(source)
        output = self.master_decoder(target, source)
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
