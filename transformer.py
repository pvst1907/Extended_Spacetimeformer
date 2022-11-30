import torch
import torch.nn as nn
from utils import load_src_trg
from encoder import MasterEncoder
from decoder import MasterDecoderWithMasking
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
                 num_basic_decoders):
        super().__init__()
        self.pred_offset = pred_offset
        self.input_size = input_size
        self.output_size = output_size
        self.max_seq_length = max_seq_length
        self.embedding_size = embedding_size

        self.master_encoder = MasterEncoder(self, num_basic_encoders, num_atten_heads)
        self.master_decoder = MasterDecoderWithMasking(self, num_basic_decoders, num_atten_heads)

    def forward(self, source, target):
        # Missing: Positional Encoding
        source = self. master_encoder(source)
        output = self.master_decoder(target, source)
        return output

    def start_training(self, sequence, batch_size, num_warmup_steps, optimizer_params, standardize=True):
        if standardize:
            scaler = preprocessing.MinMaxScaler().fit(sequence)
            sequence_std = torch.from_numpy(scaler.transform(sequence)).float()
        else:
            sequence_std = torch.from_numpy(sequence).float()

        # Generate Training Set
        train_iter = load_src_trg(sequence_std, self.seq_len, self.pred_offset, batch_size)

        beta1, beta2, epsilon = optimizer_params['beta1'], optimizer_params['beta2'], optimizer_params['epsilon']
        master_encoder_optimizer = self.ScheduledOptim(
            optim.Adam(self.master_encoder.parameters(), betas=(beta1, beta2), eps=epsilon),
            lr_mul=2,
            d_model=self.embedding_size,
            n_warmup_steps=num_warmup_steps)

        master_decoder_optimizer = self.ScheduledOptim(
            optim.Adam(self.master_decoder.parameters(), betas=(beta1, beta2), eps=epsilon),
            lr_mul=2,
            d_model=self.embedding_size,
            n_warmup_steps=num_warmup_steps)



