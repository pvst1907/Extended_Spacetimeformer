from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def load_src_trg(sequence, sequence_length, offset, batch_size=1):
    dataset = SequenceDataset(sequence, sequence_length, offset)
    src_trg = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return src_trg


class SequenceDataset(Dataset):
    def __init__(self, sequence, sequence_length=5, offset=0):
        self.sequence_length = sequence_length
        self.offset = offset
        self.sequence = sequence

    def __len__(self):
        return self.sequence.shape[0] - self.sequence_length + 1

    def __getitem__(self, i):
        i = i if i + self.sequence_length + self.offset < (self.sequence.shape[0] - self.sequence_length - self.offset) else 0
        src = self.sequence[i:i+self.sequence_length]
        trg = self.sequence[i + self.sequence_length - 1:i + self.sequence_length + self.offset - 1]
        trg_y = self.sequence[i + self.sequence_length:i + self.sequence_length + self.offset]
        return src, trg, trg_y




"""def load_sequence(X, y, sequence_length, offset, batch_size=1):
    dataset = SequenceDataset(X, y, sequence_length, offset)
    sequence = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return sequence


class SequenceDataset(Dataset):
    def __init__(self, X, y, sequence_length=5, offset=0):
        self.sequence_length = sequence_length
        self.offset = offset
        self.y = y
        self.X = X

    def __len__(self):
        return self.X.shape[0] - self.sequence_length + 1

    def __getitem__(self, i):
        i = i if i + self.offset < (self.X.shape[0] - self.sequence_length) else 0
        x = self.X[i:i + self.sequence_length]
        y = self.y[i + self.offset:i + self.offset + self.sequence_length]
        return x, y"""