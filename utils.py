from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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


def set_axes(axes,
             xlabel,
             ylabel,
             xlim,
             ylim,
             xscale,
             yscale,
             legend):
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X,
         Y=None,
         xlabel=None,
         ylabel=None,
         legend=[],
         xlim=None,
         ylim=None,
         xscale='linear',
         yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'),
         figsize=(3.5, 2.5),
         axes=None):

    def has_one_axis(X):  # True if `X` (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None: axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize






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