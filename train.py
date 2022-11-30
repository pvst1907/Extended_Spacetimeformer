from IPython import display
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import set_axes


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    def __init__(self,
                 xlabel=None,
                 ylabel=None,
                 legend=None,
                 xlim=None,
                 ylim=None,
                 xscale='linear',
                 yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'),
                 nrows=1,
                 ncols=1,
                 figsize=(5, 3)):
        if legend is None:
            legend = []

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)

        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
            self.axes[0].cla()
            for x, y, fmt in zip(self.X, self.Y, self.fmts):
                self.axes[0].plot(x, y, fmt)
            self.config_axes()
            display.display(self.fig)
            display.clear_output(wait=True)


def train_epoch(xformer,
                train_iter,
                loss,
                master_encoder_optimizer,
                master_decoder_optimizer):
    xformer.train()
    tracker = Accumulator(2)
    for src, trg, trg_y in train_iter:
        trg_y_hat = xformer(src, trg)
        l = loss(trg_y, trg_y_hat)
        master_encoder_optimizer.zero_grad()
        master_decoder_optimizer.zero_grad()
        l.mean().backward()
        master_encoder_optimizer.step_and_update_lr()
        master_decoder_optimizer.step_and_update_lr()
    tracker.add(float(l.sum()), trg_y.numel())
    return tracker[0] / tracker[1]


def train_torch(xformer,
                train_iter,
                loss,
                metric,
                epochs,
                master_encoder_optimizer,
                master_decoder_optimizer,
                patience=100,
                verbose=False,
                plot=False):
    min_loss = np.nan
    epochs_no_improve = 0
    if plot:
        animator = Animator(xlabel='epoch', xlim=[1, epochs], legend=['training loss', 'evaluation_loss'])
        for epoch in range(epochs):
            train_tracker = train_epoch(xformer, train_iter, loss, master_encoder_optimizer, master_decoder_optimizer)
            eval_tracker = evaluate_model(xformer, train_iter, metric)
            epochs_no_improve += 1
            if np.isnan(min_loss) or train_tracker < min_loss:
                min_loss = train_tracker
                state_dict = copy.deepcopy(xformer.state_dict())
                epochs_no_improve = 0
            if epoch > 5 and epochs_no_improve > patience:
                xformer.load_state_dict(state_dict)
                print('Stopped at ', epoch, train_tracker)
                break
            if verbose:
                print(train_tracker)
            if plot:
                animator.add(epoch + 1, (train_tracker, eval_tracker))
        return train_tracker, eval_tracker


def evaluate_model(xformer, data_iter, metric):
    xformer.eval()
    tracker = Accumulator(2)
    with torch.no_grad():
        for src, trg, trg_y in data_iter:
            tracker.add(float(metric(xformer(src, trg), trg_y).sum()), trg_y.numel())
    return tracker[0] / tracker[1]


class ScheduledOptim:
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
