import os

import torch
import torch.nn as nn
import torchaudio.transforms as transforms
from torchaudio.datasets import LIBRISPEECH

import numpy as np


class Normalize(nn.Module):
    """Normalize a tensor to have zero mean and one standard deviation."""

    def __call__(self, tensor):
        return (tensor - tensor.mean()) / tensor.std()


# https://github.com/MyrtleSoftware/deepspeech/blob/master/src/deepspeech/data/preprocess.py#L73
class AddContextFrames(nn.Module):
    """Add context frames to each step in the original signal.
    Args:
        n_context: Number of context frames to add to frame in the original
            signal.
    Example:
        >>> signal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> n_context = 2
        >>> print(add_context_frames(signal, n_context))
        [[0 0 0 0 0 0 1 2 3 4 5 6 7 8 9]
         [0 0 0 1 2 3 4 5 6 7 8 9 0 0 0]
         [1 2 3 4 5 6 7 8 9 0 0 0 0 0 0]]
    """

    def __init__(self, n_context):
        super(AddContextFrames, self).__init__()
        self.n_context = n_context

    def __call__(self, signal):
        """
        Args:
            signal: numpy ndarray with shape (steps, features).
        Returns:
            numpy ndarray with shape:
                (steps, features * (n_context + 1 + n_context))
        """
        # Pad to ensure first and last n_context frames in original signal have
        # at least n_context frames to their left and right respectively.
        device = signal.device
        signal = signal.cpu().squeeze(0).T
        signal = signal.data.numpy()
        steps, features = signal.shape
        padding = np.zeros((self.n_context, features), dtype=signal.dtype)
        signal = np.concatenate((padding, signal, padding))

        window_size = self.n_context + 1 + self.n_context
        strided_signal = np.lib.stride_tricks.as_strided(
            signal,
            # Shape of the new array.
            (steps, window_size, features),
            # Strides of the new array (bytes to step in each dim).
            (signal.strides[0], signal.strides[0], signal.strides[1]),
            # Disable write to prevent accidental errors as elems share memory.
            writeable=False)

        # Flatten last dim and return a copy to permit writes.
        strided_signal = strided_signal.reshape(steps, -1).copy()
        strided_signal = torch.tensor(strided_signal, device=device).T.unsqueeze(0)
        return strided_signal


class ProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms, alphabet):
        self.dataset = dataset
        self.alphabet = alphabet
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transforms = transforms.to(self.device)

    def __getitem__(self, item):
        item = self.dataset[item]
        return self._process(item)

    def __len__(self):
        return len(self.dataset)

    def _process(self, item):
        transformed = item[0].to(self.device)

        transformed = self.transforms(transformed)
        target = self.alphabet.text_to_int(item[2].lower())
        target = torch.tensor(target, dtype=torch.long)

        return transformed, target


def get_dataset(datadir, data_url):
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    dataset = LIBRISPEECH(root=datadir, url=data_url, download=True)
    return dataset


def prepare_transformations(args):
    sample_rate = 16000
    n_fft = int(sample_rate * args.window_length / 1000)  # 320
    hop_length = int(sample_rate * args.window_stride / 1000)  # 160
    melkwargs = dict(n_fft=n_fft, hop_length=hop_length)
    transform = nn.Sequential(*[
        #transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length),
        transforms.MFCC(n_mfcc=args.n_mfcc, melkwargs=melkwargs),
        AddContextFrames(args.n_context),
        Normalize(),
    ])
    return transform


def split_dataset(args, alphabet):
    transform = prepare_transformations(args)
    train_dataset = torch.utils.data.ConcatDataset(
        [ProcessedDataset(get_dataset(args.datadir, ds), transform, alphabet)
         for ds in args.train_data_urls]
    )
    val_dataset = torch.utils.data.ConcatDataset(
        [ProcessedDataset(get_dataset(args.datadir, ds), transform, alphabet)
         for ds in args.val_data_urls]
    )
    return train_dataset, val_dataset
