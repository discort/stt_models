import os

import torch
import torch.nn as nn
import torchaudio.transforms as transforms
from torchaudio.datasets import LIBRISPEECH


class ProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms, alphabet):
        self.dataset = dataset
        self.transforms = transforms
        self.alphabet = alphabet

    def __getitem__(self, item):
        item = self.dataset[item]
        return self._process(item)

    def __len__(self):
        return len(self.dataset)

    def _process(self, item):
        transformed = item[0]

        transformed = self.transforms(transformed)
        target = self.alphabet.text_to_int(item[2].lower())
        target = torch.tensor(target, dtype=torch.long, device=transformed.device)

        return transformed, target


def get_dataset(datadir, data_url):
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    dataset = LIBRISPEECH(root=datadir, url=data_url, download=True)
    return dataset


def prepare_transformations(window_length=20, window_stride=10):
    sample_rate = 16000
    n_fft = int(sample_rate * window_length / 1000)  # 320
    hop_length = int(sample_rate * window_stride / 1000)  # 160
    transform = nn.Sequential(*[
        transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length),
    ])
    return transform


def split_dataset(args, alphabet):
    transform = prepare_transformations(args.window_length, args.window_stride)
    train_dataset = torch.utils.data.ConcatDataset(
        [ProcessedDataset(get_dataset(args.datadir, ds), transform, alphabet)
         for ds in args.train_data_urls]
    )
    val_dataset = torch.utils.data.ConcatDataset(
        [ProcessedDataset(get_dataset(args.datadir, ds), transform, alphabet)
         for ds in args.val_data_urls]
    )
    return train_dataset, val_dataset
