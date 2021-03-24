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


def get_dataset(datadir, url="dev-clean"):
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    dataset = LIBRISPEECH(root=datadir, url=url, download=True)
    return dataset


def split_dataset(datadir, alphabet):
    sample_rate = 16000
    win_len = 20  # in milliseconds
    n_fft = int(sample_rate * win_len / 1000)  # 320
    hop_size = 10  # in milliseconds
    hop_length = int(sample_rate * hop_size / 1000)  # 160
    transform = nn.Sequential(*[
        transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length),
    ])
    dataset = get_dataset(datadir)
    dataset = ProcessedDataset(dataset, transform, alphabet)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    return train_dataset, test_dataset
