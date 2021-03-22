import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as transforms
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

from alphabet import Alphabet
from models import DeepSpeech


SERIAL_EXEC = xmp.MpSerialExecutor()

alphabet = Alphabet()


def _train_print(device, step, loss, tracker, writer):
    test_utils.print_training_update(
        device,
        step,
        loss.item(),
        tracker.rate(),
        tracker.global_rate(),
        summary_writer=writer)


def collate_data_process(x):
    return data_processing(x, alphabet)


class TransformLIBRISPEECH(torchaudio.datasets.LIBRISPEECH):
    def __init__(self, *args, transform=None, **kwargs):
        super(TransformLIBRISPEECH, self).__init__(*args, **kwargs)
        self.transform = transform

    def __getitem__(self, n):
        inputs, sample_rate, utterance, _, _, _ = super().__getitem__(n)
        if self.transform:
            inputs = self.transform(inputs)
        return inputs, sample_rate, utterance


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepSpeech model on TPU using librispeech dataset"
    )
    parser.add_argument(
        "--tpu-cores", type=int, default=8, choices=[1, 8]
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--datadir", default='/tmp/librispeech')
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument('--logdir', type=str, default=None)
    args = parser.parse_args()
    return args


def data_processing(data, alphabet):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for spec, _, utterance in data:
        spec = spec.squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(alphabet.text_to_int(utterance.lower())).to(torch.long)
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return spectrograms, labels, input_lengths, label_lengths


def get_dataset():
    args = parse_args()

    if not os.path.exists(args.datadir):
        os.makedirs(args.datadir)

    sample_rate = 16000
    win_len = 20  # in milliseconds
    n_fft = int(sample_rate * win_len / 1000)  # 320
    hop_size = 10  # in milliseconds
    hop_length = int(sample_rate * hop_size / 1000)  # 160
    transform = nn.Sequential(*[
        transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length),
    ])
    dataset = TransformLIBRISPEECH(
        root=args.datadir, url="dev-clean", download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    return train_dataset, test_dataset


def train_deepspeech(*_args):
    np.random.seed(200)
    torch.manual_seed(200)

    args = parse_args()

    # Using the serial executor avoids multiple processes to
    # download the same data.
    train_dataset, test_dataset = SERIAL_EXEC.run(get_dataset)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_data_process,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_data_process,
        drop_last=True)

    # Scale learning rate to world size
    lr = args.learning_rate * xm.xrt_world_size()

    # Get loss function, optimizer, and model
    device = xm.xla_device()
    model = DeepSpeech(in_features=161, hidden_size=2048, num_classes=len(alphabet))
    model = model.to(device)
    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(flags.logdir)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    criterion = nn.CTCLoss(blank=28).to(device)

    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        for step, data in enumerate(loader):
            inputs, labels, input_lengths, label_lengths = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            out = model(inputs)
            loss = criterion(out, labels, input_lengths, label_lengths)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(args.batch_size)

            if step % args.log_steps == 0:
                xm.add_step_closure(_train_print, args=(device, step, loss, tracker, writer))

    def test_loop_fn(loader):
        model.eval()
        with torch.no_grad():
            for step, data in enumerate(loader):
                inputs, labels, input_lengths, label_lengths = data
                inputs, labels = inputs.to(device), labels.to(device)
                out = model(inputs)
                loss = criterion(out, labels, input_lengths, label_lengths)
                if step % args.log_steps == 0:
                    xm.master_print('[xla:{}]({}) Val Loss={:.5f}'.format(
                                    xm.get_ordinal(), step, loss.item()), flush=True)

    # Train and eval loops
    for epoch in range(1, args.num_epochs + 1):
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device))
        xm.master_print("Finished training epoch {}".format(epoch))

        para_loader = pl.ParallelLoader(test_loader, [device])
        test_loop_fn(para_loader.per_device_loader(device))


def main():
    flags = parse_args()
    xmp.spawn(train_deepspeech, args=(), nprocs=flags.tpu_cores)


if __name__ == '__main__':
    main()
