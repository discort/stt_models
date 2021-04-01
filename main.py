import argparse
import logging
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchaudio.datasets.utils import bg_iterator

from alphabet import alphabet_factory
from dataset import split_dataset
from decoders import GreedyDecoder
from metrics import compute_wer
from models import DeepSpeech

np.random.seed(200)
torch.manual_seed(200)


def model_length_function(tensor):
    return int(tensor.shape[0]) // 2 + 1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state_dict, is_best, filename):
    tempfile = filename + ".temp"

    # Remove tempfile in case interuption during the copying from tempfile to filename
    if os.path.isfile(tempfile):
        os.remove(tempfile)

    torch.save(state_dict, tempfile)
    if os.path.isfile(tempfile):
        os.rename(tempfile, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepSpeech model on TPU using librispeech dataset"
    )
    # Loader args
    parser.add_argument("--use-tpu", type=bool, default=False)
    parser.add_argument(
        "--world-size", type=int, default=8, choices=[1, 8]
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    # Preprocessing args
    parser.add_argument(
        "--window-length", type=int, default=20, help="Widow length in ms"
    )
    parser.add_argument(
        "--window-stride", type=int, default=20, help="Stride between windows in ms"
    )
    # Optimizer args
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    # Training args
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--datadir", default='/tmp/librispeech')
    parser.add_argument("--train-data-urls", type=str, nargs="+", default=['train-clean-100'])
    parser.add_argument("--val-data-urls", type=str, nargs="+", default=['dev-clean'])
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument('--logdir', type=str, default=None)
    args = parser.parse_args()
    return args


def collate_factory(model_length_function):
    """
    Based on
    https://github.com/pytorch/audio/blob/14dd917ec60fa69ce3f7c6e3f2eaf520e67928b5/examples/pipeline_wav2letter/datasets.py
    """

    def collate_fn(batch):
        inputs = [b[0].squeeze(0).transpose(0, 1) for b in batch]
        input_lengths = torch.tensor(
            [model_length_function(i) for i in inputs],
            dtype=torch.long,
            device=inputs[0].device,
        )
        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True).unsqueeze(1)

        labels = [b[1] for b in batch]
        label_lengths = torch.tensor(
            [label.shape[0] for label in labels],
            dtype=torch.long,
            device=inputs.device,
        )
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
        return inputs, input_lengths, labels, label_lengths

    return collate_fn


def get_optimizer(args, parameters):
    if args.optimizer == "sgd":
        optimizer = optim.SGD(parameters, lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(parameters, lr=args.learning_rate)
    else:
        raise ValueError(f"Invalid choice: {args.optimizer}")
    return optimizer


def train_loop_fn(loader,
                  optimizer,
                  model,
                  criterion,
                  device,
                  epoch,
                  decoder,
                  alphabet):
    running_loss = 0.0
    total_words = 0
    cumulative_wer = 0
    dataset_len = 0
    model.train()
    for inputs, input_lengths, labels, label_lengths in bg_iterator(loader, maxsize=2):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        out = model(inputs)

        loss = criterion(out, labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()

        dataset_len += inputs.size(0)
        running_loss += loss.detach() * inputs.size(0)
        wers, n_words = compute_wer(out, labels, decoder, alphabet)
        cumulative_wer += wers
        total_words += n_words

    avg_loss = running_loss / dataset_len
    avg_wer = cumulative_wer / total_words
    print('[Train][{}] Loss={:.5f} WER={:.3f} Time={}'.format(
        epoch, avg_loss, avg_wer, time.asctime()), flush=True)


def test_loop_fn(loader,
                 model,
                 criterion,
                 device,
                 epoch,
                 decoder,
                 alphabet):
    running_loss = 0.0
    total_words = 0
    cumulative_wer = 0
    dataset_len = 0

    model.eval()
    with torch.no_grad():
        for inputs, input_lengths, labels, label_lengths in bg_iterator(loader, maxsize=2):
            inputs = inputs.to(device)
            labels = labels.to(device)

            out = model(inputs)
            loss = criterion(out, labels, input_lengths, label_lengths)

            dataset_len += inputs.size(0)
            running_loss += loss.detach() * inputs.size(0)
            wers, n_words = compute_wer(out, labels, decoder, alphabet)
            cumulative_wer += wers
            total_words += n_words

        avg_loss = running_loss / dataset_len
        avg_wer = cumulative_wer / total_words
        print('[Val][{}] Loss={:.5f} WER={:.3f} Time={}'.format(
            epoch, avg_loss, avg_wer, time.asctime()), flush=True)


def _main_xla(index, args):
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

    alphabet = alphabet_factory()
    train_dataset, test_dataset = split_dataset(args, alphabet)
    collate_fn = collate_factory(model_length_function)
    if xm.xrt_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True)

    # Scale learning rate to world size
    lr = args.learning_rate * xm.xrt_world_size()

    # Get loss function, optimizer, and model
    device = xm.xla_device()
    model = DeepSpeech(in_features=161, hidden_size=2048, num_classes=len(alphabet))
    model = model.to(device)
    optimizer = get_optimizer(args, model.parameters())
    criterion = nn.CTCLoss(blank=alphabet.mapping[alphabet.char_blank])
    decoder = GreedyDecoder()

    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    test_device_loader = pl.MpDeviceLoader(test_loader, device)

    class XLAProxyOptimizer:
        """
        XLA Proxy optimizer for compatibility with
        torch.Optimizer
        """

        def __init__(self, optimizer):
            self.optimizer = optimizer

        def zero_grad(self):
            self.optimizer.zero_grad()

        def step(self):
            xm.optimizer_step(self.optimizer)

    optimizer = XLAProxyOptimizer(optimizer)

    train_eval_fn(args.num_epochs,
                  train_device_loader,
                  test_device_loader,
                  optimizer,
                  model,
                  criterion,
                  device,
                  decoder,
                  alphabet)


def main(index, args):
    alphabet = alphabet_factory()
    train_dataset, test_dataset = split_dataset(args, alphabet)
    collate_fn = collate_factory(model_length_function)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True)

    # Get loss function, optimizer, and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSpeech(in_features=161, hidden_size=2048, num_classes=len(alphabet))
    model = model.to(device)
    logging.info("Number of parameters: %s", count_parameters(model))

    optimizer = get_optimizer(args, model.parameters())
    criterion = nn.CTCLoss(blank=alphabet.mapping[alphabet.char_blank])
    decoder = GreedyDecoder()
    train_eval_fn(args.num_epochs,
                  train_loader,
                  test_loader,
                  optimizer,
                  model,
                  criterion,
                  device,
                  decoder,
                  alphabet)


def train_eval_fn(num_epochs,
                  train_loader,
                  test_loader,
                  optimizer,
                  model,
                  criterion,
                  device,
                  decoder,
                  alphabet):
    best_loss = 1.0
    # Train and eval loops
    for epoch in range(1, num_epochs + 1):
        logging.info("Epoch: %s at %s", epoch, time.asctime())
        train_loop_fn(train_loader,
                      optimizer,
                      model,
                      criterion,
                      device,
                      epoch,
                      decoder,
                      alphabet)
        loss = test_loop_fn(test_loader,
                            model,
                            criterion,
                            device,
                            epoch,
                            decoder,
                            alphabet)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        state_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_loss": best_loss,
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(state_dict, is_best)
        logging.info("End epoch: %s at %s", epoch, time.asctime())


def spawn_main(main, args):
    if args.use_tpu:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_main_xla, args=(args,), nprocs=args.world_size)
    else:
        main(0, args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    logging.info(args)
    spawn_main(main, args)
