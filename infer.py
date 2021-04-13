import argparse
import logging
import os

import torch
import torch.nn as nn

from alphabet import alphabet_factory
from dataset import prepare_transformations, ProcessedDataset, get_dataset
from decoders import GreedyDecoder
from main import collate_factory, model_length_function, test_loop_fn
from models import DeepSpeech


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    logging.info('Size (MB): %s', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def parse_args():
    parser = argparse.ArgumentParser(
        description="model inference"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--datadir", default='/tmp/librispeech')
    parser.add_argument("--quantize", type=bool, default=False)
    # Preprocessing args
    parser.add_argument(
        "--window-length", type=int, default=20, help="Widow length in ms"
    )
    parser.add_argument(
        "--window-stride", type=int, default=20, help="Stride between windows in ms"
    )
    parser.add_argument(
        "--n_mfcc", type=int, default=26, help="Number of mfc coefficients to retain"
    )
    parser.add_argument(
        "--n_context", type=int, default=9,
        help="Number of context frames to use on each side of the current input frame"
    )
    args = parser.parse_args()
    return args


def main(args):
    alphabet = alphabet_factory()
    device = torch.device('cpu')
    checkpoint = torch.load('model_best.pth', map_location=device)
    in_features = args.n_mfcc * (2 * args.n_context + 1)
    model = DeepSpeech(in_features=in_features, hidden_size=2048, num_classes=len(alphabet))
    model.load_state_dict(checkpoint['state_dict'])
    print_size_of_model(model)
    decoder = GreedyDecoder()
    if args.quantize:
        model = torch.quantization.quantize_dynamic(
            model, {nn.RNN, nn.Linear}, dtype=torch.qint8
        )
        logging.info('quantized model')
        print_size_of_model(model)

    transform = prepare_transformations(args)
    dataset = ProcessedDataset(get_dataset(args.datadir, "dev-clean"), transform, alphabet)
    collate_fn = collate_factory(model_length_function)
    criterion = nn.CTCLoss(blank=alphabet.mapping[alphabet.char_blank])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False)
    test_loop_fn(
        dataloader,
        model,
        criterion,
        device,
        1,
        decoder,
        alphabet)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
