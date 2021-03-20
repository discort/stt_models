import torch
import torch.nn as nn


class DeepSpeech2(nn.Module):
    """

    WIP

    DeepSpeech2 model architecture from
    `"Deep Speech 2: End-to-End Speech Recognition in English and Mandarin"`
    <https://arxiv.org/abs/1512.02595> paper.

    Args:
        in_features: Number of input features
        hidden_size: Internal hidden unit size.
        rnn_layers: Number of RNN layers
        num_classes: Number of output classes
    """

    def __init__(self,
                 in_features: int,
                 hidden_size: int,
                 rnn_layers: int,
                 num_classes: int):
        super(DeepSpeech2, self).__init__()
        self.hidden_size = hidden_size
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=(41, 11),
                      stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(21, 11),
                      stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20),
        ])
        self.rnn = nn.Sequential(*[
            nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=rnn_layers, bidirectional=True)
        ])
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
