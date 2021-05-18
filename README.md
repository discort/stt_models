## Speech-to-text models 

Pytorch implementation and comparison `speech-to-text` (STT) models.

References:
- [Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567)
- [Wav2Letter](https://arxiv.org/abs/1609.03193) (WIP)
- [Jasper: An End-to-End Convolutional Neural Acoustic Model](https://arxiv.org/abs/1904.03288) (WIP)

## Train on CPU/GPU
    python main.py --num-workers 0 --batch-size 32 --train-data-urls train-clean-100 train-clean-360 --num-epochs 15 --window-stride 20 --optimizer adam --learning-rate 3e-4 --log-steps 100 --checkpoint test

- trained on `train-clean-100` `train-clean-360`.
- WER on `dev-clean` (9 epochs): 0.33 
- pre-trained weights: https://github.com/discort/stt_models/releases/tag/0.1

## ToDo:
- Parallelize model training
- write checkpointing and tensorboard logger
- training using mixed precision