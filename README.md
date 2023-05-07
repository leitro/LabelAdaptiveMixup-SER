[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
![PyTorch 1.12](https://img.shields.io/badge/pytorch-1.12-blue.svg)

# LEARNING ROBUST SELF-ATTENTION FEATURES FOR SPEECH EMOTION RECOGNITION WITH LABEL-ADAPTIVE MIXUP

Lei Kang, Lichao Zhang, Dazhi Jiang.

Accepted to ICASSP 2023.

## Hardware and Software:

- i9-10900
- 64GB RAM
- RTX3090 (24GB)

- Ubuntu 22.04
- Python 3.8
- PyTorch 1.12

## Dataset

[IEMOCAP](https://sail.usc.edu/iemocap/)

To make our results comparable to the state-of-the-art works [2, 3, 18], we merge ”excited” into ”happy” category and use speech data from four categories of ”angry”, ”happy”, ”sad” and ”neutral”, which leads to a 5531 acoustic utterances in total from 5 sessions and 10 speakers. The widely used Leave-One-Session-Out (LOSO) 5-fold cross-validation is utilized to report our final results. Thus, at each fold, 8 speakers in 4 sessions are used for training while the other 2 speakers in 1 session are used for testing.

## Train the model

- The dataset URL should be modified according to your environment in `dataset_wavMix.py`.
- Start the training process by running `python train.py`, note that the training information will be printed out once per epoch.

## Architecture of the proposed method

![arch](https://user-images.githubusercontent.com/9562709/236687709-b8d118fb-cbf3-4df0-ab38-2cda570fa0a8.png)

## Comparison with state of the arts

![res](https://user-images.githubusercontent.com/9562709/236687964-daa6d15a-3cbd-4be2-b57d-003b30c568a1.png)

## Citation

If you are using the code or benchmarks in your research, please cite our paper:

Lei Kang, Lichao Zhang, Dazhi Jiang. "Learning Robust Self-attention Features for Speech Emotion Recognition with Label-adaptive Mixup", IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2023), Rhodes Island, Greece, Jun 2023.
