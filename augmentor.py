import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
from audiomentations import *

combi1 = [0.3, 0.9]

def augmentor(wav, sr):
    aug = Compose([
        Gain(min_gain_in_db=-10, max_gain_in_db=10, p=combi1[0]),
        PolarityInversion(p=combi1[1]),
    ])
    return aug(wav, sr)

if __name__ == '__main__':
    pass
