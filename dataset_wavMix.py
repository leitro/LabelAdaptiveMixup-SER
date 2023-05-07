import random
import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
import librosa.display
import matplotlib.pyplot as plt
from augmentor import augmentor

cuda = torch.device('cuda')
NORM = True
SAMPLE_RATE = 16000 # original IEMOCAP SR
MAX_LEN = 100000 # according to histogram
speakers = {0: 'S1_Female', 1: 'S1_Male', 2: 'S2_Female', 3: 'S2_Male', 4: 'S3_Female', 5: 'S3_Male', 6: 'S4_Female', 7: 'S4_Male', 8: 'S5_Female', 9: 'S5_Male'}
emotions = {0:'ang', 1:'hap', 2:'sad', 3:'neu'}
emo2id = {c: i for i, c in emotions.items()}
base_path = '/home/tro/datasets/IEMOCAP_Emotional_Speech/'
wav_path = base_path + 'iemocap_audios/'

class speech_data(Dataset):
    def __init__(self, file_label, p_aug, p_mix):
        self.file_label = file_label
        self.p_aug = p_aug
        self.p_mix = p_mix
        self.len = self.__len__()

    def __getitem__(self, index):
        emo = [0] * len(emotions)
        file_name, label = self.file_label[index]
        wav_file = wav_path + file_name
        wav_signal, sr = librosa.load(wav_file, sr=SAMPLE_RATE)
        length, wav_signal_padded = self.pad(wav_signal)
        session_id, speaker_id, sex, emotion = label.split('-')
        emo_id = emo2id[emotion]
        if NORM:
            wav_signal_padded /= np.max(np.abs(wav_signal_padded))
        if random.random() < self.p_aug:
            wav_signal_padded = augmentor(wav_signal_padded, sr)
        if random.random() < self.p_mix:
            idx2 = random.randint(0, self.len-1)
            file_name2, label2 = self.file_label[idx2]
            wav_file2 = wav_path + file_name2
            wav_signal2, sr2 = librosa.load(wav_file2, sr=SAMPLE_RATE)
            length2, wav_signal_padded2 = self.pad(wav_signal2)
            if NORM:
                wav_signal_padded2 /= np.max(np.abs(wav_signal_padded2))
            if random.random() < self.p_aug:
                wav_signal_padded2 = augmentor(wav_signal_padded2, sr2)
            session_id2, speaker_id2, sex2, emotion2 = label2.split('-')
            emo_id2 = emo2id[emotion2]
            ## label balance
            emo[emo_id] = length / (length + length2)
            emo[emo_id2] = length2 / (length + length2)
            wav_sig_fin = 0.5 * wav_signal_padded + 0.5 * wav_signal_padded2
            name_fin = '+'.join([file_name, file_name2])
        else:
            emo[emo_id] = 1
            wav_sig_fin = wav_signal_padded
            name_fin = file_name

        trim_idx = [0, MAX_LEN]
        return name_fin, wav_sig_fin, trim_idx, emo

    def __len__(self):
        return len(self.file_label)

    def pad(self, data):
        if len(data) >= MAX_LEN:
            return MAX_LEN, data[:MAX_LEN]
        else:
            return len(data), np.pad(data, (0, MAX_LEN - len(data)), mode='mean')


def loadData(Paug, Pmix):
    with open('fold.num', 'r') as rrr:
        fold_n = int(rrr.read().strip())
    partition_path = base_path + 'partition_5folds/'
    train_set = partition_path + f'fold_{fold_n}.train'
    valid_set = partition_path + f'fold_{fold_n}.valid'
    test_set = partition_path + f'fold_{fold_n}.test'
    with open(train_set, 'r') as f_tr:
        data_tr = f_tr.readlines()
        file_label_tr = [i.strip().split(' ') for i in data_tr]
    with open(valid_set, 'r') as f_va:
        data_va = f_va.readlines()
        file_label_va = [i.strip().split(' ') for i in data_va]
    with open(test_set, 'r') as f_te:
        data_te = f_te.readlines()
        file_label_te = [i.strip().split(' ') for i in data_te]

    random.shuffle(file_label_tr)
    data_train = speech_data(file_label_tr, p_aug=Paug, p_mix=Pmix)
    data_valid = speech_data(file_label_va, p_aug=0, p_mix=0)
    data_test = speech_data(file_label_te, p_aug=0, p_mix=0)
    return data_train, data_valid, data_test

