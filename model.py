import torch
import torchaudio
import numpy as np

gru_drop_out_rate = 0.2
cuda = torch.device('cuda')
FEAT_SIZE = 64
GRU_SIZE = 1024
N_HEAD = 8
N_GRU = 2
LAMBD = 4


class SpeechModel(torch.nn.Module):
    def __init__(self, max_len, select_ft, n_emotions):
        super(SpeechModel, self).__init__()
        self.base = SpeechModelBase(max_len, select_ft)
        self.head = SpeechModelHead(n_emotions)

    def forward(self, src, src_trim_idx):
        feat = self.base(src, src_trim_idx) # seq,b,feat
        res = self.head(feat)
        return res


class SpeechModelHead(torch.nn.Module):
    def __init__(self, num_emotions):
        super(SpeechModelHead, self).__init__()
        self.num_emotions = num_emotions
        self.gru = torch.nn.GRU(GRU_SIZE, GRU_SIZE, N_GRU, dropout=gru_drop_out_rate)
        self.fc3 = torch.nn.Linear(GRU_SIZE, FEAT_SIZE)
        self.fc4 = torch.nn.Linear(FEAT_SIZE, self.num_emotions)

    def forward(self, feat):
        state = feat[0] # batch,feat
        fea = self.fc3(state)
        res = self.fc4(fea)
        return fea, res


# wav2vec2
class SpeechModelBase(torch.nn.Module):
    def __init__(self, wav_max_len, select_ft):
        super(SpeechModelBase, self).__init__()
        self.wav_max_len = wav_max_len
        self.select_ft = select_ft
        buddle = torchaudio.pipelines.HUBERT_LARGE
        self.wav2vec2 = buddle.get_model()
        self.flag = True

    def forward(self, src, src_mask_idx=None):
        features, _ = self.wav2vec2.extract_features(src)
        feat_sel = features[self.select_ft] # batch,frames,feat
        out = feat_sel.permute(1, 0, 2) # frames,batch,feat
        if self.flag:
            self.flag = False
        return out


if __name__ == '__main__':
    pass
