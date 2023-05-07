import torch
import itertools
import numpy as np
import os
from torch import optim
from dataset_wavMix import loadData, MAX_LEN, emotions, speakers
from model import SpeechModel, FEAT_SIZE
import glob
import time
from sklearn.metrics import confusion_matrix
from center_loss import CenterLoss
import time
import datetime

cuda = torch.device('cuda')


BATCH_SIZE = 16
WAV2VEC2_DROPOUT = 0.4
SELECT_FT = 21
NUM_THREAD = 4
LEARNING_RATE = 1 * 1e-4
LR_CENTER = 1 * 1e-3
lr_milestone = list(range(1, 21))
lr_gamma = 0.8
EARLY_STOP = 10


class ACC:
    def __init__(self):
        self.num = len(emotions)
        self.correct_dict = {i: 0 for i in range(self.num)}
        self.total_dict = {i: 0 for i in range(self.num)}

    def add(self, pred_ids, gt_emotion):
        pred_ids = pred_ids.detach().cpu().numpy().tolist()
        gt_emotion = gt_emotion.detach().cpu().numpy().tolist()
        for k, v in enumerate(gt_emotion):
            if pred_ids[k] == v:
                self.correct_dict[v] += 1
            self.total_dict[v] += 1

    def fin(self):
        ua = []
        correct = np.sum([self.correct_dict[i] for i in range(self.num)])
        total = np.sum([self.total_dict[i] for i in range(self.num)])
        wa = correct/total
        for i in range(self.num):
            ua.append(self.correct_dict[i]/self.total_dict[i])
        ua = np.mean(ua)
        return wa, ua


def print_confusion_matrix(matrix):
    out = ''
    for i in range(len(emotions)):
        out += f'{emotions[i]:<10}{np.array2string(matrix[i], precision=2, floatmode="fixed")}\n'
    print(out)


def run_epoch(dataloader, model, criterion, epoch, alpha, opt=None):
    mode = ''
    if opt:
        mode = 'train'
        model_opt, cent_opt = opt
    else:
        if epoch is list:
            mode = 'test'
            epoch = epoch[0]
        else:
            mode = 'valid'
    total_loss = [0] * 2
    conf_pred = []
    conf_gt = []

    acc = ACC()

    for i, (file_names, wav_signals, sig_trim_idxes, gt_emotion) in enumerate(dataloader):
        batch_size = len(file_names)
        wav_signals = wav_signals.to(cuda)
        gt_emotion = gt_emotion.to(cuda)

        log_sm, kl_crit, center_crit = criterion
        feat_emo, res_emo = model(wav_signals, sig_trim_idxes)
        loss_res = kl_crit(log_sm(res_emo), gt_emotion)
        loss_cen = center_crit(feat_emo, gt_emotion.argmax(1))
        loss = loss_res + alpha * loss_cen

        if mode == 'train':
            model_opt.zero_grad()
            cent_opt.zero_grad()
            loss.backward()
            model_opt.step()
            cent_opt.step()
        else:
            pred_ids = res_emo.argmax(1)
            gt_ids = gt_emotion.argmax(1)
            acc.add(pred_ids, gt_ids)
            conf_pred.extend(pred_ids.detach().cpu().numpy().tolist())
            conf_gt.extend(gt_ids.detach().cpu().numpy().tolist())

        total_loss[0] += loss_res.item()
        total_loss[1] += loss_cen.item()
    if mode != 'train':
        mm = confusion_matrix(conf_gt, conf_pred, normalize='true') # prediction
        mm2 = confusion_matrix(conf_gt, conf_pred, normalize='pred') # recall
        wa, ua = acc.fin()
    else:
        mm = None
        mm2 = None
        wa, ua = None, None
    return [tl/(i+1) for tl in total_loss], (wa, ua), (mm, mm2)

def collate_batch(batch):
    name_list = []
    signal_list = []
    trimid_list = []
    emotion_list = []
    for file_name, wav_signal_padded, trim_idx, label in batch:
        name_list.append(file_name)
        signal_list.append(wav_signal_padded)
        trimid_list.append(trim_idx)
        emotion_list.append(label)
    signal_array = np.array(signal_list)
    emo_array = np.array(emotion_list)
    return name_list, torch.tensor(signal_array), trimid_list, torch.tensor(emo_array, dtype=torch.float32)

def train(Paug, Pmix, alpha, model_file=None):
    data_train, data_valid, data_test = loadData(Paug, Pmix)
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREAD, collate_fn=collate_batch)
    valid_dataloader = torch.utils.data.DataLoader(data_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, collate_fn=collate_batch)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, collate_fn=collate_batch)
    model = SpeechModel(MAX_LEN, SELECT_FT, len(emotions))
    model.to(cuda)
    if WAV2VEC2_DROPOUT:
        for layer in range(SELECT_FT+1):
            model.base.wav2vec2.encoder.transformer.layers[layer].dropout.p=WAV2VEC2_DROPOUT

    if model_file:
        pretrain_dict = torch.load(model_file)
        model_dict = model.state_dict()
        pretrain_dict = {k:v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print(f'Pretrain model {model_file} loaded.')
    else:
        print('Train from scratch.')
    model_opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    kl_crit = torch.nn.KLDivLoss(reduction='batchmean')
    kl_crit.to(cuda)
    log_sm = torch.nn.LogSoftmax(dim=-1)
    log_sm.to(cuda)
    center_crit = CenterLoss(num_classes=len(emotions), feat_dim=FEAT_SIZE)
    cent_opt = torch.optim.Adam(center_crit.parameters(), lr=LR_CENTER, betas=(0.9, 0.98), eps=1e-9)
    opt = [model_opt, cent_opt]
    criterion = [log_sm, kl_crit, center_crit]
    scheduler_model = optim.lr_scheduler.MultiStepLR(model_opt, milestones=lr_milestone, gamma=lr_gamma)
    scheduler_center = optim.lr_scheduler.MultiStepLR(cent_opt, milestones=lr_milestone, gamma=lr_gamma)

    max_acc = 0
    max_acc_idx = 0
    max_acc_count = 0

    for epoch in range(100):
        model.train()
        lr = scheduler_model.get_last_lr()[0]
        start = time.time()
        loss, (wa, ua), conf_matrix = run_epoch(train_dataloader, model, criterion, epoch, alpha, opt)
        scheduler_model.step()
        scheduler_center.step()
        if type(loss) is list or type(loss) is tuple:
            loss_str = []
            for i in range(len(loss)):
                loss_str.append(f'{loss[i]:.2f}')
            loss_str = '|'.join(loss_str)
        elif type(loss) is float:
            loss_str = f'{loss:.2f}'
        print(f'Train Epoch [{epoch}]: Loss: {loss_str}, lr: {lr:.6f}, time: {time.time()-start:.1f}s')

        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save(model.state_dict(), f'checkpoints/speech_emotion_model_aux-{epoch}.model')

        # Evaluation
        start_t = time.time()
        loss_t, (wa_t, ua_t), conf_matrix_t = test(valid_dataloader, model, criterion, epoch, alpha)
        if type(loss_t) is list or type(loss_t) is tuple:
            loss_str_t = []
            for i in range(len(loss_t)):
                loss_str_t.append(f'{loss_t[i]:.2f}')
            loss_str_t = '|'.join(loss_str_t)
        elif type(loss_t) is float:
            loss_str_t = f'{loss_t:.2f}'
        print(f'Valid Epoch [{epoch}]: Loss: {loss_str_t}, Acc: [WA{wa_t*100:.2f}%-UA{ua_t*100:.2f}%], time: {time.time()-start_t:.1f}s')
        print_confusion_matrix(conf_matrix_t[0])
        print_confusion_matrix(conf_matrix_t[1])
        acc_t = wa_t
        if acc_t > max_acc:
            max_acc = acc_t
            max_acc_idx = epoch
            max_acc_count = 0
            rm_old_model(max_acc_idx)
        else:
            max_acc_count += 1
        if max_acc_count >= EARLY_STOP:
            loss_tt, (wa_tt, ua_tt), conf_matrix_tt = test(test_dataloader, model, criterion, max_acc_idx, alpha)
            print(f'Early stops at {epoch} epoch, best epoch is {max_acc_idx} with validation accuracy: {max_acc*100:.2f}% and test accuracy: {wa_tt*100:.2f}%')
            return max_acc_idx, wa_tt*100

def rm_old_model(idx):
    models = glob.glob('checkpoints/*.model')
    for m in models:
        epoch = int(m.split('.')[0].split('-')[1])
        if epoch < idx:
            os.system(f'rm checkpoints/speech_emotion_model_aux-{epoch}.model')

def test(test_dataloader, model, criterion, epoch, alpha):
    model.eval()
    with torch.no_grad():
        loss_t, (wa_t, ua_t), conf_matrix = run_epoch(test_dataloader, model, criterion, epoch, alpha)
    return loss_t, (wa_t, ua_t), conf_matrix


if __name__ == '__main__':
    Paug = 0.8
    Pmix = 0.5
    alpha = 0.002
    for foldn in range(5):
        with open('fold.num', 'w') as fofo:
            fofo.write(str(foldn))
        best_epoch, best_acc = train(Paug, Pmix, alpha)
        print(f'RESULT: fold:{foldn}-epoch:{best_epoch}-acc:{best_acc:.2f}%')
