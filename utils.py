import os
import torch
import numpy as np

def get_utt_list(src_dir, ext='wav'):
    l_utt = []
    for _, _, fs in os.walk(src_dir):
        for f in fs:
            if os.path.splitext(f)[1] != '.'+ext:
                continue
            l_utt.append(f)

    return l_utt

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def make_d_label(lines):
    idx = 0
    dic_label = {}
    list_label = []
    for line in lines:
        label = line.strip().split('/')[1].split('-')[0]
        if label not in dic_label:
            dic_label[label] = idx
            list_label.append(label)
            idx += 1
    return dic_label, list_label

def split_dcase2020_fold_strict(fold_scp, lines):
    fold_lines = open(fold_scp, 'r').readlines()
    l_return = []
    l_fold = []

    for line in fold_lines[1:]:
        l_fold.append(line.strip().split('\t')[0].split('/')[1])
    for line in lines:
        if line in l_fold:
            l_return.append(line)

    return l_return

def split_dcase2020_sed(lines):
    trn_folds = ['fold3', 'fold4', 'fold5', 'fold6']
    evl_folds = ['fold1']

    trn_lines = []
    evl_lines = []
    for l in lines:
        fold_idx = l.strip().split('_')[0]
        if fold_idx in trn_folds:
            trn_lines.append(l)
        elif fold_idx in evl_folds:
            evl_lines.append(l)
        else:
            continue
    return trn_lines, evl_lines

def cycle(iterable):
    """
    convert dataloader to iterator
    :param iterable:
    :return:
    """
    while True:
        for x in iterable:
            yield x

def uptohere():
    print('='*5+ '\nsuccess\n'+'='*5)
    exit()

sed_labels = ['alarm', 'baby', 'crash', 'dog', 'engine', 'female_screem', \
              'female_speech', 'fire', 'footsteps', 'knock', 'male_screem', \
              'male_speech', 'phone', 'piano']
sed_label2idx = {lb: i for i, lb in enumerate(sed_labels)}
sed_idx2label = {i: lb for i, lb in enumerate(sed_labels)}
