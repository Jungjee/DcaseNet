import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

from utils import *
from metric import compute_sed_scores, calculate_per_class_lwlrap

#####
# ASC & SED & TAG Joint Training
#####
def train_joint_3task(model, trnset_gen_ASC, trnset_gen_SED, trnset_gen_TAG, epoch, args, device, criterion, optimizer, lr_scheduler):
    #train phase
    model.train()
    idx_ct_start = int(args.nb_iter_per_epoch*epoch)
    if args.do_mixup:
        mixup = True if epoch > args.mixup_start else False
    else:
        mixup = False
    loss = 0.
    loss_SED = 0.
    loss_ASC = 0.
    loss_TAG = 0.
    with tqdm(total = args.nb_iter_per_epoch, ncols = 100) as pbar:
        for idx in range(args.nb_iter_per_epoch):
            _loss = 0.
            #####
            # ASC feed forward
            #####
            if 'ASC' in args.task:
                m_batch, m_label = next(trnset_gen_ASC)
                m_batch, m_label = m_batch.to(device), m_label.to(device)

                #mixup data if condition is met
                if mixup:
                    m_batch, m_label_a, m_label_b, lam = mixup_data(m_batch, m_label,
                        alpha = args.mixup_alpha,
                        use_cuda = True)
                    m_batch, m_label_a, m_label_b = map(torch.autograd.Variable, [m_batch, m_label_a, m_label_b])
                    out_ASC = model(m_batch, mode = ['ASC'])['ASC']
                    _loss_ASC = mixup_criterion(criterion['cce_ASC'], out_ASC, m_label_a, m_label_b, lam)
                else:
                    out_ASC = model(m_batch, mode = ['ASC'])['ASC']
                    _loss_ASC = criterion['cce_ASC'](out_ASC, m_label)

                _loss += args.loss_weight_ASC * _loss_ASC
                loss_ASC += _loss_ASC.detach().cpu().numpy()
            
            #####
            # SED feed forward
            #####
            if 'SED' in args.task:
                m_batch, m_label = next(trnset_gen_SED)
                m_batch, m_label = m_batch.to(device), m_label.to(device, dtype=torch.float)
                #out_SED = model(m_batch, mode = ['SED'])['SED']
                '''
                if args.verbose>=2: 
                    print(out_SED.size())
                    print(m_label.size())
                '''
                #_loss_SED = criterion['bce_SED'](out_SED, m_label)
                #mixup data if condition is met
                if mixup:
                    m_batch, m_label_a, m_label_b, lam = mixup_data(m_batch, m_label,
                        alpha = args.mixup_alpha,
                        use_cuda = True)
                    m_batch, m_label_a, m_label_b = map(torch.autograd.Variable, [m_batch, m_label_a, m_label_b])
                    out_SED = model(m_batch, mode = ['SED'])['SED']
                    _loss_SED = mixup_criterion(criterion['bce_SED'], out_SED, m_label_a, m_label_b, lam)
                else:
                    out_SED = model(m_batch, mode = ['SED'])['SED']
                    _loss_SED = criterion['bce_SED'](out_SED, m_label)

                _loss += args.loss_weight_SED * _loss_SED
                loss_SED += _loss_SED.detach().cpu().numpy()

            #####
            # TAG feed forward
            #####
            if 'TAG' in args.task:
                m_batch, m_label = next(trnset_gen_TAG)
                m_batch, m_label = m_batch.to(device), m_label.to(device, dtype=torch.float)
                out_TAG = model(m_batch, mode = ['TAG'])['TAG']
                '''
                if args.verbose>=2: 
                    print(out_TAG)
                    print(m_label)
                '''
                #_loss_TAG = criterion['bce_TAG'](out_TAG, m_label)
                if mixup:
                    m_batch, m_label_a, m_label_b, lam = mixup_data(m_batch, m_label,
                        alpha = args.mixup_alpha,
                        use_cuda = True)
                    m_batch, m_label_a, m_label_b = map(torch.autograd.Variable, [m_batch, m_label_a, m_label_b])
                    out_TAG = model(m_batch, mode = ['TAG'])['TAG']
                    _loss_TAG = mixup_criterion(criterion['bce_TAG'], out_TAG, m_label_a, m_label_b, lam)
                else:
                    out_TAG = model(m_batch, mode = ['TAG'])['TAG']
                    _loss_TAG = criterion['bce_TAG'](out_TAG, m_label)

                _loss += args.loss_weight_TAG * _loss_TAG
                loss_TAG += _loss_TAG.detach().cpu().numpy()

            loss += _loss.detach().cpu().numpy()
            #print(loss, loss_ASC)
            #####
            # Joint back-prop
            #####
            #loss = args.loss_weight_SED * _loss_SED + args.loss_weight_ASC * _loss_ASC + args.loss_weight_TAG * _loss_TAG
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            if idx % args.nb_iter_per_log == 0:
                description = '%s\tepoch: %d '%(args.name, epoch)
                if idx != 0:
                    if 'SED' in args.task:
                        loss_SED /= args.nb_iter_per_log
                        description += 'SED: %.3f '%(loss_SED)
                        loss_SED = 0.
                    if 'ASC' in args.task:
                        loss_ASC /= args.nb_iter_per_log
                        description += 'ASC: %.3f '%(loss_ASC)
                        loss_ASC = 0.
                    if 'TAG' in args.task:
                        loss_TAG /= args.nb_iter_per_log
                        description += 'TAG: %.3f '%(loss_TAG)
                        loss_TAG = 0.
                    loss /= args.nb_iter_per_log
                    description += 'TOT: %.4f'%(loss)
                    loss = 0.

                pbar.set_description(description)
                if idx != 0:
                    pbar.update(args.nb_iter_per_log)
                else:
                    pbar.update(1)
                for p_group in optimizer.param_groups:
                    lr = p_group['lr']
                    break

            if args.do_lr_decay:
                if args.lr_decay == 'cosine':
                    lr_scheduler.step()
                else:
                    raise NotImplementedError('Not just yet..')

#####
# SED
#####
def evaluate_SED(model, evlset_gen, device, args):
    y_true = []
    y_pred = []
    y_keys = []
    model.eval()
    with torch.set_grad_enabled(False):
        for m_batch, m_label, m_keys in evlset_gen:
            m_batch = m_batch.to(device)
            out = model(m_batch, mode = ['SED'])['SED']
            if args.verbose >= 3: print('out shape: ', out.size()) #should be (bs, 600, 14)
            y_true.append(m_label.numpy())
            y_keys.append(m_keys)
            y_pred.append(out.detach().cpu().numpy())


    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_keys = np.concatenate(y_keys, axis=0)

    if args.verbose >=2: 
        print('y_true:',y_true.shape)
        print('y_pred:',y_pred.shape)
        print('y_keys:',y_keys.shape)
        assert len(y_true) == len(y_pred) == len(y_keys)

    d_decode = OrderedDict()
    d_true = OrderedDict()
    for p, k, t in zip(y_pred, y_keys, y_true):
        k_seg = k.split(':')[0]
        if k_seg not in d_decode:
            d_decode[k_seg] = np.zeros((4, 600, 14))
            d_true[k_seg] = t
        c = int(k.strip().split(':')[-1][2])-1   #channel
        d_decode[k_seg][c,:,:] = p

    y_pred, y_true = [], []
    for k, v in d_decode.items():
        y_pred.append(np.mean(v, axis = 0))
        y_true.append(d_true[k])
    y_pred = np.concatenate(y_pred, axis = 0)
    y_true = np.concatenate(y_true, axis = 0)
    if args.verbose >=2: 
        print('y_true:',y_true.shape)
        print('y_pred:',y_pred.shape)


    er, f1 = compute_sed_scores(
        pred = (y_pred>args.sed_threshold).astype(np.float32),
        gt = y_true,
        nb_frames_1s = 50
    )
    if args.verbose>=2:
        print('y_true: ', y_true.shape)
        print('y_pred: ', y_pred.shape)
        assert y_true.shape == y_pred.shape
    if args.verbose > 0: print('Eval SED success')

    return er, f1

#####
# ASC
#####
def evaluate_ASC(model, evlset_gen, device, args):
    y_pred = []
    y_true = []
    model.eval()
    with torch.set_grad_enabled(False):
        for m_batch, m_label in evlset_gen:
            m_batch = m_batch.view(-1, 1, args.nb_mels, args.nb_frames_ASC+1).to(device)
            out = model(m_batch, mode = ['ASC'])['ASC']
            out = F.softmax(out, dim=-1).view(-1, 3, out.size(1)).mean(dim=1, keepdim=False)
            m_label = list(m_label.numpy())
            y_pred.extend(list(out.cpu().numpy())) #>>> (16, 64?)
            y_true.extend(m_label)

        y_pred = np.argmax(np.array(y_pred), axis=1).tolist()
        conf_mat = confusion_matrix(y_true = y_true, y_pred = y_pred)
        nb_cor = 0
        for i in range(len(conf_mat)):
            nb_cor += conf_mat[i,i]
            conf_mat[i,i] = 0
        acc = nb_cor / len(y_true) * 100 
    if args.verbose > 0: print('Eval ASC success')

    return acc, conf_mat

#####
# TAG 
#####
def evaluate_TAG(model, evlset_gen, device, args):
    model.eval()
    y_pred = np.zeros([0, 80], np.float32)
    y_true = np.zeros([0, 80], np.float32)
    with torch.set_grad_enabled(False):
        for m_batch, m_label in evlset_gen:
            #print(m_batch.size())
            #exit()
            #m_batch = m_batch.view(-1, 1, args.nb_mels, args.nb_frames_ASC+1).to(device)
            m_batch = m_batch.to(device)
            out = torch.sigmoid(model(m_batch, mode = ['TAG'])['TAG'])
            m_label = list(m_label.numpy())

            y_pred = np.concatenate([y_pred, out.detach().cpu().numpy()])
            y_true = np.concatenate([y_true, m_label])
            #print(y_pred.shape)
            #print(y_true.shape)

    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(y_true, y_pred)
    lwlrap = np.sum(per_class_lwlrap * weight_per_class)

    if args.verbose > 0: print('Eval TAG success')
    return lwlrap