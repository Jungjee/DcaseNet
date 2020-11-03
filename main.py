import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import pickle as pk
import pandas as pd
import argparse
import importlib
import soundfile as sf

from tqdm import tqdm
from loss import binary_cross_entropy
from utils import *
from parser import get_args
from metric import metric_manager
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from trainer import train_joint_3task, evaluate_ASC, evaluate_SED, evaluate_TAG
from torchsummary import summary
from sklearn.model_selection import KFold
from dataloaders import get_loaders_ASC, get_loaders_SED, get_loaders_TAG

def main():
    #parse arguments 
    args = get_args()

    #device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    #strictly reproducible, with potential speed loss as a trade-off!
    if args.reproducible:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #pre-process DB for first time
    #get DB list: SED
    if 'SED' in args.task:
        if not os.path.exists(args.DB_SED+'log_mel_spec_label/'):
            lines_SED = get_utt_list(args.DB_SED+args.wav_SED)
            from preprocess import extract_log_mel_spec_sed
            print(extract_log_mel_spec_sed(lines_SED, args))

        lines_SED = get_utt_list(args.DB_SED+'log_mel_spec_label/', ext='h5')
        trn_lines_SED, evl_lines_SED = split_dcase2020_sed(lines_SED)
        if args.verbose > 0:
            print('SED DB statistics')
            print('# tot samp: {}\n'\
              '# trn samp: {}\n'\
              '# evl samp: {}\n'.format(len(lines_SED), len(trn_lines_SED), len(evl_lines_SED)))
            print(sed_labels)
            print(sed_label2idx)
            print(sed_idx2label)
        del lines_SED

    #get DB list: ASC
    if 'ASC' in args.task:
        lines_ASC = get_utt_list(args.DB_ASC+args.wav_ASC)

        if os.path.exists(args.DB_ASC+args.meta_scp):
            d_label_ASC, l_label_ASC = pk.load(open(args.DB_ASC+args.d_label_ASC, 'rb'))
        else:
            with open(args.DB_ASC+args.meta_scp) as f:
                l_meta_ASC = f.readlines()
            d_label_ASC, l_label_ASC = make_d_label(l_meta_ASC[1:])
            pk.dump([d_label_ASC, l_label_ASC], open(args.DB_ASC+args.d_label_ASC, 'wb'))

        trn_lines_ASC = split_dcase2020_fold_strict(fold_scp = args.DB_ASC+args.fold_trn, lines = lines_ASC)
        evl_lines_ASC = split_dcase2020_fold_strict(fold_scp = args.DB_ASC+args.fold_evl, lines = lines_ASC)
        if args.verbose > 0 :
            print('ASC DB statistics')
            print('# trn samp: {}\n# evl samp: {}'.format(len(trn_lines_ASC), len(evl_lines_ASC)))
            print(d_label_ASC)
            print(l_label_ASC)

    #get DB list: Audio tagging
    if 'TAG' in args.task:
        df_TAG = pd.read_csv(args.DB_TAG+'train_curated.csv')
        tmp_df = pd.read_csv(args.DB_TAG+'sample_submission.csv')
        l_label_TAG = tmp_df.columns[1:].tolist()  #get 80 audio tagging labels as list
        del tmp_df
    
        for l in l_label_TAG:
            df_TAG[l] = df_TAG['labels'].apply(lambda x: l in x)
        df_TAG['path'] = args.DB_TAG + 'train_curated/' + df_TAG['fname']
        # all arguments must be fixed for reproducing original 
        # fold configuration reported in Akiyams et al.'s paper.
        trn_idx_TAG, evl_idx_TAG = list(KFold(
            n_splits=5,
            shuffle=True,
            random_state=42).split(np.arange(len(df_TAG))))[0]
        df_trn_TAG = df_TAG.iloc[trn_idx_TAG].reset_index(drop=True)
        df_evl_TAG = df_TAG.iloc[evl_idx_TAG].reset_index(drop=True)
        del df_TAG
    
        if args.verbose > 0:
            print('Audio tagging DB statistics')
            print('# trn samp: {}\n# evl samp: {}'.format(len(df_trn_TAG.fname), len(df_evl_TAG.fname)))
            print(l_label_TAG)
    

    #####
    #define dataset generators
    #####
    #SED
    if 'SED' in args.task:
        largs = {'trn_lines': trn_lines_SED,
                 'evl_lines': evl_lines_SED
        }
        trnset_gen_SED, evlset_gen_SED = get_loaders_SED(largs, args)
        trnset_gen_SED_itr = cycle(trnset_gen_SED)
    else:
        trnset_gen_SED_itr = None

    #ASC
    if 'ASC' in args.task:
        largs = {
            'trn_lines': trn_lines_ASC,
            'evl_lines': evl_lines_ASC,
            'd_label': d_label_ASC,
        }
        trnset_gen_ASC, evlset_gen_ASC = get_loaders_ASC(largs, args)
        trnset_gen_ASC_itr = cycle(trnset_gen_ASC)
    else:
        trnset_gen_ASC_itr = None

    #TAG
    if 'TAG' in args.task:
        largs = {
            'trn': df_trn_TAG,
            'evl': df_evl_TAG,
            'l_label': l_label_TAG
        }
        trnset_gen_TAG, evlset_gen_TAG = get_loaders_TAG(largs, args)
        trnset_gen_TAG_itr = cycle(trnset_gen_TAG)
    else:
        trnset_gen_TAG_itr = None 

    #set save directory
    save_dir = args.save_dir+args.name+'/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    if not os.path.exists(save_dir+'results/'): os.makedirs(save_dir+'results/')
    if not os.path.exists(save_dir+'weights/'): os.makedirs(save_dir+'weights/')
    
    #log parameters to local and comet_ml server
    f_params = open(save_dir+'f_params.txt', 'w')
    for k, v in sorted(vars(args).items()):
        if args.verbose > 0: print(k, v)
        f_params.write('{}:\t{}\n'.format(k, v))
    f_params.close()

    #define model
    module = importlib.import_module('models.{}'.format(args.model_scp))
    _model = getattr(module, args.model_name)
    model = _model(**args.model)
    model_summ = summary(model, (1, 128, 251), mode = args.task)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    if args.verbose >0: print('nb_params: %d'%nb_params)
    with open(save_dir+'modelsumm.txt', 'w') as f: f.write(str(model_summ))

    #load weights
    if 'fine-tune' in args.name or 'Eval' in args.name:
        model.load_state_dict(torch.load(args.dir_model_weight))
    model = model.to(device)
    
    if 'Eval' in args.name:
        if 'ASC' in args.task:
            acc, conf_mat = evaluate_ASC(model = model,
                evlset_gen = evlset_gen_ASC,
                device = device,
                args = args,
            )
            print('ASC acc:\t{}'.format(acc))

        if 'SED' in args.task:
            er, f1 = evaluate_SED(model = model,
                evlset_gen = evlset_gen_SED,
                device = device,
                args = args,
            )
            print('ER:{}\tF1:{}\t'.format(er, f1))

        if 'TAG' in args.task:
            lwlrap = evaluate_TAG(model = model,
                evlset_gen = evlset_gen_TAG,
                device = device,
                args = args
            )
            print('Lwlrap:{}\t'.format(lwlrap))
    else:
        #set ojbective funtions
        criterion = {
            'bce_SED': binary_cross_entropy,
            'cce_ASC': nn.CrossEntropyLoss().cuda(),
            'bce_TAG': nn.BCEWithLogitsLoss().cuda()
        }

        #set optimizer
        params = list(model.parameters())
        if args.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(params,
                lr = args.lr,
                momentum = args.opt_mom,
                weight_decay = args.wd,
                nesterov = args.nesterov)
        elif args.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),
                lr = args.lr,
                weight_decay = args.wd,
                amsgrad = args.amsgrad)
        else:
            raise NotImplementedError('Optimizer not implemented, got:{}'.format(args.optimizer))

        #set learning rate decay
        if bool(args.do_lr_decay):
            if args.lr_decay == 'cosine':
                lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.nb_iter_per_epoch * args.lrdec_t0, eta_min=0.000001)
            else:
                raise NotImplementedError('Not implemented yet')
        else:
            lr_scheduler = None

        f_eval = open(save_dir + 'eval_results.txt', 'a', buffering=1)
        metric_man = metric_manager(
            task=args.task,
            save_dir=save_dir+'weights/',
            model=model,
            save_best_only=args.save_best_only
        ) 

        for epoch in tqdm(range(args.epoch), ncols=100):
            train_joint_3task(
                model=model,
                args=args,
                trnset_gen_ASC=trnset_gen_ASC_itr,
                trnset_gen_SED=trnset_gen_SED_itr,
                trnset_gen_TAG=trnset_gen_TAG_itr,
                epoch=epoch,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )
            description = 'Epoch{}:\t'.format(epoch)

            if 'ASC' in args.task:
                acc, conf_mat = evaluate_ASC(
                    model=model,
                    evlset_gen=evlset_gen_ASC,
                    device=device,
                    args=args,
                )
                description += 'Acc:{}\t'.format(acc)
                metric_man.update_ASC(epoch=epoch, acc=acc, conf_mat=conf_mat, l_label=l_label_ASC)

            if 'SED' in args.task:
                er, f1 = evaluate_SED(
                    model=model,
                    evlset_gen=evlset_gen_SED,
                    device=device,
                    args=args,
                )
                description += 'ER:{}\tF1:{}\t'.format(er, f1)
                metric_man.update_SED(epoch=epoch, er=er, f1=f1)

            if 'TAG' in args.task:
                lwlrap = evaluate_TAG(
                    model=model,
                    evlset_gen=evlset_gen_TAG,
                    device=device,
                    args=args
                )
                description += 'Lwlrap:{}\t'.format(lwlrap)
                metric_man.update_TAG(epoch=epoch, lwlrap=lwlrap)
            
            f_eval.write(description+'\n')
        f_eval.close()

if __name__ == '__main__':
    main()