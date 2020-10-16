from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import KFold
from torchsummary import summary

import os
import struct
import torch
import argparse
import json
import sys
import torch
import importlib
import soundfile as sf
import pickle as pk
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from parser import get_args
from dataloaders import get_loaders_ASC, get_loaders_SED, get_loaders_TAG
from utils import *
from trainer import train_joint_3task, evaluate_ASC, evaluate_SED, evaluate_TAG
from loss import binary_cross_entropy
from metric import metric_manager

def main():
    #parse arguments 
    args = get_args()

    #device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    #get DB list: SED
    if 'SED' in args.task:
        #########
        ## Add ##
        #########
        # add for checking 'log_mel_spec_label' exist 
        ##############################################################################
        if not os.path.exists(args.DB_SED+'log_mel_spec_label/'):
            lines_SED = get_utt_list(args.DB_SED+args.wav_SED)
            from preprocess import extract_log_mel_spec_sed
            print(extract_log_mel_spec_sed(lines_SED, args))
        ##############################################################################

        lines_SED = get_utt_list(args.DB_SED+'log_mel_spec_label/', ext = 'h5')
        trn_lines_SED, evl_lines_SED =split_dcase2020_sed(lines_SED)
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



        ##############
        ## modified ##
        ##############
        # Change: use flag -> check existence
        # TODO: please select one and remove this
        ######################################################################
        """
        # prev
        if args.make_d_label_ASC:
            with open(args.DB_ASC+args.meta_scp) as f:
                l_meta_ASC = f.readlines()
            d_label_ASC, l_label_ASC = make_d_label(l_meta_ASC[1:])
            pk.dump([d_label_ASC, l_label_ASC], open(args.DB_ASC+args.d_label_ASC, 'wb'))
        else:
            d_label_ASC, l_label_ASC = pk.load(open(args.DB_ASC+args.d_label_ASC, 'rb'))
        """
        # current
        if os.path.exists(args.DB_ASC+args.meta_scp):
            d_label_ASC, l_label_ASC = pk.load(open(args.DB_ASC+args.d_label_ASC, 'rb'))
        else:
            with open(args.DB_ASC+args.meta_scp) as f:
                l_meta_ASC = f.readlines()
            d_label_ASC, l_label_ASC = make_d_label(l_meta_ASC[1:])
            pk.dump([d_label_ASC, l_label_ASC], open(args.DB_ASC+args.d_label_ASC, 'wb'))
        ######################################################################



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
        #print(df_trn_TAG)
        #print(df_evl_TAG)
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
        '''
        if args.verbose>=2:
            for a in trnset_gen_SED:
                print(a[0].size())
                print(a[1].size())
                break
            for a in evlset_gen_SED:
                print(a[0].size())
                print(a[1].size())
                break
        '''
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
        '''
        if args.verbose>=2:
            print(len(trnset_gen_ASC))
            print(len(evlset_gen_ASC))
            for a in trnset_gen_ASC:
                print(a[0].size())
                print(a[1].size())
                break
            for a in evlset_gen_ASC:
                print(a[0].size())
                print(a[1].size())
                break
        '''
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
        '''
        if args.verbose>=2:
            print(len(trnset_gen_TAG))
            print(len(evlset_gen_TAG))
            for a in trnset_gen_TAG:
                print(a[0].size())
                print(a[1].size())
                break
            for a in evlset_gen_TAG:
                print(a[0].size())
                print(a[1].size())
                break
        exit()
        '''
    else:
        trnset_gen_TAG_itr = None
    
    #define model
    module = importlib.import_module('models.{}'.format(args.model_scp))
    _model = getattr(module, args.model_name)
    model = _model(**args.model)

    #load weights
    model.load_state_dict(torch.load(args.dir_model_weight))
    model = model.to(device)

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

if __name__ == '__main__':
    main()