import torch
import h5py
import numpy as np
import soundfile as sf
import torchaudio as ta

from torch.utils import data
from utils import *

#####
# SED
#####
def get_loaders_SED(loader_args, args):
    trnset = Dataset_DCASE2020_sed(
        lines = loader_args['trn_lines'],
        base_dir = args.DB_SED+args.h5_dir_SED,
        trn = True
    )
    trnset_gen = data.DataLoader(
        trnset,
        batch_size = args.bs_SED,
        shuffle = True,
        num_workers = args.nb_worker,
        pin_memory = True,
        drop_last = True
    )

    evlset = Dataset_DCASE2020_sed(
        lines = loader_args['evl_lines'],
        base_dir = args.DB_SED+args.h5_dir_SED,
        trn = False,
    )
    evlset_gen = data.DataLoader(
        evlset,
        batch_size = args.bs_SED,
        shuffle = False,
        num_workers = args.nb_worker*2,
        pin_memory = True,
        drop_last = False
    )

    return trnset_gen, evlset_gen

class Dataset_DCASE2020_sed(data.Dataset):
    def __init__(
        self,
        lines,
        trn = True,
        base_dir='',
        ):
        self.lines = lines 
        self.trn = trn
        self.base_dir = base_dir

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        with h5py.File(self.base_dir+self.lines[index], 'r') as f:
            X = f['log_mel_spec'][()]
            y = f['label_sed'][()]

        if self.trn:
            stt_idx = np.random.randint(0, 1500) // 5
            X = X[:,stt_idx*5:stt_idx*5+1500].reshape(1,128,1500)
            y = y[stt_idx:stt_idx+300,:]
        else:
            X = X.reshape(1,128,-1)

        return (X, y) if self.trn else (X, y, self.lines[index])


#####
# ASC
#####
def get_loaders_ASC(loader_args, args):
    trnset = Dataset_DCASE2020_t1(
        lines = loader_args['trn_lines'],
        base_dir = args.DB_ASC+args.wav_ASC,
        d_label = loader_args['d_label'],
        verbose = args.verbose
    )
    trnset_gen = data.DataLoader(
        trnset,
        batch_size = args.bs_ASC,
        shuffle = True,
        num_workers = args.nb_worker,
        pin_memory = True,
        drop_last = True
    )

    evlset = Dataset_DCASE2020_t1(
        lines = loader_args['evl_lines'],
        trn = False,
        base_dir = args.DB_ASC+args.wav_ASC,
        d_label = loader_args['d_label'],
        verbose = args.verbose
    )
    evlset_gen = data.DataLoader(
        evlset,
        batch_size = args.bs_ASC//3,
        shuffle = False,
        num_workers = args.nb_worker//2,
        pin_memory = True,
        drop_last = False
    )

    return trnset_gen, evlset_gen

class Dataset_DCASE2020_t1(data.Dataset):
    def __init__(self, lines, nb_frames = 0, trn = True, base_dir = '', d_label = '', verbose = 0):
        self.lines = lines 
        self.d_label = d_label
        self.base_dir = base_dir
        #self.nb_frames = nb_frames
        self.trn = trn
        self.verbose = verbose
        #self.return_label = return_label

        self.resample = ta.transforms.Resample(
            orig_freq=44100,
            new_freq=24000,
            resampling_method='sinc_interpolation')
        self.melspec = ta.transforms.MelSpectrogram(
            24000,
            n_fft=2048,
            win_length=int(24000 * 0.001 * 40),
            hop_length=int(24000 * 0.001 * 20),
            window_fn=torch.hamming_window,
            n_mels=128)
        self.nb_samps = int(24000 * 0.001 * 20 * 250)   #(#samp_rate, into ms, #frames, #frames)
        self.margin = int(240000 - self.nb_samps)
        if not trn: self.TTA_mid_idx = int(self.nb_samps/2)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        k = self.lines[index]
        try:
            X, samp_rate = ta.load(self.base_dir+k, normalization = True)   #X.size : (1, 441000)
            #if self.verbose > 3: print('Loaded: ', X.size())
            X = self.resample(X) #X.size    : (1, 240000)
            #if self.verbose > 3: print('Resampled: ', X.size())
            #assert samp_rate == self.samp_rate
        except:
            raise ValueError('Unable to laod utt %s'%k)
        X = self._pre_emphasis(X)
        #if self.verbose > 3: print('Pre-emphasized: ', X.size())

        if self.trn:
            st_idx = np.random.randint(0, self.margin)
            X = X[:, st_idx:st_idx+self.nb_samps]
        else:
            l_X = []
            l_X.append(X[:,:self.nb_samps])
            l_X.append(X[:,self.TTA_mid_idx:self.TTA_mid_idx+self.nb_samps])
            l_X.append(X[:,-self.nb_samps:])
            X = torch.stack(l_X)
        X = self.melspec(X)

        ###### 
        X = torch.log(X)    #2020.8.2.

        #if self.verbose > 3: print('Mel-spec: ', X.size())
        X = self._utt_mvn(X)
        #print(X.size())   
        #trn: (1, 128, 251) dev: (3, 128, 251)

        #if self.trn:
        #    y = self.d_label[k.split('-')[0]]
        #    return X, y
        #else:
        #    return X
        y = self.d_label[k.split('-')[0]]
        return X, y

    
    def _pre_emphasis(self, x):
        return x[:,1:] - 0.97 * x[:, :-1] 

    def _utt_mvn(self, x):
        _m = x.mean(dim=-1, keepdim = True)
        _s = x.std(dim=-1, keepdim = True)
        _s[_s<0.001] = 0.001
        return (x - _m) / _s

#####
# Audio tagging
#####
def get_loaders_TAG(loader_args, args):
    trnset = Dataset_DCASE2019_TAG(
        X = loader_args['trn']['path'],
        y = loader_args['trn'][loader_args['l_label']].values,
        trn = True,
        verbose = args.verbose
    )
    trnset_gen = data.DataLoader(
        trnset,
        batch_size = args.bs_TAG,
        shuffle = True,
        num_workers = args.nb_worker,
        pin_memory = True,
        drop_last = True
    )

    evlset = Dataset_DCASE2019_TAG(
        X = loader_args['evl']['path'],
        y = loader_args['evl'][loader_args['l_label']].values,
        trn = False,
        verbose = args.verbose
    )

    evlset_gen = data.DataLoader(
        evlset,
        batch_size = 1,
        shuffle = False,
        num_workers = 2,
        pin_memory = True,
        drop_last = False
    )

    return trnset_gen, evlset_gen

class Dataset_DCASE2019_TAG(data.Dataset):
    def __init__(self, X, y, trn, verbose = 0):
        self.X = X
        self.y = y
        self.trn = trn
        self.verbose = verbose

        self.resample = ta.transforms.Resample(
            orig_freq=44100,
            new_freq=24000,
            resampling_method='sinc_interpolation')
        self.melspec = ta.transforms.MelSpectrogram(
            24000,
            n_fft=2048,
            win_length=int(24000 * 0.001 * 40),
            hop_length=int(24000 * 0.001 * 20),
            window_fn=torch.hamming_window,
            n_mels=128)
        self.nb_samps = int(24000 * 0.001 * 20 * 250)   #(#samp_rate, into ms, #ms, #frames)
        #self.margin = int(240000 - self.nb_samps)
        #if not trn: self.TTA_mid_idx = int(self.nb_samps/2)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        k = self.X[index]
        try:
            X, samp_rate = ta.load(k, normalization = True)   #X.size : (1, 441000)
            #if self.verbose > 3: print('Loaded: ', X.size())
            X = self.resample(X) #X.size    : (1, 240000)
            #if self.verbose > 3: print('Resampled: ', X.size())
            #assert samp_rate == self.samp_rate
        except:
            raise ValueError('Unable to laod utt %s'%k)
        X = self._pre_emphasis(X)
        #if self.verbose > 3: print('Pre-emphasized: ', X.size())

        if self.trn:
            #print(X.size())
            while X.size(1) < self.nb_samps:
                X = torch.cat([X, X], dim=1)
            #print(X.size())
            margin = int(X.size(1) - self.nb_samps)
            #print(self.nb_samps)
            #print(margin)
            st_idx = np.random.randint(0, margin) if margin != 0 else 0
            X = X[:, st_idx:st_idx+self.nb_samps]
        '''
        else:
            TTA_mid_idx = X.size(1)//2 - self.nb_samps//2
            l_X = []
            l_X.append(X[:,:self.nb_samps])
            l_X.append(X[:,TTA_mid_idx:TTA_mid_idx+self.nb_samps])
            l_X.append(X[:,-self.nb_samps:])
            X = torch.stack(l_X)
        '''
        X = self.melspec(X)
        if not self.trn and X.size(-1) < 40:
            while X.size(-1) < 40:
                X = torch.cat([X, X], dim=-1)

        X[X==0] = 1e-7
        #print('='*5)
        #print(X)
        X = torch.log(X)    #2020.8.2.
        #print(X)
        #if self.verbose > 3: print('Mel-spec: ', X.size())
        X = self._utt_mvn(X)
        #print(X)
        #print('='*5)
        y = self.y[index].astype(np.float32)
        return X, y

    
    def _pre_emphasis(self, x):
        return x[:,1:] - 0.97 * x[:, :-1] 

    def _utt_mvn(self, x):
        _m = x.mean(dim=-1, keepdim = True)
        _s = x.std(dim=-1, keepdim = True)
        _s[_s<0.001] = 0.001
        return (x - _m) / _s