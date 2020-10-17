import torch
import numpy as np
import soundfile as sf
import torchaudio as ta
import h5py
import pandas as pd

from torch.utils import data
from tqdm import tqdm
from utils import *

def extract_log_mel_spec_sed(lines, args):
    # path setting
    base_dir = args.DB_SED+args.wav_SED
    meta_dir = args.DB_SED+'metadata_dev/'
    h5_dir = args.DB_SED+'log_mel_spec_label/'
    if not os.path.exists(h5_dir): os.makedirs(h5_dir)

    # init data processor
    log_mel_spec_extractor = ta.transforms.MelSpectrogram(
        24000,
        n_fft=2048,
        win_length=int(24000 * 0.001 * 40),
        hop_length=int(24000 * 0.001 * 20),
        window_fn=torch.hamming_window,
        n_mels=128
    )

    # extract log mel spectrogram
    for l in tqdm(lines):
        if args.verbose>=2: print('l:', l)
        fn = os.path.splitext(l)[0]
        X, samp_rate = ta.load(base_dir+fn+'.wav', normalization = True)   #X.size : (1, 441000)
        if args.verbose>=2: print('init load:', X.size())
        try:
            assert samp_rate == 24000
        except:
            print('samp_rate:', samp_rate)
        X = _pre_emphasis(X)
        X = log_mel_spec_extractor(X)
        X = torch.log(X)
        if args.verbose>=2: print('after melspec', X.size())
        X = _utt_mvn(X).numpy()
        if args.verbose>=2: print('after melspec', X.shape)
        if args.verbose>=2: print('dtype:', X[0][0][0].dtype)

        #load labels
        d_label = _load_output_format_file(meta_dir+fn+'.csv')
        y = _get_labels_for_file(d_label)


        for idx, X_mono in enumerate(X):
            with h5py.File(h5_dir+fn+':ch{}.h5'.format(idx+1), 'w') as hf:
                hf.create_dataset('log_mel_spec', data = X_mono, dtype = np.float32)
                hf.create_dataset('label_sed', data = y, dtype = np.float32)
    return True

def _load_output_format_file(_output_format_file):
    """
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        if len(_words) == 5: #read polar coordinates format, we ignore the track count 
            _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4])])
        elif len(_words) == 6: # read Cartesian coordinates format, we ignore the track count
            _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4]), float(_words[5])])
    _fid.close()
    return _output_dict

def _get_labels_for_file(_desc_file):
    """
    Reads description file and returns classification based SED labels and regression based DOA labels

    :param _desc_file: metadata description file
    :return: label_mat: labels of the format [sed_label, doa_label],
    where sed_label is of dimension [nb_frames, nb_classes] which is 1 for active sound event else zero
    where doa_labels is of dimension [nb_frames, 3*nb_classes], nb_classes each for x, y, z axis,
    """

    se_label = np.zeros((600, 14))
    for frame_ind, active_event_list in _desc_file.items():
        for active_event in active_event_list:
            se_label[frame_ind, active_event[0]] = 1
    return se_label

def _pre_emphasis(x): return x[:,1:] - 0.97 * x[:, :-1]

def _utt_mvn(x):
    _m = x.mean(dim=-1, keepdim = True)
    _s = x.std(dim=-1, keepdim = True)
    _s[_s<0.001] = 0.001
    return (x - _m) / _s
