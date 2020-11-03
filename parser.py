import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-task', type = str, nargs='+', default = ['ASC', 'SED', 'TAG'])


    ##########
    ## PATH ##
    ##########
    parser.add_argument('-name', type = str, required = True)
    parser.add_argument('-save_dir', type = str, required = True)
    parser.add_argument('-dir_model_weight', type = str, default = '')

    # ASC
    parser.add_argument('-DB_ASC', type = str, required = True)
    parser.add_argument('-meta_scp', type = str, default = 'TAU-urban-acoustic-scenes-2020-mobile-development/meta.csv')
    parser.add_argument('-fold_trn', type = str, default = 'TAU-urban-acoustic-scenes-2020-mobile-development/evaluation_setup/fold1_train.csv')
    parser.add_argument('-fold_evl', type = str, default = 'TAU-urban-acoustic-scenes-2020-mobile-development/evaluation_setup/fold1_evaluate.csv')
    parser.add_argument('-d_label_ASC', type = str, default = 'TAU-urban-acoustic-scenes-2020-mobile-development/d_label.pk')
    parser.add_argument('-wav_ASC', type = str, default = 'TAU-urban-acoustic-scenes-2020-mobile-development/audio/')

    # SED
    parser.add_argument('-DB_SED', type = str, required = True)
    parser.add_argument('-wav_SED', type = str, default = 'mic_dev/')
    parser.add_argument('-h5_dir_SED', type = str, default = 'log_mel_spec_label/')

    # TAG
    parser.add_argument('-DB_TAG', type = str, required = True)


    ##################
    ## hyper-params ##
    ##################
    # batch
    parser.add_argument('-bs_ASC', type = int, default = 1)
    parser.add_argument('-bs_SED', type = int, default = 1)
    parser.add_argument('-bs_TAG', type = int, default = 1)

    # optimizer
    parser.add_argument('-optimizer', type = str, default = 'sgd')
    parser.add_argument('-lr', type = float, default = 0.001)
    parser.add_argument('-lr_decay', type = str, default = 'cosine')
    parser.add_argument('-lrdec_t0', type = int, default = 80)
    parser.add_argument('-wd', type = float, default = 0.001)
    parser.add_argument('-opt_mom', type = float, default = 0.9)

    # system
    parser.add_argument('-epoch', type = int, default = 80)
    parser.add_argument('-nb_worker', type = int, default = 2)
    parser.add_argument('-nb_iter_per_log', type = int, default = 5)
    parser.add_argument('-nb_iter_per_epoch', type = int, default = 500)

    # data
    parser.add_argument('-nb_mels', type = int, default = 128)
    parser.add_argument('-nb_frame_1s', type = int, default = 50)
    parser.add_argument('-nb_frames_ASC', type = int, default = 250)
    parser.add_argument('-mixup_start', type = int, default = 5)
    parser.add_argument('-mixup_alpha', type = float, default = 0.1)

    # other
    parser.add_argument('-sed_threshold', type = float, default = 0.3)
    parser.add_argument('-verbose', type = int, default = 0)
    parser.add_argument('-epoch_per_task', type = int, default = 2)
    parser.add_argument('-loss_weight_SED', type = float, default = 1.)
    parser.add_argument('-loss_weight_ASC', type = float, default = 1.)
    parser.add_argument('-loss_weight_TAG', type = float, default = 1.)


    ##############
    ## DNN-args ##
    ##############
    parser.add_argument('-model_scp', type = str, required = True)
    parser.add_argument('-model_name', type = str, required = True)
    parser.add_argument('-m_filts_ASC', type = int, default = 256)
    parser.add_argument('-m_blocks_ASC', type = int, default = 3)
    parser.add_argument('-m_strides_ASC', type = int, default = 2)
    parser.add_argument('-m_code_ASC', type = int, default = 128)


    ##########
    ## flag ##
    ##########
    parser.add_argument('-amsgrad', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-nesterov', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-save_best_only', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-reproducible', type = str2bool, nargs='?', const=True, default = False)
    parser.add_argument('-do_lr_decay', type = str2bool, nargs='?', const=True, default = True)
    parser.add_argument('-do_mixup', type = str2bool, nargs='?', const=True, default = False)


    args = parser.parse_args()
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == 'm_':
            if args.verbose > 0: print(k, v)
            args.model[k[2:]] = v
    return args

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')