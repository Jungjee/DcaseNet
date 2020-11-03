#!/bin/bash

gpu_id=0 
nb_worker=24
phase=0 #set phase to 1 if you wand to fine-tune using the provided pre-trained model
joint_model_name="joint_trn_DcaseNet"

if [ ${phase} -eq 1]; then
  mv ./weights/DcaseNet_v3_joint* /exp/DNNs/${joint_model_name}/weights/
fi

if [ ${phase} -le 0 ]; then
  ######
  # Joint train ASC, TAG, and SED tasks using DcaseNet-v3 architecture
  CUDA_VISIBLE_DEVICES=${gpu_id} python main.py -verbose 0 \
    -nb_iter_per_epoch 500 \
    -task ASC SED TAG \
    -name ${joint_model_name} -model_scp DcaseNet -model_name get_DcaseNet_v3 \
    -nb_worker ${nb_worker} \
    -wd 0.000 -epoch 160 \
    -bs_ASC 32 -bs_SED 24 -bs_TAG 32 \
    -optimizer adam -amsgrad 0 \
    -do_lr_decay 1 -lr_decay cosine -lrdec_t0 80 \
    -do_mixup 1 -mixup_start 5 \
    -loss_weight_ASC 1 -loss_weight_SED 1 -loss_weight_TAG 1 
fi

if [ ${phase} -le 1 ]; then
  #####
  # fine-tune for ASC
  exp_name="fine-tune_ASC"
  CUDA_VISIBLE_DEVICES=${gpu_id} python main.py -verbose 0 \
    -nb_iter_per_epoch 500 \
    -task ASC \
    -name ${exp_name} -model_scp DcaseNet -model_name get_DcaseNet_v3 \
    -nb_worker ${nb_worker} \
    -wd 0.000 -epoch 160 \
    -bs_ASC 32 -bs_SED 24 -bs_TAG 32 \
    -optimizer adam -amsgrad 0 \
    -do_lr_decay 1 -lr_decay cosine -lrdec_t0 80 \
    -loss_weight_ASC 1 -loss_weight_SED 1 -loss_weight_TAG 1 \
    -do_mixup 1 -mixup_start 5 \
    -dir_model_weight /exp/DNNs/${joint_model_name}/weights/best_ASC.pt
fi

if [ ${phase} -le 1 ]; then
  #####
  # fine-tune for TAG
  exp_name="fine-tune_TAG"
  CUDA_VISIBLE_DEVICES=${gpu_id} python main.py -verbose 0 \
    -nb_iter_per_epoch 500 \
    -task TAG \
    -name ${exp_name} -model_scp DcaseNet -model_name get_DcaseNet_v3 \
    -nb_worker ${nb_worker} \
    -wd 0.000 -epoch 160 \
    -bs_ASC 32 -bs_SED 24 -bs_TAG 32 \
    -optimizer adam -amsgrad 0 \
    -do_lr_decay 1 -lr_decay cosine -lrdec_t0 80 \
    -loss_weight_ASC 1 -loss_weight_SED 1 -loss_weight_TAG 1 \
    -do_mixup 1 -mixup_start 5 \
    -dir_model_weight /exp/DNNs/${joint_model_name}/weights/best_TAG.pt
fi

if [ ${phase} -le 1 ]; then
  #####
  # fine-tune for SED
  exp_name="fine-tune_SED"
  CUDA_VISIBLE_DEVICES=${gpu_id} python main.py -verbose 0 \
    -nb_iter_per_epoch 500 \
    -task TAG \
    -name ${exp_name} -model_scp DcaseNet -model_name get_DcaseNet_v3 \
    -nb_worker ${nb_worker} \
    -wd 0.000 -epoch 160 \
    -bs_ASC 32 -bs_SED 24 -bs_TAG 32 \
    -optimizer adam -amsgrad 0 \
    -do_lr_decay 1 -lr_decay cosine -lrdec_t0 80 \
    -loss_weight_ASC 1 -loss_weight_SED 1 -loss_weight_TAG 1 \
    -do_mixup 1 -mixup_start 5 \
    -dir_model_weight /exp/DNNs/${joint_model_name}/weights/best_SED.pt
fi