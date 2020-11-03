#!/bin/bash

gpu_id=0 
nb_worker=24

exp_name="EvalASC"
CUDA_VISIBLE_DEVICES=${gpu_id} python main.py -verbose 0 \
  -task ASC \
  -name ${exp_name} -model_scp DcaseNet -model_name get_DcaseNet_v3 \
  -nb_worker ${nb_worker} \
  -bs_ASC 32 -bs_SED 24 -bs_TAG 32 \
  -dir_model_weight ./weights/DcaseNet_v3_finetune_best_ASC.pt

exp_name="EvalTAG"
CUDA_VISIBLE_DEVICES=${gpu_id} python main.py -verbose 0 \
  -task TAG \
  -name ${exp_name} -model_scp DcaseNet -model_name get_DcaseNet_v3 \
  -nb_worker ${nb_worker} \
  -bs_ASC 32 -bs_SED 24 -bs_TAG 32 \
  -dir_model_weight ./weights/DcaseNet_v3_finetune_best_TAG.pt

exp_name="EvalSED"
CUDA_VISIBLE_DEVICES=${gpu_id} python main.py -verbose 0 \
  -task SED \
  -name ${exp_name} -model_scp DcaseNet -model_name get_DcaseNet_v3 \
  -nb_worker ${nb_worker} \
  -bs_ASC 32 -bs_SED 24 -bs_TAG 32 \
  -dir_model_weight ./weights/DcaseNet_v3_finetune_best_SED.pt