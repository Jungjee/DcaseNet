gpu_id=0 
nb_worker=24
save_dir = [SET_YOUR_PATH, ex)/result/DCASENET/]
db_tag = [SET_YOUR_PATH, ex)/data/DCASE2019_Task2/]
db_asc = [SET_YOUR_PATH, ex)/data/DCASE2020_Task1A/]
db_sed = [SET_YOUR_PATH, ex)/data/DCASE2020_Task3/]

#########
# EvalASC 
#########
CUDA_VISIBLE_DEVICES=${gpu_id} python main.py -verbose 0 \
  -nb_iter_per_epoch 500 \
  -task ASC\
  -name EvalASC \
  -model_scp DcaseNet \
  -model_name get_DcaseNet_v3 \
  -nb_worker ${nb_worker} \
  -wd 0.0000\
  -epoch 160 \
  -bs_ASC 32 \
  -bs_SED 24 \
  -bs_TAG 32 \
  -optimizer adam \
  -amsgrad 0 \
  -do_lr_decay 1 \
  -lr_decay cosine \
  -lrdec_t0 80 \
  -do_mixup 1 \
  -mixup_start 5 \
  -loss_weight_ASC 1 \
  -loss_weight_SED 1 \
  -loss_weight_TAG 1 \
  -save_dir ${save_dir} \
  -DB_TAG ${db_tag} \
  -DB_ASC ${db_asc} \
  -DB_SED ${db_sed}  

#########
# EvalTAG 
#########
CUDA_VISIBLE_DEVICES=${gpu_id} python main.py -verbose 0 \
  -nb_iter_per_epoch 500 \
  -task TAG\
  -name EvalTAG \
  -model_scp DcaseNet \
  -model_name get_DcaseNet_v3 \
  -nb_worker ${nb_worker} \
  -wd 0.0000\
  -epoch 160 \
  -bs_ASC 32 \
  -bs_SED 24 \
  -bs_TAG 32 \
  -optimizer adam \
  -amsgrad 0 \
  -do_lr_decay 1 \
  -lr_decay cosine \
  -lrdec_t0 80 \
  -do_mixup 1 \
  -mixup_start 5 \
  -loss_weight_ASC 1 \
  -loss_weight_SED 1 \
  -loss_weight_TAG 1 \
  -save_dir ${save_dir} \
  -DB_TAG ${db_tag} \
  -DB_ASC ${db_asc} \
  -DB_SED ${db_sed}  

#########
# EvalSED 
#########
CUDA_VISIBLE_DEVICES=${gpu_id} python main.py -verbose 0 \
  -nb_iter_per_epoch 500 \
  -task SED\
  -name EvalSED \
  -model_scp DcaseNet \
  -model_name get_DcaseNet_v3 \
  -nb_worker ${nb_worker} \
  -wd 0.0000\
  -epoch 160 \
  -bs_ASC 32 \
  -bs_SED 24 \
  -bs_TAG 32 \
  -optimizer adam \
  -amsgrad 0 \
  -do_lr_decay 1 \
  -lr_decay cosine \
  -lrdec_t0 80 \
  -do_mixup 1 \
  -mixup_start 5 \
  -loss_weight_ASC 1 \
  -loss_weight_SED 1 \
  -loss_weight_TAG 1 \
  -save_dir ${save_dir} \
  -DB_TAG ${db_tag} \
  -DB_ASC ${db_asc} \
  -DB_SED ${db_sed}  