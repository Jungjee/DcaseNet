#!/bin/bash
# (Optional)

#####
# Use Nvidia GPU cloud with nvidia-docker
# to reproduce experiments
local_dir=./
save_dir=[DIR_TO_SAVE_AND_LOG_EXPERIMENTS]
DB_dir=[DIR_OF_YOUR_DOWNLOADED_DBS]
docker_image=nvcr.io/nvidia/pytorch:19.10-py3 #when conducting experiments for DcaseNet, we 1. launched this official image then 2. manually updated torch version to "1.4.0".

sudo nvidia-docker run -it --rm --ipc=host --shm-size 20G

