# Overview
This github project includes PyTorch implementation for reproducing experiments and DNN models used in the paper
[DcaseNet: A Joint Pre-trained Neep Neural Network for Detecting and Classifying Acoustic Scenes and Events]( https://arxiv.org/abs/2009.09642 ).

DcaseNet is a DNN which jointly performs acoustic scene classification (ASC), audio tagging (TAG), and sound event detection (SED) simultaneously.
It adopts a two-phase training. In the first phase, joint training of three tasks are performed. Then, the model is fine-tuned for each task. 


# Usage (not validated yet)

## Environment Setting
We used Nvidia GPU Cloud for conducting our experiments. Training was done using one Nvidia Titan Rtx GPU. Our settings are avalable at [launch_nvidia-gpu-cloud.sh]( https://github.com/Jungjee/DcaseNet/blob/master/launch_nvidia-gpu-cloud.sh )

## Train

1. Download three datasets: DCASE 2020 challenge Task 1-a, DCASE 2019 challenge Task 2, and DCASE 2020 challenge Task 3 and configure directories.
2. (selectively) Enter virtual environment using NGC. 
3. run [train.sh](https://github.com/Jungjee/DcaseNet/blob/master/train.sh)

If you prefer to use pre-trained joint DcaseNet and fine-tune only,
set phase to 1 before executing the script. 


##  Evaluation

1. Download three datasets: DCASE 2020 challenge Task 1-a, DCASE 2019 challenge Task 2, and DCASE 2020 challenge Task 3 and configure directories.
2. Run [evaluate.sh](https://github.com/Jungjee/DcaseNet/blob/master/evaluate.sh)

##### Email jeewon.leo.jung@gmail.com for other details :-).

# BibTex

This reposity provides the code for reproducing below paper. 
```
@article{jung2020dcasenet,
  title={DCASENET: A joint pre-trained deep neural network for detecting and classifying acoustic scenes and events},
  author={Jung, Jee-weon and Kim, Hye-jin and Kim, Ju-ho and Yu, Ha-Jin},
  journal={arXiv preprint arXiv:2009.09642},
  year={2020}
}
```

# TO-DO
1. Check scripts.

# Log
- 2020.09.24. : Initial commit
