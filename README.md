# Overview
This GitHub project includes PyTorch implementation for reproducing experiments and DNN models used in the paper
[DcaseNet: An integrated pretrained deep neural network for detecting and classifying acoustic scenes and events]( https://arxiv.org/abs/2009.09642 ), accepted for presentation at IEEE ICASSP 2021.

DcaseNet is a DNN which jointly performs acoustic scene classification (ASC), audio tagging (TAG), and sound event detection (SED) simultaneously.
It adopts a two-phase training. In the first phase, joint training of three tasks is performed. Then, the model is fine-tuned for each task. 


# Usage

## Environment Setting
We used Nvidia GPU Cloud for conducting our experiments. The training was done using one Nvidia Titan RTX GPU. Our settings are available at [launch_nvidia-gpu-cloud.sh]( https://github.com/Jungjee/DcaseNet/blob/master/launch_nvidia-gpu-cloud.sh )

## Train

1. Download three datasets: DCASE 2020 challenge Task 1-a, DCASE 2019 challenge Task 2, and DCASE 2020 challenge Task 3 and configure directories.
2. (selectively) Enter virtual environment using NGC. 
3. Set parameters in [train.sh](https://github.com/Jungjee/DcaseNet/blob/master/train.sh)
4. run train.sh

If you prefer to use pre-trained joint DcaseNet and fine-tune only, remove 'Joint' experiment on train.sh and copy [Joint weights](https://github.com/Jungjee/DcaseNet/tree/master/weights/Joint) into your 'save_dir'

##  Evaluation

1. Download three datasets: DCASE 2020 challenge Task 1-a, DCASE 2019 challenge Task 2, and DCASE 2020 challenge Task 3 and configure directories.
2. Set parameters in [evaluate_trained_models.sh](https://github.com/Jungjee/DcaseNet/blob/master/evaluate_trained_models.sh)
3. Run evaluate_trained_models.sh

## Windows
There's a simple GUI program in [DCASENetShellScriptBuilder](https://github.com/Jungjee/DcaseNet/tree/develop/DCASENetShellScriptBuilder) that generates a script that one can run on Windows OS.
After configuring a few checkboxes and setting directories for datasets, the generated script trains and evaluates.
This program is provided by yeongsoo, and no further maintenance will be done. 

The program has three rows: 
  (i)   On which tasks will the user conduct joint training
        (By checking none, it will use pretrained DcaseNet using all three tasks)
  (ii)  On which tasks to perform fine-tuning 
        (checking more than one task will train separate DcaseNets for each fine-tune task)
        (recommended to should check at least on task)
  (iii) On which tasks to perform the evaluation
        (recommended to be the same with upper row)

Below, there are text boxes where one can set directories of the downloaded datasets and save trained models. 
Note that when setting dataset directories, the code in this repo expects the folder that comes out after unzipping it. 

![DCASENetShellScriptBuilder](https://github.com/Jungjee/DcaseNet/blob/develop/DCASENetShellScriptBuilder/image.PNG?raw=true)

##### Email jeewon.leo.jung@gmail.com for other details :-).

# BibTex

This repository provides the code for reproducing the below paper. 
```
@inproceedings{jung2021dcasenet,
  title={DCASENet: An integrated pretrained deep neural network for detecting and classifying acoustic scenes and events},
  author={Jung, Jee-weon and Shim, Hye-jin and Kim, Ju-ho and Yu, Ha-Jin},
  booktitle={Proc. ICASSP},
  pages={621--625},
  year={2021},
  organization={IEEE}
}
```

# TO-DO

# Log
- 2020.09.24. : Initial commit
- 2020.10.18. : Overall validation & refactoring (thanks to yeongsoo)
- 2020.11.04. : Added filetrees & Refactoring finish
