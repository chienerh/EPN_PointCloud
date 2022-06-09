# EPN-NetVLAD
This repository utilize Equivariant Point Network (EPN) repository [EPN_PointCloud](https://github.com/nintendops/EPN_PointCloud) to perform place recogntion task.


## Set Up
See [docker](docker) folder for how to use docker image and build docker container.
The module and additional dependencies can be installed with 
```
cd vgtk
python setup.py install
```

## Experiments

### Datasets
This repository is tested with Oxford Robocar benchmark created by PointNetVLAD, and can be downloaded [here](https://drive.google.com/drive/folders/1Wn1Lvvk0oAkwOUwR0R6apbrekdXAUg7D). 

### Pretrained Model

A pretrained weight with `atten_epn_netvlad_select` model can be downloaded using this [link](https://drive.google.com/file/d/1VuBSSi5CsXB73iYtl8Fn3nNBs9i90iNA/view?usp=sharing)

### Training
Set the cofigurations for training in [config.py](config.py) file, then use the following command to train the model

```
CUDA_VISIBLE_DEVICES=0 python run_oxford.py experiment -d PATH_TO_OXFORD
```

### Evaluation

The following lines can be used for the evaluation of each experiment

```
CUDA_VISIBLE_DEVICES=0 python evaluate_place_recognition.py experiment -d PATH_TO_OXFORD 
```