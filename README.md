# Pytorch implementation of our method PLPB. And supplementary.pdf is also attached in zip.


## Installation
* Install Pytorch 0.4.1 and CUDA 9.0
```

cd PLPB
```

## Train
* Download datasets from [here](https://drive.google.com/file/d/1B7ArHRBjt2Dx29a3A6X_lGhD0vDVr3sy/view).
* Download source domain model from `./logs/source/robust-checkpoint.pth.tar` as the robust model
 or specify the data path in `./train_source.py` and then train `./train_source.py`.
* Save source domain model into folder `./logs/source`.

* specify the model path and data path in `./generate_pseudo_label.py` and then train `./generate_pseudo_label.py`, obtaining standard pseudo label.
* Save generated pseudo labels into folder `./generate_pseudo/mask`.
* specify the model path and data path in `./generate_pseudo_bound.py` and then train `./generate_pseudo_bound.py`, obtaining standard pseudo boundary.
* Save generated pseudo labels into folder `./generate_pseudo/bound`.

* Run `./train_target.py` to start the target domain training process.

## Acknowledgement
The code for source domain training is modified from [BEAL](https://github.com/emma-sjwang/BEAL). 