# Robust Source-Free Domain Adaptation for Fundus Image Segmentation
Datasets & Code for the WACV 2024 paper 'Robust Source-Free Domain Adaptation for Fundus Image Segmentation' [Paper](https://arxiv.org/abs/2310.16665). 

In this study, we propose a two-stage training strategy for robust domain adaptation. In the source training stage, we utilize adversarial sample augmentation to enhance the robustness and generalization capability of the source model. And in the target training stage, we propose a novel robust pseudo-label and pseudo-boundary (PLPB) method, which effectively utilizes unlabeled target data to generate pseudo labels and pseudo boundaries that enable model self-adaptation without requiring source data. Extensive experimental results on cross-domain fundus image segmentation confirm the effectiveness and versatility of our method.

## Paper
[Robust Source-Free Domain Adaptation for Fundus Image Segmentation](https://arxiv.org/abs/2310.16665) WACV 2024
![image](https://github.com/LinGrayy/PLPB/assets/49065934/84cfe4bd-d584-4742-8f4d-311bd2929928)

## Pytorch implementation of our method PLPB.

## Installation
* Install Pytorch 0.4.1 and CUDA 9.0
* Clone this repo
```
git clone https://github.com/LinGrayy/PLPB
cd PLPB
```

## Train
* Download datasets from [here](https://drive.google.com/file/d/1B7ArHRBjt2Dx29a3A6X_lGhD0vDVr3sy/view).
* Download the source domain model from `./logs/source/robust-checkpoint.pth.tar` as the robust model
 or specify the data path in `./train_source.py` and then train `./train_source.py`.
* Save the source domain model into folder `./logs/source`.

* specify the model path and data path in `./generate_pseudo_label.py` and then train `./generate_pseudo_label.py`, obtaining standard pseudo label.
* Save generated pseudo labels into folder `./generate_pseudo/mask`.
* specify the model path and data path in `./generate_pseudo_bound.py` and then train `./generate_pseudo_bound.py`, obtaining standard pseudo boundary.
* Save generated pseudo labels into the folder `./generate_pseudo/bound`.

* Run `./train_target.py` to start the target domain training process.

## Acknowledgement
The code for source domain training is modified from [BEAL](https://github.com/emma-sjwang/BEAL) and [DPL](https://github.com/cchen-cc/SFDA-DPL). 
