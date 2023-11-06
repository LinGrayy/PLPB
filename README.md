# PLPB
Datasets &amp; Code for the WACV 2024 paper 'Robust Source-Free Domain Adaptation for Fundus Image Segmentation' [paper](http://arxiv.org/abs/2310.16665) 


In this study, we propose a two-stage training strategy for robust domain adaptation. In the source training stage, we utilize adversarial sample augmentation to enhance the robustness and generalization capability of the source model. And in the target training stage, we propose a novel robust pseudo-label and pseudo-boundary (PLPB) method, which effectively utilizes unlabeled target data to generate pseudo labels and pseudo boundaries that enable model self-adaptation without requiring source data. Extensive experimental results on cross-domain fundus image segmentation confirm the effectiveness and versatility of our method.

# Acknowledgement
The code for source domain training is modified from [BEAL] (https://github.com/emma-sjwang/BEAL).
