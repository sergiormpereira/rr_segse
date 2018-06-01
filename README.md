# Adaptive feature recombination and recalibration for semantic segmentation

## Description

This repository contains the subject's ID used for Training, Validation, and Test.

More details can be found in our paper.


## Contents

```brats2017_subjects_sets.py``` we divided BRATS 2017 Training set into the following subsets: Training (60%), Validation (20%), and Test (20%). This script contains the subject's ID of each subset (can be used for BRATS 2018, too).


## Citation

If you found this code useful, please, cite our paper:

Sérgio Pereira, Victor Alves, and Carlos A. Silva, "Adaptive feature recombination and recalibration for semantic segmentation: application to brain tumor segmentation in MRI", Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2018.


## Abstract

Convolutional neural networks (CNNs) have been successfully used for brain tumor segmentation, specifically, fully convolutional networks (FCNs). FCNs can segment a set of voxels at once, having a direct spatial correspondence between units in feature maps (FMs) at a given location and the corresponding classified voxels. In convolutional layers, FMs are merged to create new FMs, so, channel combination is crucial. However, not all FMs have the same relevance for a given class. Recently, in classification problems, Squeeze-and-Excitation (SE) blocks have been proposed to re-calibrate FMs as a whole, and suppress the less informative ones. However, this is not optimal in FCN due to the spatial correspondence between units and voxels. In this article, we propose feature recombination through linear expansion and compression to create more complex features for semantic segmentation. Additionally, we propose a segmentation SE (SegSE) block for feature recalibration that collects contextual information, while maintaining the spatial meaning. Finally, we evaluate the proposed methods in brain tumor segmentation, using publicly available data.


## Requirements
The code was tested (in Linux Mint 18) with the following packages:

- ```python 3.6.4 ```
- ```numpy 1.14.0```
- ```keras 2.1.6```
- ```theano 0.9.0```


## Contact
For information related with the paper, please, feel free to contact me via e-mail: id5692@alunos.uminho.pt
