# Adaptive feature recombination and recalibration for semantic segmentation

## Description

This repository contains the implementation of recombination and recalibration (SegSE) blocks. Also, it contains the subject's ID used for Training, Validation, and Test in BRATS 2017.

More details can be found in our paper [[Extended journal paper](https://ieeexplore.ieee.org/abstract/document/8718639) | [MICCAI paper](https://link.springer.com/chapter/10.1007/978-3-030-00931-1_81) | [MICCAI ArXiv preprint](https://arxiv.org/pdf/1806.02318.pdf)].


## Contents

```recombination_recalibration.py``` this module contains the implementation of the proposed blocks: recombination, recalibration (SegSE), and recombination and recalibration.

```brats2017_subjects_sets.py``` we divided BRATS 2017 Training set into the following subsets: Training (60%), Validation (20%), and Test (20%). This script contains the subject's ID of each subset (it can also be used for BRATS 2018, since the provided Training set is equal).

This can be used to compare directly with us, using the Test set (results in the paper, Table 1).

```whole_tumor_binary_segmentation``` this directory contains the binary whole tumor segmentations from the WT-FCN. They are used in our hierarchical segmentation approach. 


## Citation

If you found this code useful, please, cite at least the first of the following papers:

- Pereira, Sérgio, et al. "Adaptive feature recombination and recalibration for semantic segmentation with Fully Convolutional Networks." IEEE transactions on medical imaging (2019).

- Sérgio Pereira, Victor Alves, and Carlos A. Silva, "Adaptive feature recombination and recalibration for semantic segmentation: application to brain tumor segmentation in MRI", Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2018.


## Abstract

Fully Convolutional Networks have been achieving remarkable results in image semantic segmentation, while being efficient. Such efficiency results from the capability of segmenting several voxels in a single forward pass. So, there is a direct spatial correspondence between a unit in a feature map and the voxel in the same location. In a convolutional layer, the kernel spans over all channels and extracts information from them. We observe that linear recombination of feature maps by increasing the number of channels followed by compression may enhance their discriminative power. Moreover, not all feature maps have the same relevance for the classes being predicted. In order to learn the inter-channel relationships and recalibrate the channels to suppress the less relevant ones, Squeeze and Excitation blocks were proposed in the context of image clas sification with Convolutional Neural Networks. However, this is not well adapted for segmentation with Fully Convolutional Networks since they segment several objects simultaneously, hence a feature map may contain relevant information only in some locations. In this paper, we propose recombination of features and a spatially adaptive recalibration block that is adapted for semantic segmentation with Fully Convolutional — Networks the SegSE block. Feature maps are recalibrated by considering the cross-channel information together with spatial relevance. Experimental results indicate that Recombination and Recalibration improve the results of a competitive baseline, and generalize across three different problems: brain tumor segmentation, stroke penumbra estimation, and ischemic stroke lesion outcome prediction. The obtained results are competitive or outperform the state of the art in the three applications.


## Contact
For information related with the paper, please feel free to contact me via e-mail: id5692@alunos.uminho.pt
