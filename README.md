# face.evoLVe: Face Recognition Library based on PyTorch

* Evolve to be more comprehensive, effective and efficient for face related analytics \& applications!

* About the name:
  * "face" means this repo is dedicated for face related analytics \& applications.
  * "evolve" means unleash your greatness to be better and better. "LV" are capitalized to acknowledge the nurturing of Learning and Vision ([LV](http://www.lv-nus.org)) group, Nation University of Singapore (NUS).
  
 * This work was done during Jian Zhao served as a short-term "Texpert" Research Scientist at FiT DeepSea Lab of Tencent, Shenzhen China.

|Author|Jian Zhao|
|---|---
|Homepage|https://zhaoj9014.github.io

## License

The code of face.evoLVe is released under the MIT License.

****
## Contents
* [Introduction](#Introduction)
* [Pre-Requisites](#Pre-Requisites)
* [Face Alignment](#Face-Alignment)

## face.evoLVe for Face Recognition

### Introduction

This repo provides a comprehensive face recognition library for face related analytics \& applications, including face alignment (detection, landmark localization, affine transformation, *etc.*), data processing (*e.g.*, augmentation, data balancing, normalization, *etc.*), various backbones (*e.g.*, ResNet, IR-SE, ResNeXt, SE-ResNeXt, DenseNet, LightCNN, MobileNet, ShuffleNet, DPN, *etc.*), various losses (*e.g.*, Softmax, Focal, Center, SphereFace, CosineFace, AmSoftmax, ArcFace, Triplet, *etc.*) and bags of tricks for improving performance (*e.g.*, training refinements, model tweaks, knowledge distillation, *etc.*).

All data before \& after alignment, source codes and trained models are provided.

This repo can help researchers/engineers develop deep face recognition models and algorithms quickly for practical use and deployment.


<img src="https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/disp/Fig1.png" width="500px"/>

### Pre-Requisites

* Linux or macOS
* [Python 3.7](https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh) (for training \& validation) and [Python 2.7](https://repo.continuum.io/archive/Anaconda2-2018.12-Linux-x86_64.sh) (for visualization w/ tensorboardx)
* PyTorch 1.0 (for traininig \& validation, install w/ `pip install torch torchvision`)
* MXNet 1.3.1 (optinal, for data processing, install w/ `pip install mxnet-cu90`)
* TensorFlow 1.12 (optinal, for visualization, install w/ `pip install tensorflow-gpu`)
* tensorboardX 1.6 (optinal, for visualization, install w/ `pip install tensorboardX`)
* OpenCV 3.4.5 (install w/ `pip install opencv-python`)
* bcolz 1.2.0 (install w/ `pip install bcolz`)

While not required, for optimal performance it is **highly** recommended to run the code using a CUDA enabled GPU. We used 4 NVIDIA Tesla P40 in parallel.

### Face Alignment
