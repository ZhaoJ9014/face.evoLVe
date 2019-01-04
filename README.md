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
* [Usage](#Usage)
* [Face Alignment](#Face-Alignment)
* [Data Processing](#Data-Processing)
* [Training and Validation](#Training-and-Validation)
* [Data Zoo](#Data-Zoo)
* [Model Zoo](#Model-Zoo)
* [Citation](#Citation)

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

### Usage

* Clone the repo: `git clone https://github.com/ZhaoJ9014/face.evoLVe.PyTorch.git`.
* `mkdir data checkpoint log` at appropriate directory to store your train/val/test data, checkpoints and training logs.
* Ensure each database folder has the following structure:
```
./data/db_name/
        -> id1/
            -> 1.jpg
            -> ...
        -> id2/
            -> 1.jpg
            -> ...
        -> ...
            -> ...
            -> ...
```
* Run the codes of the corresponding section for specific purpose.

### Face Alignment

TO DO

### Data Processing

TO DO

### Training and Validation

TO DO

### Data Zoo

TO DO

### Model Zoo

TO DO

### Citation
- Please consult and consider citing the following papers:


      @article{zhao2018look,
      title={Look Across Elapse: Disentangled Representation Learning and Photorealistic Cross-Age Face Synthesis for Age-Invariant Face Recognition},
      author={Zhao, Jian and Cheng, Yu and Cheng, Yi and Yang, Yang and Lan, Haochong and Zhao, Fang and Xiong, Lin and Xu, Yan and Li, Jianshu and Pranata, Sugiri and others},
      journal={arXiv preprint arXiv:1809.00338},
      year={2018}
      }
      
      
      @article{zhao20183d,
      title={3D-Aided Dual-Agent GANs for Unconstrained Face Recognition},
      author={Zhao, Jian and Xiong, Lin and Li, Jianshu and Xing, Junliang and Yan, Shuicheng and Feng, Jiashi},
      journal={T-PAMI},
      year={2018}
      }
      
      
      @inproceedings{zhao2017dual,
      title={Dual-agent gans for photorealistic and identity preserving profile face synthesis},
      author={Zhao, Jian and Xiong, Lin and Jayashree, Panasonic Karlekar and Li, Jianshu and Zhao, Fang and Wang, Zhecan and Pranata,           Panasonic Sugiri and Shen, Panasonic Shengmei and Yan, Shuicheng and Feng, Jiashi},
      booktitle={NIPS},
      pages={66--76},
      year={2017}
      }
      
      
      @inproceedings{zhao2018towards,
      title={Towards Pose Invariant Face Recognition in the Wild},
      author={Zhao, Jian and Cheng, Yu and Xu, Yan and Xiong, Lin and Li, Jianshu and Zhao, Fang and Jayashree, Karlekar and Pranata,         Sugiri and Shen, Shengmei and Xing, Junliang and others},
      booktitle={CVPR},
      pages={2207--2216},
      year={2018}
      }
      
      
      @inproceedings{zhao3d,
      title={3D-Aided Deep Pose-Invariant Face Recognition},
      author={Zhao, Jian and Xiong, Lin and Cheng, Yu and Cheng, Yi and Li, Jianshu and Zhou, Li and Xu, Yan and Karlekar, Jayashree and       Pranata, Sugiri and Shen, Shengmei and others},
      booktitle={IJCAI},
      pages={1184--1190},
      year={2018}
      }


      @inproceedings{cheng2017know,
      title={Know you at one glance: A compact vector representation for low-shot learning},
      author={Cheng, Yu and Zhao, Jian and Wang, Zhecan and Xu, Yan and Jayashree, Karlekar and Shen, Shengmei and Feng, Jiashi},
      booktitle={ICCVW},
      pages={1924--1932},
      year={2017}
      }
