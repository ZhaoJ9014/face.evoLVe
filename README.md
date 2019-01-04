# face.evoLVe: High-Performance Face Recognition Library based on PyTorch

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
* [Achievement](#Achievement)
* [Acknowledgement](#Acknowledgement)
* [Citation](#Citation)

## face.evoLVe for High-Performance Face Recognition

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
* Refer to the codes of corresponding sections for specific purposes.

### Face Alignment

<img src="https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/disp/Fig2.png" width="900px"/>

* This sub-section is based on the work of [MTCNN](https://arxiv.org/pdf/1604.02878.pdf).
* Folder: ```./align```
* Face detection, landmark localization APIs and visualization toy example:
``` 
from PIL import Image
from detector import detect_faces
from visualization_utils import show_results
       
img = Image.open('some_img.jpg') # modify the image path to yours
bounding_boxes, landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
show_results(img, bounding_boxes, landmarks) # visualize the results
``` 
* Face alignment API (perform face detection, landmark localization and alignment with affine transformations to a whole database folder and store aligned results to a new folder with the same structure): ```face_align.py```. 
* Keynotes for customed use: 1) modify the ```source_root``` and ```dest_root``` at Line 8 \& 9 of ```face_align.py``` to your own dirs, respectively; 2) modify the ```crop_size``` at Line 29 of ```face_align.py``` according to your case; 3) if you changed the ```crop_size``` at Line 29 of ```face_align.py```, you should also adjust the ```REFERENCE_FACIAL_POINTS``` values between Line 7--11 of ```align_trans.py``` accordingly; 4) you can also pass customed ```min_face_size```, ```thresholds``` and ```nms_thresholds``` to the ```detect_faces``` function of ```detector.py``` according to practical requirements.

### Data Processing

TO DO

### Training and Validation

TO DO

### Data Zoo

TO DO

### Model Zoo

TO DO

### Achievement

* 2017 No.1 on ICCV 2017 MS-Celeb-1M Large-Scale Face Recognition [Hard Set](https://www.msceleb.org/leaderboard/iccvworkshop-c1)/[Random Set](https://www.msceleb.org/leaderboard/iccvworkshop-c1)/[Low-Shot Learning](https://www.msceleb.org/leaderboard/c2) Challenges. [WeChat News](http://mp.weixin.qq.com/s/-G94Mj-8972i2HtEcIZDpA), [NUS ECE News](http://ece.nus.edu.sg/drupal/?q=node/215), [NUS ECE Poster](https://zhaoj9014.github.io/pub/ECE_Poster.jpeg), [Award Certificate for Track-1](https://zhaoj9014.github.io/pub/MS-Track1.jpeg), [Award Certificate for Track-2](https://zhaoj9014.github.io/pub/MS-Track2.jpeg), [Award Ceremony](https://zhaoj9014.github.io/pub/MS-Awards.jpeg).

* 2017 No.1 on National Institute of Standards and Technology (NIST) IARPA Janus Benchmark A (IJB-A) Unconstrained Face [Verification](https://zhaoj9014.github.io/pub/IJBA_11_report.pdf) challenge and [Identification](https://zhaoj9014.github.io/pub/IJBA_1N_report.pdf) challenge. [WeChat News](https://mp.weixin.qq.com/s/s9H_OXX-CCakrTAQUFDm8g).

* State-of-the-art performance on 

    * MS-Celeb-1M (Challenge1 Hard Set Coverage@P=0.95: 79.10%; Challenge1 Random Set Coverage@P=0.95: 87.50%; Challenge2 Development Set Coverage@P=0.99: 100.00%; Challenge2 Base Set Top 1 Accuracy: 99.74%; Challenge2 Novel Set Coverage@P=0.99: 99.01%).

    * IJB-A (1:1 Veification TAR@FAR=0.1: 99.6%±0.1%; 1:1 Veification TAR@FAR=0.01: 99.1%±0.2%; 1:1 Veification TAR@FAR=0.001: 97.9%±0.4%; 1:N Identification FNIR@FPIR=0.1: 1.3%±0.3%; 1:N Identification FNIR@FPIR=0.01: 5.4%±4.7%; 1:N Identification Rank1 Accuracy: 99.2%±0.1%; 1:N Identification Rank5 Accuracy: 99.7%±0.1%; 1:N Identification Rank10 Accuracy: 99.8%±0.1%).

    * Labeled Faces in the Wild (LFW) (Accuracy: 99.85%±0.217%).

    * Celebrities in Frontal-Profile (CFP) (Frontal-Profile Accuracy: 96.01%±0.84%; Frontal-Profile EER: 4.43%±1.04%; Frontal-Profile AUC: 99.00%±0.35%; Frontal-Frontal Accuracy: 99.64%±0.25%; Frontal-Frontal EER: 0.54%±0.37%; Frontal-Frontal AUC: 99.98%±0.03%).

    * CMU Multi-PIE (Rank1 Accuracy Setting-1 under ±90°: 76.12%; Rank1 Accuracy Setting-2 under ±90°: 86.73%).

### Acknowledgement

* This repo is inspired by [InsightFace.MXNet](https://github.com/deepinsight/insightface), [InsightFace.PyTorch](https://github.com/TreB1eN/InsightFace_Pytorch), [ArcFace.PyTorch](https://github.com/ronghuaiyang/arcface-pytorch), [MTCNN.MXNet](https://github.com/pangyupo/mxnet_mtcnn_face_detection) and [PretrainedModels.PyTorch](https://github.com/Cadene/pretrained-models.pytorch).
* The work of Jian Zhao was partially supported by China Scholarship Council (CSC) grant 201503170248.

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
