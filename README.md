# face.evoLVe: High-Performance Face Recognition Library based on PyTorch

* Evolve to be more comprehensive, effective and efficient for face related analytics \& applications!
* About the name:
  * "face" means this repo is dedicated for face related analytics \& applications.
  * "evolve" means unleash your greatness to be better and better. "LV" are capitalized to acknowledge the nurturing of Learning and Vision ([LV](http://www.lv-nus.org)) group, Nation University of Singapore (NUS).
* This work was done during Jian Zhao served as a short-term "Texpert" Research Scientist at Tencent FiT DeepSea Lab, Shenzhen, China.

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
:information_desk_person:

<img src="https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/disp/Fig1.png" width="500px"/>

* This repo provides a comprehensive face recognition library for face related analytics \& applications, including face alignment (detection, landmark localization, affine transformation, *etc.*), data processing (*e.g.*, augmentation, data balancing, normalization, *etc.*), various backbones (*e.g.*, ResNet, IR-SE, ResNeXt, SE-ResNeXt, DenseNet, LightCNN, MobileNet, ShuffleNet, DPN, *etc.*), various losses (*e.g.*, Softmax, Focal, Center, SphereFace, CosineFace, AmSoftmax, ArcFace, Triplet, *etc.*) and bags of tricks for improving performance (*e.g.*, training refinements, model tweaks, knowledge distillation, *etc.*).
* All data before \& after alignment, source codes and trained models are provided.
* This repo can help researchers/engineers develop deep face recognition models and algorithms quickly for practical use and deployment.

### Pre-Requisites 
:cake:

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
:orange_book:

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
:triangular_ruler:

<img src="https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/disp/Fig2.png" width="900px"/>
<img src="https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/disp/Fig3.png" width="500px"/>

* This section is based on the work of [MTCNN](https://arxiv.org/pdf/1604.02878.pdf).
* Folder: ```./align```
* Face detection, landmark localization APIs and visualization toy example with ipython notebook:
```python 
from PIL import Image
from detector import detect_faces
from visualization_utils import show_results
       
img = Image.open('some_img.jpg') # modify the image path to yours
bounding_boxes, landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
show_results(img, bounding_boxes, landmarks) # visualize the results
``` 
* Face alignment API (perform face detection, landmark localization and alignment with affine transformations on a whole database folder ```source_root``` with the directory structure as demonstrated in Sec. [Usage](#Usage), and store the aligned results to a new folder ```dest_root``` with the same directory structure): 
```
python face_align.py -source_root [source_root] -dest_root [dest_root] -crop_size [crop_size]

# python face_align.py -source_root './data/test' -dest_root './data/test_Aligned' -crop_size 112
```
* For macOS users, there is no need to worry about ```*.DS_Store``` files which may ruin your data, since they will be automatically removed when you run the scripts.
* Keynotes for customed use: 1) specify the arguments of ```source_root```, ```dest_root``` and ```crop_size``` to your own values when you run ```face_align.py```; 2) pass your customed ```min_face_size```, ```thresholds``` and ```nms_thresholds``` values to the ```detect_faces``` function of ```detector.py``` to match your practical requirements.

### Data Processing 
:bar_chart:

* Folder: ```./balance```
* Remove low-shot data API (remove the low-shot classes with less than ```min_num``` samples in the training set ```root``` with the directory structure as demonstrated in Sec. [Usage](#Usage) for data balance and effective model training):
```
python remove_lowshot.py -root [root] -min_num [min_num]

# python remove_lowshot.py -root './data/train' -min_num 10
```
* Keynotes for customed use: specify the arguments of ```root``` and ```min_num``` to your own values when you run ```remove_lowshot.py```.
* We prefer to include other data processing tricks, *e.g.*, augmentation (flip horizontally, scale hue/satuation/brightness with coefficients uniformly drawn from \[0.6,1.4\], add PCA noise with a coefficient sampled from a normal distribution N(0,0.1), *etc.*), weighted random sampling, normalization, *etc.* to the main training script in Sec. [Training and Validation](#Training-and-Validation) to be self-contained.

### Training and Validation 
:coffee:

TO DO

### Data Zoo 
:tiger:

|Database|Version|\#Identity|\#Image|\#Frame|\#Video|Download Link|
|:---:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|Raw|5,749|13,233|-|-|[Google Drive](https://drive.google.com/file/d/1JIgAXYqXrH-RbUvcsB3B6LXctLU9ijBA/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1VzSI_xqiBw-uHKyRbi6zzw)|
|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|Align_Raw|5,749|13,233|-|-|[Google Drive](https://drive.google.com/file/d/11h-QIrhuszY3PzT17Q5eXw8yrewgqX7m/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1Ir8kAcQjBJA6A_pWPL9ozQ)|
|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|Align_112x112|5,749|13,233|-|-| |
|[CALFW](https://arxiv.org/pdf/1708.08197.pdf)|Raw|4,025|12,174|-|-|[Google Drive](https://drive.google.com/file/d/1LcIDIfeZ027tbyUJDbaDt12ZoMVJuoMp/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/17IzL_nGzedup1gcPuob0NQ)|
|[CALFW](https://arxiv.org/pdf/1708.08197.pdf)|Align_112x112|4,025|12,174|-|-| |
|[CPLFW](http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf)|Raw|3,884|11,652|-|-|[Google Drive](https://drive.google.com/file/d/1WipxZ1QXs_Fi6Y5qEFDayEgos3rHDRnS/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1gJuZZcm-2crTrqKI0sa5sA)|
|[CPLFW](http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf)|Align_112x112|3,884|11,652|-|-| |
|[CASIA-WebFace](https://arxiv.org/pdf/1411.7923.pdf)|Raw_v1|10,575|494,414|-|-|[Baidu Drive](https://pan.baidu.com/s/1xh073sKX3IYp9xPm9S6F5Q)|
|[CASIA-WebFace](https://arxiv.org/pdf/1411.7923.pdf)|Raw_v2|10,575|494,414|-|-|[Google Drive](https://drive.google.com/file/d/19R6Svdj5HbUA0y6aJv3P1WkIR5wXeCnO/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1cZqsRxln-JmrA4xevLfjYQ)|
|[CASIA-WebFace](https://arxiv.org/pdf/1411.7923.pdf)|Clean|10,575|455,594|-|-|[Google Drive](https://drive.google.com/file/d/1wJC2aPA4AC0rI-tAL2BFs2M8vfcpX-w6/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1x_VJlG9WV1OdrrJ7ARUZQw)|
|[MS-Celeb-1M](https://arxiv.org/pdf/1607.08221.pdf)|Clean|10,000|5,084,127|-|-|[Google Drive](https://drive.google.com/file/d/18FxgfXgKwuYzY3DmWJXNJuY51TPmC9yH/view?usp=sharing)|
|[MS-Celeb-1M](https://arxiv.org/pdf/1607.08221.pdf)|Align_112x112|85,742|5,822,653|-|-| ||
|[Vggface2](https://arxiv.org/pdf/1710.08092.pdf)|Clean|8,631|3,086,894|-|-|[Google Drive](https://drive.google.com/file/d/1jdZw6ZmB7JRK6RS6QP3YEr2sufJ5ibtO/view?usp=sharing)|
|[AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf)|Raw|570|16,488|-|-|[Google Drive](https://drive.google.com/file/d/1FoZDyzTrs8r_oFM3Xqmi3iAHsnoirTRA/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1-E_hkW-bXsXNYRiAhRPM7A)|
|[AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf)|Align_112x112|570|16,488|-|-| |
|[IJB-A](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Klare_Pushing_the_Frontiers_2015_CVPR_paper.pdf)|Clean|500|5,712|20,412|2,085| |
|[IJB-B](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w6/papers/Whitelam_IARPA_Janus_Benchmark-B_CVPR_2017_paper.pdf)|Raw|1,845|21,798|55,026|7,011|[Google Drive](https://drive.google.com/file/d/15oibCHL3NX-q-QV8q_UAmbIr9e_M0n1R/view?usp=sharing)|
|[CFP](http://www.cfpw.io/paper.pdf)|Raw|500|7,000|-|-|[Google Drive](https://drive.google.com/file/d/1tGNtqzWeUx3BYAxRHBbH1Wy7AmyFtZkU/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/10Qq64LO_RWKD2cr_D32_6A)|
* Remark: unzip CASIA-WebFace clean version with 
```
unzip casia-maxpy-clean.zip    
cd casia-maxpy-clean    
zip -F CASIA-maxpy-clean.zip --out CASIA-maxpy-clean_fix.zip    
unzip CASIA-maxpy-clean_fix.zip
```
* Due to release license issue, for other face related databases, please make contact with us in person for more details.

### Model Zoo 
:monkey:

TO DO

### Achievement 
:confetti_ball:

* 2017 No.1 on ICCV 2017 MS-Celeb-1M Large-Scale Face Recognition [Hard Set](https://www.msceleb.org/leaderboard/iccvworkshop-c1)/[Random Set](https://www.msceleb.org/leaderboard/iccvworkshop-c1)/[Low-Shot Learning](https://www.msceleb.org/leaderboard/c2) Challenges. [WeChat News](http://mp.weixin.qq.com/s/-G94Mj-8972i2HtEcIZDpA), [NUS ECE News](http://ece.nus.edu.sg/drupal/?q=node/215), [NUS ECE Poster](https://zhaoj9014.github.io/pub/ECE_Poster.jpeg), [Award Certificate for Track-1](https://zhaoj9014.github.io/pub/MS-Track1.jpeg), [Award Certificate for Track-2](https://zhaoj9014.github.io/pub/MS-Track2.jpeg), [Award Ceremony](https://zhaoj9014.github.io/pub/MS-Awards.jpeg).
* 2017 No.1 on National Institute of Standards and Technology (NIST) IARPA Janus Benchmark A (IJB-A) Unconstrained Face [Verification](https://zhaoj9014.github.io/pub/IJBA_11_report.pdf) challenge and [Identification](https://zhaoj9014.github.io/pub/IJBA_1N_report.pdf) challenge. [WeChat News](https://mp.weixin.qq.com/s/s9H_OXX-CCakrTAQUFDm8g).

* State-of-the-art performance on 

    * MS-Celeb-1M (Challenge1 Hard Set Coverage@P=0.95: 79.10%; Challenge1 Random Set Coverage@P=0.95: 87.50%; Challenge2 Development Set Coverage@P=0.99: 100.00%; Challenge2 Base Set Top 1 Accuracy: 99.74%; Challenge2 Novel Set Coverage@P=0.99: 99.01%).
    * IJB-A (1:1 Veification TAR@FAR=0.1: 99.6%±0.1%; 1:1 Veification TAR@FAR=0.01: 99.1%±0.2%; 1:1 Veification TAR@FAR=0.001: 97.9%±0.4%; 1:N Identification FNIR@FPIR=0.1: 1.3%±0.3%; 1:N Identification FNIR@FPIR=0.01: 5.4%±4.7%; 1:N Identification Rank1 Accuracy: 99.2%±0.1%; 1:N Identification Rank5 Accuracy: 99.7%±0.1%; 1:N Identification Rank10 Accuracy: 99.8%±0.1%).
    * IJB-C (1:1 Veification TAR@FAR=1e-5: 82.6%).
    * Labeled Faces in the Wild (LFW) (Accuracy: 99.85%±0.217%).
    * Celebrities in Frontal-Profile (CFP) (Frontal-Profile Accuracy: 96.01%±0.84%; Frontal-Profile EER: 4.43%±1.04%; Frontal-Profile AUC: 99.00%±0.35%; Frontal-Frontal Accuracy: 99.64%±0.25%; Frontal-Frontal EER: 0.54%±0.37%; Frontal-Frontal AUC: 99.98%±0.03%).
    * CMU Multi-PIE (Rank1 Accuracy Setting-1 under ±90°: 76.12%; Rank1 Accuracy Setting-2 under ±90°: 86.73%).
    * MORPH Album2 (Rank1 Accuracy Setting-1: 99.65%; Rank1 Accuracy Setting-2: 99.26%).
    * CACD-VS (Accuracy: 99.76%).
    * FG-NET (Rank1 Accuracy: 93.20%).

### Acknowledgement 
:two_men_holding_hands:

* This repo is inspired by [InsightFace.MXNet](https://github.com/deepinsight/insightface), [InsightFace.PyTorch](https://github.com/TreB1eN/InsightFace_Pytorch), [ArcFace.PyTorch](https://github.com/ronghuaiyang/arcface-pytorch), [MTCNN.MXNet](https://github.com/pangyupo/mxnet_mtcnn_face_detection) and [PretrainedModels.PyTorch](https://github.com/Cadene/pretrained-models.pytorch).
* The work of Jian Zhao was partially supported by China Scholarship Council (CSC) grant 201503170248.
* We would like to thank [Prof. Jiashi Feng](https://sites.google.com/site/jshfeng/), [Dr. Jianshu Li](https://sites.google.com/view/li-js), Mr. Yu Cheng (Learning and Vision group, National University of Singapore), Mr. Yuan Xin, Mr. Di Wu, Mr. Zhenyuan Shen (Tencent FiT DeepSea Lab, China), [Prof. Ran He](http://www.nlpr.ia.ac.cn/english/irds/People/rhe.html), [Prof. Junliang Xing](http://people.ucas.ac.cn/~0051452?language=en), [Mr. Xiang Wu](http://alfredxiangwu.github.io/) (Institute of Automation, Chinese Academy of Sciences), [Prof. Guosheng Hu](https://www.linkedin.com/in/guosheng-hu-6801b333/) (AnyVision Inc., U.K.), [Dr. Lin Xiong](https://bruinxiong.github.io/xionglin.github.io/) (JD Digits AI Lab, U.S.), Miss Yi Cheng (Panasonic R\&D Center, Singapore) for helpful discussions.

### Citation 
:bookmark_tabs:

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
