## 简介

这是基于PaddlePaddle框架复现的训练代码，
同时提供了
[PaddleSlim](https://github.com/PaddlePaddle/Paddle-Lite)
模型量化方案、
[Paddle-Inference](https://paddle-inference.readthedocs.io/en/latest/)
服务端GPU部署方案、
[PaddleLite](https://github.com/PaddlePaddle/Paddle-Lite)
边缘端部署方案。

Raw face image:

<img src="https://github.com/Reatris/lab/blob/master/DB.png" width="571"/>

* 图.《爱情公寓》场景人物面部识别

<img src="https://github.com/Reatris/lab/blob/master/result_Moment.jpg" width="1280"/>

****
## Contents
* [Data Zoo](###Data-Zoo)
* [Model Zoo](###Model-Zoo)
* [Inference](##Inference)
****

### Data-Zoo 
:tiger:

|数据集|Version|\#Identity|\#Image|\#Frame|\#Video|下载地址|
|:---:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|Raw|5,749|13,233|-|-|[Google Drive](https://drive.google.com/file/d/1JIgAXYqXrH-RbUvcsB3B6LXctLU9ijBA/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1VzSI_xqiBw-uHKyRbi6zzw)
|[Vgg-Face2-clean](https://arxiv.org/pdf/1710.08092.pdf)|Align_112x112|1,333|440,028|-|-|[AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/102305), [-]()
|[CASIA-WebFace](https://arxiv.org/pdf/1411.7923.pdf)|Align_112x112|10,575|455,594|-|-|[AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/103163), [-]()
### Model-Zoo

* Model

  |Backbone|Head|Loss|
  |:---:|:---:|:---:|
  |[ResNet-50](https://arxiv.org/pdf/1512.03385.pdf)|[ArcFace](https://arxiv.org/pdf/1801.07698.pdf)|[Focal](https://arxiv.org/pdf/1708.02002.pdf)|


  |benchmark|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|[Vgg-Face2-clean](https://arxiv.org/pdf/1710.08092.pdf)|[CASIA-WebFace](https://arxiv.org/pdf/1411.7923.pdf)|[CFP_FF](http://www.cfpw.io/paper.pdf)|[CFP_FP](http://www.cfpw.io/paper.pdf)|[AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf)|[CALFW](https://arxiv.org/pdf/1708.08197.pdf)|[CPLFW](http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf)|[Vggface2_FP](https://arxiv.org/pdf/1710.08092.pdf)|
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  |Top1准确率（%）|99.219(95.950)|99.219(98.243)|99.219(99.365)|-|-|-|-|
  |Top5准确率（%）|99.219(97.542)|99.609(99.445)|100.000(99.726)|-|-|-|-|

* Training \& validation statistics
  
    <img src="https://github.com/Reatris/lab/blob/master/fig1.png" width="1000px"/>
      



****

### 环境依赖
  paddlepaddle==2.1.0,paddleslim==2.1.0等

### 代码结构以及详细说明

* .
* ├── align   #人脸对齐与处理
* │   ├── align_trans.py
* │   ├── box_utils.py
* │   ├── detector.py
* │   ├── face_align.py
* │   ├── face_resize.py
* │   ├── get_nets.py
* │   └── __init__.py
* ├── backbone
* │   ├── __init__.py
* │   ├── model_irse.py
* │   ├── model_resnet.py
* │   └── resnet_pp.py
* ├── config.py   #训练配置文件
* ├── data    # 数据集
* ├── dataload.py   # 数据读取与处理
* ├── head
* │   ├── __init__.py
* │   └── metrics.py
* ├── __init__.py
* ├── loss
* │   ├── focal.py
* │   └── __init__.py
* ├── mult_gpu_training.py
* ├── PaddleInference-demo  # PaddleInference推理部署方案
* │   ├── FaceDatabase
* │   ├── main.py
* │   ├── model
* │   └── utils.py
* ├── Paddle-Lite-Inference-demo  # Paddle-Lite推理部署方案
* │   ├── FaceDatabase
* │   ├── main.py
* │   ├── model
* │   ├── MTCNN.py
* │   ├── README.md
* │   └── utils.py
* ├── pretrained #预训练模型存放路径
* ├── quant #模型量化训练
* │   ├── quant_post_dynamic.py
* │   ├── quant_post_static.py
* │   └── README.md
* ├── start_mult_gpu_train.py#多GPU训练
* ├── train.py#单GPU训练
* └── utils.py



### 快速开始

* 在`paddle/data` 配置好数据集(见config.py)
* 运行`train.py`文件即可开始训练



## Inference&推理部署

**以下是我在不同设备不同模型上的推理情况**

|Model|Device|Inference Engine|TRT加速|(quant_aware_int8)量化|模型体积(MB)|时延(ms)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Backbone(ResNet50)|GPU TESLA v100|Paddle-Inference|Y|-|153|2.71|
|Backbone(ResNet50)|GPU GTX 1650Ti|Paddle-Inference|-|-|153|5.42|
|Backbone(ResNet50)|GPU MaxWell(jetson nano)|Paddle-Inference|-|-|153|48.42|
|Backbone(ResNet50)|CPU Raspberry Pi 4B|Paddle-Lite|-|-|153|243.22|
|Backbone(ResNet50)|CPU Raspberry Pi 4B|Paddle-Lite|-|Y|39|167.81|
|MTCNN|GPU TESLA v100|Paddle-Inference|-|-|-|14.13|
|MTCNN|GPU GTX 1650Ti|Paddle-Inference|-|-|-|15.56|
|MTCNN|GPU MaxWell(jetson nano)|Paddle-Inference|-|-|-|138.41|
|MTCNN|CPU Raspberry Pi 4B|Paddle-Lite|-|-|-|210.49|

###Paddle-Inference 部署

Paddle Inference为飞桨核心框架推理引擎。Paddle Inference功能特性丰富，性能优异，针对服务器端应用场景进行了深度的适配优化，做到高吞吐、低时延，保证了飞桨模型在服务器端即训即用，快速部署。

此项目通过Paddle Inference进行部署应用，在此Repo中提供了Python的使用样例

如果你对Paddle Inference有所疑惑，可以访问下面这个链接

[Paddle-Inference-API 文档](https://paddle-inference.readthedocs.io/en/latest/)

* PaddleInference-demo
*    ├── FaceDatabase
*    ├── main.py
*    ├── model
*    └── utils.py


### Paddle-Lite 部署

Paddle Lite是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位支持包括移动端、嵌入式以及服务器端在内的多硬件平台。

当前Paddle Lite不仅在百度内部业务中得到全面应用，也成功支持了众多外部用户和企业的生产任务

此项目通过Paddle Lite进行部署应用，我在此Repo中提供了Python的使用样例

如果你对此处有所疑惑，可以访问[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) , [Paddle-Lite's documentation](https://paddle-lite.readthedocs.io/zh/latest/index.html)

* Paddle-Lite-Inference-demo
* ├── FaceDatabase
* ├── main.py
* ├── model
* ├── MTCNN.py
* ├── README.md
* └── utils.py

### 模型量化

[PaddleSlim](https://paddleslim.readthedocs.io/zh_CN/latest/intro.html)
是一个专注于深度学习模型压缩的工具库，提供剪裁、量化、蒸馏、和模型结构搜索等模型压缩策略，帮助用户快速实现模型的小型化。

在此Repo中我提供了PaddleSlim针对本模型进行量化的方法供大家参考
* quant
* ├── quant_post_dynamic.py #动态离线量化
* ├── quant_post_static.py #静态离线量化
* └── README.py
* 量化训练是最佳的量化方式，训练前可以在`config.py`中开启量化训练