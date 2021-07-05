## 简介

这是基于PaddlePaddle框架复现的训练代码
****
## Contents
* [Data Zoo](#Data-Zoo)
* [Model Zoo](#Model-Zoo)
****
### Data Zoo 
:tiger:

|数据集|Version|\#Identity|\#Image|\#Frame|\#Video|下载地址|
|:---:|:----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|Raw|5,749|13,233|-|-|[Google Drive](https://drive.google.com/file/d/1JIgAXYqXrH-RbUvcsB3B6LXctLU9ijBA/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1VzSI_xqiBw-uHKyRbi6zzw)

### Model Zoo

* Model

  |Backbone|Head|Loss|
  |:---:|:---:|:---:|
  |[ResNet-50](https://arxiv.org/pdf/1512.03385.pdf)|[ArcFace](https://arxiv.org/pdf/1801.07698.pdf)|[Focal](https://arxiv.org/pdf/1708.02002.pdf)|


  |benchmark|[LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf)|[CFP_FF](http://www.cfpw.io/paper.pdf)|[CFP_FP](http://www.cfpw.io/paper.pdf)|[AgeDB](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf)|[CALFW](https://arxiv.org/pdf/1708.08197.pdf)|[CPLFW](http://www.whdeng.cn/CPLFW/Cross-Pose-LFW.pdf)|[Vggface2_FP](https://arxiv.org/pdf/1710.08092.pdf)|
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  |Top1准确率（%）|99.219(95.950)|-|-|-|-|-|-|
  |Top5准确率（%）|99.219(97.542)|-|-|-|-|-|-|

****

### 环境依赖
  paddlepaddle==2.1.0

  更详细的见requirements.txt

### 代码结构以及详细说明

* `paddle/backbone`文件夹下是关于backbone的code 目前有ResNet系列和IR_SE系列
* `paddle/head`文件夹内包含了`'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax','Softmax'`的code,其中ppResNet使用了预训练模型
* `paddle/loss`文件夹内是`Focal loss`的code
* `paddle/config.py` 训练配置文件
* `paddle/dataload.py` 加载数据集
* `paddle/train.py` 单卡训练


### 快速开始

* 在`paddle/data` 配置好数据集(见config.py)
* 运行`train.py`文件即可开始训练
  
  