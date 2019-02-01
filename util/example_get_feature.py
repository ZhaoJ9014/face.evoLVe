# before run me, please put me at the root folder of face.evoLVe.PyTorch-master
import torch
from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from util.extract_feature_v1 import extract_feature_v1
from util.extract_feature_v2 import extract_feature_v2
import numpy as np

#======= hyperparameters & data loaders =======#
cfg = configurations[1]

SEED = cfg['SEED'] # random seed for reproduce results
torch.manual_seed(SEED)

BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

INPUT_SIZE = cfg['INPUT_SIZE']
RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
RGB_STD = cfg['RGB_STD']
EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
BATCH_SIZE = cfg['BATCH_SIZE']

DEVICE = cfg['DEVICE']


#======= model & loss & optimizer =======#
BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE),
                 'ResNet_101': ResNet_101(INPUT_SIZE),
                 'ResNet_152': ResNet_152(INPUT_SIZE),
                 'IR_50': IR_50(INPUT_SIZE),
                 'IR_101': IR_101(INPUT_SIZE),
                 'IR_152': IR_152(INPUT_SIZE),
                 'IR_SE_50': IR_SE_50(INPUT_SIZE),
                 'IR_SE_101': IR_SE_101(INPUT_SIZE),
                 'IR_SE_152': IR_SE_152(INPUT_SIZE)}
BACKBONE = BACKBONE_DICT['BACKBONE_NAME']
print("=" * 60)
print(BACKBONE)

print("{} Backbone Generated".format(BACKBONE_NAME))
print("=" * 60)

fea_v1 = extract_feature_v1('./util/test/', BACKBONE, './util/Backbone_IR_50_Epoch_30_Batch_122040_Time_2019-02-01-10-22_checkpoint.pth', tta = True)

fea_v2 = extract_feature_v2('./util/test/1/1BHchangjinhua_reg_face0.bmp', BACKBONE, './util/Backbone_IR_50_Epoch_30_Batch_122040_Time_2019-02-01-10-22_checkpoint.pth', tta = True)


np.save("features_v1.npy", fea_v1)
np.save("features_v2.npy", fea_v2)

features_v1 = np.load("features_v1.npy")
features_v2 = np.load("features_v2.npy")

if features_v1.all() == features_v2.all():
    print('True')
