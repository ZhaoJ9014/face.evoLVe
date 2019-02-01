# before run me, please put me at the root folder of face.evoLVe.PyTorch-master
import torch
from config import configurations
from backbone.model_irse import IR_50
from util.extract_feature_v1 import extract_feature_v1
from util.extract_feature_v2 import extract_feature_v2
import numpy as np


cfg = configurations[1]
SEED = cfg['SEED']
torch.manual_seed(SEED)
INPUT_SIZE = cfg['INPUT_SIZE']

BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE),
                 'ResNet_101': ResNet_101(INPUT_SIZE),
                 'ResNet_152': ResNet_152(INPUT_SIZE),
                 'IR_50': IR_50(INPUT_SIZE),
                 'IR_101': IR_101(INPUT_SIZE),
                 'IR_152': IR_152(INPUT_SIZE),
                 'IR_SE_50': IR_SE_50(INPUT_SIZE),
                 'IR_SE_101': IR_SE_101(INPUT_SIZE),
                 'IR_SE_152': IR_SE_152(INPUT_SIZE)}
BACKBONE = IR_50(INPUT_SIZE)
print(BACKBONE)

fea_v1 = extract_feature_v1('./util/test/', BACKBONE, './util/backbone_ir50.pth', tta = True)
fea_v2 = extract_feature_v2('./util/test/1/1BHchangjinhua_reg_face0.bmp', BACKBONE, './util/backbone_ir50.pth', tta = True)

np.save("features_v1.npy", fea_v1)
np.save("features_v2.npy", fea_v2)

features_v1 = np.load("features_v1.npy")
features_v2 = np.load("features_v2.npy")

if features_v1.all() == features_v2.all():
    print('True')
