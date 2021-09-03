from paddlelite.lite import *


def creat_predictor(model_dir):
    '''
    返回face检测器
    '''
    config = MobileConfig()
    config.set_model_from_file(model_dir)
    predictor = create_paddle_predictor(config)
    return predictor
