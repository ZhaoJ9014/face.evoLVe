# 使用动态离线量化 文档说明https://paddle-lite.readthedocs.io/zh/latest/user_guides/quant_post_dynamic.html
import paddle
from paddleslim.quant import quant_post_dynamic

if __name__ == '__main__':
    paddle.enable_static()
    model_dir = 'output/'
    save_model_dir = 'Backbone_epoch99'
    quant_post_dynamic(model_dir=model_dir,
                        model_filename='output/Backbone_epoch99.pdmodel',
                    params_filename='output/Backbone_epoch99.pdiparams',
                    save_model_dir=save_model_dir,
                    save_params_filename='__params__',
                    weight_bits=16,
                    quantizable_op_type=['conv2d', 'mul'],
                    generate_test_model=False)