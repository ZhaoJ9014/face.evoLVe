# 静态离线量化 文档地址https://paddle-lite.readthedocs.io/zh/latest/user_guides/quant_post_static.html
import paddle,os
import paddleslim
import tqdm,cv2
import numpy as np
import paddle.vision as vision


def batch_generator(data_reader):
    def __reader__():
        for indx, data in enumerate(data_reader()):
            yield data
    return __reader__


def data_reader():
    # 数据读取器，这里要单独写个
    data_root = 'data/Casia_maxpy_clean'
    images_data = []
    images_label = []
    batch_size = 128
    count_value = 0
    class_file_list = os.listdir(data_root)
    for i in tqdm.tqdm(range(len(class_file_list)),ncols=80):
        images_path = os.listdir(os.path.join(data_root, class_file_list[i]))
        for image_path in images_path:
            images_data.append(
                vision.image_load(data_root + '/' + class_file_list[i] + '/' + image_path, backend='cv2'))
            images_label.append(i)
            count_value += 1
    print(count_value)
    assert len(images_data) == len(images_label)
    def reader():
        for i in range(int(len(images_data)/batch_size)-1):
            data = images_data[i*batch_size:(i+1)*batch_size]
            data = np.array(data)
            data = np.transpose(data,(0,3, 1, 2))
            data = (data - 127.5) / 127.5
            yield data
    return reader
if __name__ == '__main__':
    paddle.enable_static()    # 静态图模式
    USE_GPU = True  # 使用GPU
    simple_generator = batch_generator(data_reader())
    place = paddle.CUDAPlace(0) if USE_GPU else paddle.CPUPlace()               
    exe = paddle.static.Executor(place)
    paddleslim.quant.quant_post_static(
            executor=exe,
            model_dir='./output',
            model_filename='/output/Backbone_epoch99.pdmodel',
            params_filename='/output/Backbone_epoch99.pdiparams',
            quantize_model_path='quant_post_static_model',
            sample_generator=simple_generator,
            batch_size=128,
            batch_nums=10)