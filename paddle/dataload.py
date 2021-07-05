import paddle,cv2
import os, tqdm
import paddle.vision as vision
import numpy as np


class LFWDataset(paddle.io.Dataset):
    def __init__(self, data_root, tramsform):
        super(LFWDataset, self).__init__()
        self.data_root = data_root
        if data_root == 'defeault':
            # mnist FOR test
            self.num_classes = 10
            self.mnist = paddle.vision.datasets.MNIST(mode='test')
            self.len = len(self.mnist)
        else:
            self.tramsform = tramsform
            self.weight, self.image_data, self.image_label, self.num_classes = data_prepare(data_root)
            self.mask = np.arange(len(self.weight))
            self.num_classes = len(self.image_data)
            self.len = len(self.image_data)

    def __getitem__(self, index):
        if self.data_root == 'defeault':
            # mnist FOR test
            data = np.array(self.mnist[index][0]).astype('float32')
            data = paddle.to_tensor(cv2.resize(data, (112, 112)) / 255.0 - 0.5)
            data = paddle.expand(data, [3, 112, 112])
            label = paddle.to_tensor(self.mnist[index][1])
        else:
            simpled_index = np.random.choice(self.mask, p=self.weight, replace=True)
            data = paddle.to_tensor(self.tramsform(self.image_data[simpled_index]))
            label = paddle.to_tensor(self.image_label[simpled_index])
        return data, label

    def __len__(self):
        return self.len


def data_prepare(data_root):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = []
    images_data = []
    images_label = []
    class_file_list = os.listdir(data_root)
    for i in tqdm.tqdm(range(len(class_file_list))):
        images_path = os.listdir(os.path.join(data_root, class_file_list[i]))
        count_value = 0
        for image_path in images_path:
            images_data.append(
                vision.image_load(data_root + '/' + class_file_list[i] + '/' + image_path, backend='cv2'))
            images_label.append(i)
            count_value += 1
        count.append(count_value)
    num_classes = len(class_file_list)
    num_sample = len(images_label)
    weight_per_class = [0.] * num_classes
    for i in tqdm.tqdm(range(num_classes)):
        weight_per_class[i] = 1 / (num_classes * count[i])
    weight = [0] * num_sample
    for idx, val in enumerate(images_label):
        weight[idx] = weight_per_class[images_label[idx]]

    return weight, images_data, images_label, num_classes
