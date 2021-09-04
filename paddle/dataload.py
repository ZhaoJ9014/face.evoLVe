import paddle
import os, tqdm,cv2
import numpy as np
from paddle.vision import transforms
import paddle.vision as vision


class BalancingClassDataset(paddle.io.Dataset):
    def __init__(self, data_root, input_size, mean, std):
        super(BalancingClassDataset, self).__init__()
        self.input_size = input_size
        self.data_root = data_root
        self.mean = mean
        self.std = std
        self.trans  = transforms.Compose([
        # transforms.Resize([int(self.input_size[0]), int(self.input_size[1])]),  # smaller side resized
        transforms.Transpose(order=(2, 0, 1)),
        transforms.Normalize(mean=self.mean,
                             std=self.std),
    ])
        self.weight, self.image_data, self.image_label, self.num_classes = self.data_prepare()
        self.mask = np.arange(len(self.weight))
        self.num_classes = len(self.image_data)
        self.len = len(self.image_data)

    def __getitem__(self, index):
        simpled_index = np.random.choice(self.mask, p=self.weight, replace=True)
        input_data = paddle.to_tensor(self.trans(self.image_data[simpled_index]))
        label = paddle.to_tensor(self.image_label[simpled_index])
        return input_data, label

    def __len__(self):
        return self.len


    def data_prepare(self):
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
        data_root = self.data_root
        class_file_list = os.listdir(data_root)
        for i in tqdm.tqdm(range(len(class_file_list)),ncols=80):
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
        for i in range(num_classes):
            weight_per_class[i] = 1 / (num_classes * count[i])
        weight = [0] * num_sample
        for idx, val in enumerate(images_label):
            weight[idx] = weight_per_class[images_label[idx]]

        return weight, images_data, images_label, num_classes


class NormalDataset(paddle.io.Dataset):
    def __init__(self, data_root, input_size, mean, std):
        super(NormalDataset, self).__init__()
        self.mean = mean
        self.std = std
        self.input_size = input_size
        self.data_root = data_root
        self.trans = transforms.Compose([
            # transforms.Resize([int(self.input_size[0]), int(self.input_size[1])]),  # smaller side resized
            transforms.Transpose(order=(2, 0, 1)),
            transforms.Normalize(mean=self.mean,
                                 std=self.std),
        ])
        self.image_data, self.image_label = self.data_prepare()
        self.num_classes = len(self.image_data)
        self.len = len(self.image_data)

    def __getitem__(self, index):
        input_data = paddle.to_tensor(self.trans(self.image_data[index]))
        label = paddle.to_tensor(self.image_label[index])
        return input_data, label

    def __len__(self):
        return self.len

    def data_prepare(self):
        data_root = self.data_root
        images_data = []
        images_label = []
        count_value = 0
        class_file_list = os.listdir(self.data_root)
        for i in tqdm.tqdm(range(len(class_file_list)),ncols=80):
            images_path = os.listdir(os.path.join(self.data_root, class_file_list[i]))
            for image_path in images_path:
                images_data.append(
                    vision.image_load(data_root + '/' + class_file_list[i] + '/' + image_path, backend='cv2'))
                images_label.append(i)
                count_value += 1
        print(count_value)
        assert len(images_data) == len(images_label)

        return images_data, images_label



if __name__ == '__main__':
    data = NormalDataset('data/casiasmall', [112, 112], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data.__getitem__(1)
