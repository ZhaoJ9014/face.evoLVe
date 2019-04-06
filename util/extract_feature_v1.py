# Helper function for extracting features from pre-trained models
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import os


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def de_preprocess(tensor):

    return tensor * 0.5 + 0.5


hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


def extract_feature(data_root, backbone, model_root, input_size = [112, 112], rgb_mean = [0.5, 0.5, 0.5], rgb_std = [0.5, 0.5, 0.5], embedding_size = 512, batch_size = 512, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = True):

    # pre-requisites
    assert(os.path.exists(data_root))
    print('Testing Data Root:', data_root)
    assert (os.path.exists(model_root))
    print('Backbone Model Root:', model_root)

    # define data loader
    transform = transforms.Compose([
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
        transforms.CenterCrop([input_size[0], input_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean = rgb_mean, std = rgb_std)])
    dataset = datasets.ImageFolder(data_root, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 0)
    NUM_CLASS = len(loader.dataset.classes)
    print("Number of Classes: {}".format(NUM_CLASS))

    # load backbone from a checkpoint
    print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root))
    backbone.to(device)

    # extract features
    backbone.eval() # set to evaluation mode
    idx = 0
    features = np.zeros([len(loader.dataset), embedding_size])
    with torch.no_grad():
        iter_loader = iter(loader)
        while idx + batch_size <= len(loader.dataset):
            batch, _ = iter_loader.next()
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = backbone(batch.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                features[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                features[idx:idx + batch_size] = l2_norm(backbone(batch.to(device))).cpu()
            idx += batch_size

        if idx < len(loader.dataset):
            batch, _ = iter_loader.next()
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = backbone(batch.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                features[idx:] = l2_norm(emb_batch)
            else:
                features[idx:] = l2_norm(backbone(batch.to(device)).cpu())
                
#     np.save("features.npy", features) 
#     features = np.load("features.npy")

    return features
