import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val

from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import os


if __name__ == '__main__':

    #======= hyperparameters & dataLoaders =======#
    cfg = configurations[1]

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT'] # the root to log your train/val status

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME'] # support: ['Focal', 'Softmax']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR'] # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH'] # total epoch number (use the firt 1/5 epochs to warm up)
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate

    DEVICE = cfg['DEVICE'] # use GPU or CPU
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    writer = SummaryWriter(LOG_ROOT) # writer for buffering intermedium results

    train_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])

    dataset_train = datasets.ImageFolder(os.path.join(DATA_ROOT, 'imgs'), train_transform)

    # create a weighted random sampler to process imbalanced data
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = BATCH_SIZE, sampler = sampler, pin_memory = PIN_MEMORY,
        num_workers = NUM_WORKERS, drop_last = DROP_LAST
    )

    NUM_CLASS = len(train_loader.dataset.classes)
    print("Number of Training Classes: {}".format(NUM_CLASS))

    agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame = get_val_data(DATA_ROOT)


    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE), 'ResNet_101': ResNet_101(INPUT_SIZE), 'ResNet_152': ResNet_152(INPUT_SIZE),
                     'IR_50': IR_50(INPUT_SIZE), 'IR_101': IR_101(INPUT_SIZE), 'IR_152': IR_152(INPUT_SIZE),
                     'IR_SE_50': IR_SE_50(INPUT_SIZE), 'IR_SE_101': IR_SE_101(INPUT_SIZE), 'IR_SE_152': IR_SE_152(INPUT_SIZE)}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    HEAD_DICT = {'ArcFace': ArcFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS),
                 'CosFace': CosFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS),
                'SphereFace': SphereFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS),
                'Am_softmax': Am_softmax(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS)}
    HEAD = HEAD_DICT[HEAD_NAME]
    print("=" * 60)
    print(HEAD)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)

    LOSS_DICT = {'Focal': FocalLoss(), 'Softmax': nn.CrossEntropyLoss()}
    LOSS = LOSS_DICT[LOSS_NAME]
    print("=" * 60)
    print(LOSS)
    print("{} Loss Generated".format(LOSS_NAME))
    print("=" * 60)

    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # do not do weight decay for batch_norm parameters
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE) # do not do weight decay for batch_norm parameters
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)
    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn,
                            'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr = LR, momentum = MOMENTUM)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
        HEAD = nn.DataParallel(HEAD, device_ids = GPU_ID)
        HEAD = HEAD.to(DEVICE)
    else:
        # single-GPU/CPU setting
        BACKBONE = BACKBONE.to(DEVICE)
        HEAD = HEAD.to(DEVICE)


    # ======= train & validation & save checkpoint =======#
    DISP_LOSS_FREQ = len(train_loader) // 100  # interval to display training loss & acc
    EVALUATE_FREQ = len(train_loader) // 10  # interval to perform validation
    SAVE_FREQ = len(train_loader) // 5  # interval to save checkpoints

    NUM_EPOCH_WARM_UP = NUM_EPOCH // 5 # use the first 1/5 epochs to warm up
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP # use the first 1/5 epochs to warm up
    batch = 0  # batch index

    BACKBONE.train()  # set to training mode
    HEAD.train()

    for epoch in range(NUM_EPOCH): # start training process

        if epoch == STAGES[0]: # adjust LR for each training stage after warm up
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[1]:
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[2]:
            schedule_lr(OPTIMIZER)

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(iter(train_loader)):

            if batch <= NUM_BATCH_WARM_UP - 1: # adjust LR for each training batch during warm up
                warm_up_lr(batch, NUM_BATCH_WARM_UP - 1, LR, OPTIMIZER)

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            features = BACKBONE(inputs)
            outputs = HEAD(features, labels)
            _, preds = torch.max(outputs, 1)
            loss = LOSS(outputs, labels)
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()

            if (batch % DISP_LOSS_FREQ == 0) and batch != 0: # dispaly training loss & acc every DISP_LOSS_FREQ
                display_preds = outputs.data.cpu().numpy()
                display_preds = np.argmax(display_preds, axis = 1)
                dispaly_labels = labels.data.cpu().numpy()
                display_acc = np.mean((display_preds == dispaly_labels).astype(float))
                print("=" * 60)
                if batch <= NUM_BATCH_WARM_UP - 1:
                    print("During Warm Up Process:")
                else:
                    print("During Normal Training Process:")
                print("Epoch {}/{} Batch {}/{}, Training Loss {} Acc {}".format(epoch, NUM_EPOCH - 1, batch, len(train_loader) * NUM_EPOCH - 1, loss.data.item(), display_acc))
                print("=" * 60)

            if (batch % EVALUATE_FREQ) == 0 and batch != 0: # perform validation every EVALUATE_FREQ
                print("=" * 60)
                if batch <= NUM_BATCH_WARM_UP - 1:
                    print("During Warm Up Process:")
                else:
                    print("During Normal Training Process:")
                print("Perform Validation on AgeDB_30, LFW and CFP_FP...")
                accuracy_agedb_30, best_threshold_agedb_30, roc_curve_agedb_30 = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, agedb_30, agedb_30_issame)
                accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame)
                accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame)
                print("Epoch {}/{} Batch {}/{}, Evaluation: AgeDB_30 Acc: {}, LFW Acc: {}, CFP_FP Acc: {}".format(epoch, NUM_EPOCH - 1, batch, len(train_loader) * NUM_EPOCH - 1, accuracy_agedb_30, accuracy_lfw, accuracy_cfp_fp))
                print("=" * 60)


            if (batch % SAVE_FREQ) == 0 and batch != 0: # save checkpoints (only save BACKBONE) every SAVE_FREQ
                torch.save(BACKBONE.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Head_{}_Loss_{}_agedb_30_acc_{}_lfw_acc_{}_cfp_fp_acc_{}_epoch_{}_batch_{}_time_{}".format(BACKBONE_NAME, HEAD_NAME, LOSS_NAME, accuracy_agedb_30, accuracy_lfw, accuracy_cfp_fp, epoch, batch, get_time())))

            running_loss += loss.data.item() * inputs.data.size(0) # compute training loss & acc every epoch
            running_corrects += torch.sum(preds == labels.data)

            batch += 1 # batch index

        # training statistics per epoch (buffer for visualization)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        writer.add_scalar("Training_Loss", epoch_loss, epoch)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch)
        print("=" * 60)
        if epoch <= NUM_EPOCH_WARM_UP - 1:
            print("During Warm Up Process:")
        else:
            print("During Normal Training Process:")
        print("Epoch {}/{}, Training Loss {} Acc {}".format(epoch, NUM_EPOCH - 1, epoch_loss, epoch_acc))
        print("=" * 60)

        # validation statistics per epoch (buffer for visualization)
        print("=" * 60)
        if epoch <= NUM_EPOCH_WARM_UP - 1:
            print("During Warm Up Process:")
        else:
            print("During Normal Training Process:")
        print("Perform Validation on AgeDB_30, LFW and CFP_FP...")
        accuracy_agedb_30, best_threshold_agedb_30, roc_curve_agedb_30 = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, agedb_30, agedb_30_issame)
        buffer_val(writer, "AgeDB_30", accuracy_agedb_30, best_threshold_agedb_30, roc_curve_agedb_30, epoch)
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame)
        buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch)
        accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame)
        buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch)
        print("Epoch {}/{}, Evaluation: AgeDB_30 Acc: {}, LFW Acc: {}, CFP_FP Acc: {}".format(epoch, NUM_EPOCH - 1, accuracy_agedb_30, accuracy_lfw, accuracy_cfp_fp))
        print("=" * 60)
