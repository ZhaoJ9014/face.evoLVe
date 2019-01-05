import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from backbone.ir_se import ir50, ir101, ir152, irse50, irse101, irse152
from backbone.resnet import resnet50, resnet101, resnet152
from head.metrics import ArcFace, CosineFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, schedule_lr, perform_validation, buffer_val, get_time
from config import configurations
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    #======= parameters, tensorboardx summarywriter and dataLoaders =======#
    cfg = configurations[1]

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_PATH = cfg['DATA_PATH'] # the parent path where your training, val and testing data are stored
    MODEL_PATH = cfg['MODEL_PATH'] # the path to buffer your checkpoint models
    LOG_PATH = cfg['LOG_PATH'] # the path to log your training and validation status for visualization

    MODEL_NAME = cfg['MODEL_NAME'] 

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN']
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST'] # whether to drop the last batch to ensure consistent batch_norm
    LR = cfg['LR']
    NUM_EPOCH = cfg['NUM_EPOCH']
    DROP_RATIO = cfg['DROP_RATIO']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    LR_STEP = cfg['LR_STEP']

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your gpu ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']

    writer = SummaryWriter(LOG_PATH)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN,
                             std=RGB_STD),
    ])

    dataset_train = datasets.ImageFolder(os.path.join(DATA_PATH, 'imgs'), train_transform)
    
    # For unbalanced dataset we create a weighted sampler
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, sampler=sampler, drop_last=True)
    num_class = len(train_loader.dataset.classes)
    print('Number of Training Classes: %d' % num_class)

    agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame = get_val_data(DATA_PATH)

    #======= model & loss & optimizer =======#
    if MODEL_NAME == 'irse50':
        backbone = irse50(DROP_RATIO)
    print(backbone)
    print("Backbone Generated")

    head = ArcFace(EMBEDDING_SIZE, num_class, s=30, m=0.5, easy_margin=False)
    print(head)
    print("Head Generated")

    # loss
    criterion = FocalLoss(gamma=2)

    # optimizer
    if MODEL_NAME.find('ir') >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(backbone) # do not do weight decay for batch_norm parameters 
        _, head_paras_wo_bn = separate_irse_bn_paras(head)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(backbone) # do not do weight decay for batch_norm parameters 
        _, head_paras_wo_bn = separate_resnet_bn_paras(head)
        
    optimizer = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr=LR, momentum=MOMENTUM)
    print(optimizer)
    print('Optimizer Generated')

    if MULTI_GPU:
        # multi-GPU setting
        backbone = nn.DataParallel(backbone, device_ids=GPU_ID)
        backbone = backbone.to(DEVICE)
        head = nn.DataParallel(head, device_ids=GPU_ID)
        head = head.to(DEVICE)
    else:
        # single-GPU/cpu setting
        backbone = backbone.to(DEVICE)
        head = head.to(DEVICE)

    #======= train & validation & save checkpoint =======#
    dispaly_freq = len(train_loader) // 100 # frequency to display training loss & acc
    validate_freq = len(train_loader) // 10 # frequency to perform validation
    save_freq = len(train_loader) // 5 # frequency to save checkpoints

    for epoch in range(NUM_EPOCH):

        backbone.train()

        if epoch == LR_STEP[0]: # adjust LR every LR_STEP
            schedule_lr(optimizer)
        if epoch == LR_STEP[1]:
            schedule_lr(optimizer)      
        if epoch == LR_STEP[2]:
            schedule_lr(optimizer)   

        batch = 0
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(iter(train_loader)):

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            features = backbone(inputs)
            outputs = head(features, labels)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch % dispaly_freq == 0) and batch != 0: # dispaly training loss & acc every buffer_loss_freq
                display_preds = outputs.data.cpu().numpy()
                display_preds = np.argmax(display_preds, axis=1)
                dispaly_labels = labels.data.cpu().numpy()
                display_acc = np.mean((display_preds == dispaly_labels).astype(float))
                print('Epoch {}/{} Batch {}/{}, Training Loss {} Acc {}'.format(epoch, NUM_EPOCH - 1, batch, len(train_loader) - 1, loss.data.item(), display_acc))

            if (batch % validate_freq) == 0 and batch != 0: # perform validation every evaluate_freq
                print("Perform Validation on AgeDB_30, LFW and CFP_FP...")
                accuracy_agedb_30, best_threshold_agedb_30, roc_curve_agedb_30 = perform_validation(backbone, agedb_30, agedb_30_issame)
                accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_validation(backbone, lfw, lfw_issame)
                accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_validation(backbone, cfp_fp, cfp_fp_issame)
                print("Epoch {}/{} Batch {}/{}, Evaluation: AgeDB_30 Acc: {}, LFW Acc: {}, CFP_FP Acc: {}".format(epoch, NUM_EPOCH - 1, batch, len(train_loader) - 1, accuracy_agedb_30, accuracy_lfw, accuracy_cfp_fp))
                backbone.train()

            if (batch % save_freq) == 0 and batch != 0: # save checkpoints every save_freq
                torch.save(backbone.state_dict(), os.path.join(MODEL_PATH, 'model_{}_agedb_30_acc_{}_lfw_acc_{}_cfp_fp_acc_{}_epoch_{}_batch_{}_time_{}'.format(MODEL_NAME, accuracy_agedb_30, accuracy_lfw, accuracy_cfp_fp, epoch, batch, get_time())))

            running_loss += loss.data.item() * inputs.data.size(0) # compute training loss & acc every epoch
            running_corrects += torch.sum(preds == labels.data) 

            batch += 1 # batch index

        # training statistics per epoch (buffer for visualization)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        writer.add_scalar('Training_Loss', epoch_loss, epoch)
        writer.add_scalar('Training_Accuracy', epoch_acc, epoch)
        print('Epoch {}/{}, Training Loss {} Acc {}'.format(epoch, NUM_EPOCH - 1, epoch_loss, epoch_acc))

        # validation statistics per epoch (buffer for visualization)
        accuracy_agedb_30, best_threshold_agedb_30, roc_curve_agedb_30 = perform_validation(backbone, agedb_30, agedb_30_issame)
        buffer_val(writer, 'AgeDB_30', accuracy_agedb_30, best_threshold_agedb_30, roc_curve_agedb_30, epoch)
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_validation(backbone, lfw, lfw_issame)
        buffer_val(writer, 'LFW', accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch)
        accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_validation(backbone, cfp_fp, cfp_fp_issame)
        buffer_val(writer, 'CFP_FP', accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch)
        print("Epoch {}/{}, Evaluation: AgeDB_30 Acc: {}, LFW Acc: {}, CFP_FP Acc: {}".format(epoch, NUM_EPOCH - 1, accuracy_agedb_30, accuracy_lfw, accuracy_cfp_fp))
