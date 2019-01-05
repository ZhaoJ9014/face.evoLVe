import torch


configurations = {
    1: dict(
        SEED=1337, # random seed for reproduce results
        DATA_PATH='/media/pc/6T/jasonjzhao/data/faces_emore', # the parent path where your training, val and testing data are stored
        MODEL_PATH='/media/pc/6T/jasonjzhao/buffer/model', # the path to buffer your checkpoint models
        LOG_PATH='/media/pc/6T/jasonjzhao/buffer/log', # the path to log your training and validation status for visualization
        MODEL_NAME='irse50', # support: ['resnet50', 'resnet101', 'resnet152', 'ir50', 'ir101', 'ir152', 'irse50', 'irse101', 'irse152']
        INPUT_SIZE=[112, 112],
        RGB_MEAN=[0.5, 0.5, 0.5],
        RGB_STD=[0.5, 0.5, 0.5],
        EMBEDDING_SIZE=512, # feature dimension
        BATCH_SIZE=512,
        DROP_LAST=True, # whether to drop the last batch to ensure consistent batch_norm
        LR=0.1,
        NUM_EPOCH=20,
        DROP_RATIO=0.6,
        WEIGHT_DECAY=5e-4,
        LR_STEP=[12,15,18], # epoch step number to decay learning rate
        MOMENTUM=0.9,
        DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU=True, # flag to use multiple GPUs
        GPU_ID=[0,1,2,3], # specify your gpu ids
        PIN_MEMORY=True,
        NUM_WORKERS=0,
),
}
