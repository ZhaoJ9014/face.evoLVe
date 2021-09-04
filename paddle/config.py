
configurations = {
    1: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = 'data/vggface_48w',
        # DATA_ROOT = 'data/casiasmall',
        #the parent root where your train/val/test data are stored
        # If you want to test quickly, you can use the dataset ['defeault']
        # If you need to customize the dataset, please define it according to the following format.
                    # your datafile
                    #   ├── class file 0
                    #   │       └── image0.jpg
                    #   ├── class file 1
                    #   │       └── image0.jpg
                    #   ├── class file 2
                    #   │       ├── image0.jpg
                    #   │       └── image1.jpg
                    #   └── class file 3
                    #           ├── image0.jpg
                    #           └── image1.jpg
                    # .........

        MODEL_ROOT = 'output', # the root to buffer your checkpoints
        LOG_ROOT = 'log', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = './', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = './', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'ppResNet_50', # support: ['ppResNet_50','ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['ArcFace', 'CosFace', 'SphereFace', 'Am_softmax','Softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112]
        RGB_MEAN=[127.5, 127.5, 127.5],  # for normalize inputs to [-1, 1]
        RGB_STD=[127.5, 127.5, 127.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 256,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.001, # initial LR
        NUM_EPOCH = 100, # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [35, 65, 95], # epoch stages to decay learning rate
        GPU_ID = [0], # specify your GPU ids , Mult_gpu training please set GPUs in file start_mult_gpu_train.py
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
        USE_PRETRAINED = True,# If you use the ppResNet_50 backbone, you can load the pre trained weights
        SAVE_CHECKPOINT = False,# save checkpoint 
        SAVE_QUANT_MODEL = False, # model quant_aware training refer https://paddle-lite.readthedocs.io/zh/latest/user_guides/quant_aware.html 
),
}