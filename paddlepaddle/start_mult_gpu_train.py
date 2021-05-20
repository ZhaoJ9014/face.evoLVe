# coding=utf-8
import os

if __name__ == "__main__":
    os.system(
        f"fleetrun --gpus=0,1,2,3,4,5,6,7 start_mult_gpu_train.py")
