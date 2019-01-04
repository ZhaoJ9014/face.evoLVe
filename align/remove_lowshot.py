import os, shutil

root = './test' # Modify to your dir
min_num = 10 # Delete the classes with less than 10 samples

for subfolder in os.listdir(root):
    file_num = len(os.listdir(os.path.join(root, subfolder)))
    if file_num <= min_num:
        print('Class {} has less than {} samples, removed!'.format(subfolder, min_num))
        shutil.rmtree(os.path.join(root, subfolder))
