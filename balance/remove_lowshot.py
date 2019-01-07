import os, shutil
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'remove low-shot classes')
    parser.add_argument("-root", "--root", help = "specify your dir",default = './data/train', type = str)
    parser.add_argument("-min_num", "--min_num", help = "remove the classes with less than min_num samples", default = 10, type = int)
    args = parser.parse_args()

    root = args.root # specify your dir
    min_num = args.min_num # remove the classes with less than min_num samples

    cwd = os.getcwd()  # delete '.DS_Store' existed in the source_root
    os.chdir(root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)

    for subfolder in os.listdir(root):
        file_num = len(os.listdir(os.path.join(root, subfolder)))
        if file_num <= min_num:
            print("Class {} has less than {} samples, removed!".format(subfolder, min_num))
            shutil.rmtree(os.path.join(root, subfolder))