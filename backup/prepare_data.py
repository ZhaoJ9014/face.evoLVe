from config import get_config
from data.data_pipe import load_bin, load_mx_rec
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'for extracting faces_emore data')
    parser.add_argument("-r", "--rec_path", help="mxnet record file path",default = 'faces_emore', type = str)
    args = parser.parse_args()
    conf = get_config()
    rec_path = conf.data_path/args.rec_path
    load_mx_rec(rec_path)
    
    bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
    
    for i in range(len(bin_files)):
        load_bin(rec_path/(bin_files[i] + '.bin'), rec_path/bin_files[i], conf.test_transform)