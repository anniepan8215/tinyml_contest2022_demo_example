import os

from help_code_demo import ToTensor, IEGM_DataSET, txt_to_numpy, loadCSV
import numpy as np
import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


def load_data_to_dict(root_dir, names_list, names_dict, idx):
    text_path = root_dir + names_list[idx]  # local path of target data

    if not os.path.isfile(text_path):
        print(text_path + 'does not exist')
        return None

    IEGM_seg = txt_to_numpy(text_path, 1250).reshape(1, 1250, 1)
    label = int(names_dict[names_list[idx]])
    sample = {'IEGM_seg': IEGM_seg, 'label': label}

    return sample  # return a dictionary with data in numpy.array and label


def load_data_to_array(root_dir, names_list, names_dict, idx):

    text_path = root_dir + names_list[idx]  # local path of target data

    if not os.path.isfile(text_path):
        print(text_path + 'does not exist')
        return None

    IEGM_seg = txt_to_numpy(text_path, 1250).reshape(1, 1250, 1)
    label = int(names_dict[names_list[idx]])
    sample = np.array([label,IEGM_seg])

    return sample  # return a np.array with data in numpy.array and label



def main():
    # Hyperparameters
    SIZE = args.size # data point number per data
    path_data = args.path_data
    path_indices = args.path_indices
    # names_list = []
    names_dict = {}


    # loading data
    csvdata_all = loadCSV(os.path.join(path_indices, 'train' + '_indice.csv'))
    csvdata_all.update(loadCSV(os.path.join(path_indices, 'test' + '_indice.csv')))

    for i, (data_name, label) in enumerate(csvdata_all.items()):
        names_dict[str(data_name)] = str(label[0])  # store data name and its index as dictionary

    names_list = list(names_dict.keys())  # all data file name

    data_1 = load_data_to_dict(path_data, names_list, names_dict, 0)

    t = np.arange(0,SIZE,1)
    y_time = data_1['IEGM_seg'].squeeze()
    # grating = np.sin(2 * np.pi * t / 200)
    # print(grating.shape)
    y_freq = np.fft.fft(y_time)
    freq = np.fft.fftfreq(t.shape[-1])

    plt.subplot(211)
    plt.plot(t,y_time)

    plt.subplot(212)
    plt.plot(freq,y_freq)

    plt.show()






if __name__ == '__main__':
    # set data information
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--size', type=int, default=1250)  # data length
    argparser.add_argument('--path_data', type=str, default='./data/')  # data location
    argparser.add_argument('--path_indices', type=str, default='./data_indices')  # indices location

    args = argparser.parse_args()
    #
    # device = torch.device("cuda:" + str(args.cuda))
    #
    # print("device is --------------", device)

    main()
