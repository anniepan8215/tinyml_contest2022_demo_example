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
    # for pandas
    text_path = root_dir + names_list[idx]  # local path of target data

    if not os.path.isfile(text_path):
        print(text_path + 'does not exist')
        return None

    IEGM_seg = txt_to_numpy(text_path, 1250).squeeze()
    # print(IEGM_seg)
    label = int(names_dict[names_list[idx]])
    # sample = np.append(IEGM_seg, label)
    df = pd.DataFrame([[names_list[idx], IEGM_seg, label]], columns=['File Name', 'Data', 'Label'])

    return df  # return dataframe type dataset with column name


def data_plot(data, SIZE):
    # plot in both time and frequency domain
    t = np.arange(0, SIZE, 1)
    y_time = data['IEGM_seg'].squeeze()
    # grating = np.sin(2 * np.pi * t / 200)
    # print(grating.shape)
    y_freq = np.fft.fft(y_time)
    freq = np.fft.fftfreq(t.shape[-1])

    plt.subplot(211)
    plt.plot(t, y_time)

    plt.subplot(212)
    plt.plot(freq, y_freq)

    plt.show()


def fft_transfer(y_time,SIZE):
    t = np.arange(0, SIZE, 1)
    y_freq = np.fft.fft(y_time)  # calculate fft on series
    freq = np.fft.fftfreq(t.shape[-1])  # frequency
    return (freq,y_freq)

def status(x) :
    return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),
                      x.quantile(.75),x.mean(),x.max(),x.idxmax(),x.mad(),x.var(),
                      x.std(),x.skew(),x.kurt()],index=['总数','最小值','最小值位置','25%分位数',
                    '中位数','75%分位数','均值','最大值','最大值位数','平均绝对偏差','方差','标准差','偏度','峰度'])

def main():
    # Hyperparameters
    SIZE = args.size  # data point number per data
    path_data = args.path_data
    path_indices = args.path_indices
    # names_list = []
    names_dict = {}

    # loading data files from test_indics.csv and train_indice.csv
    csvdata_all = loadCSV(os.path.join(path_indices, 'train' + '_indice.csv'))
    csvdata_all.update(loadCSV(os.path.join(path_indices, 'test' + '_indice.csv')))
    print("Loading file path complete")

    for i, (data_name, label) in enumerate(csvdata_all.items()):
        names_dict[str(data_name)] = label[0] # store data name and its index as dictionary

    names_list = list(names_dict.keys())  # all data file name

    data_all = pd.DataFrame()
    for i in range(len(names_list)):
        # for i in range(5):
        df = load_data_to_array(path_data, names_list, names_dict, i)
        data_all = pd.concat([data_all, df], ignore_index=True)
    print("loading dataframe complete")

    print(data_all['Label'].value_counts())
    '''
    for all given data:
        0    15987
        1    14226
    '''

    data_0 = data_all.loc[data_all['Label'] == 0]
    data_1 = data_all.loc[data_all['Label'] == 1]





    # t = np.arange(0,SIZE,1)
    # y_time = data_1['IEGM_seg'].squeeze()
    # y_freq = np.fft.fft(y_time)  # calculate fft on series
    # freq = np.fft.fftfreq(t.shape[-1])  # frequency
    # # how to use np.fft.fft2()?


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
