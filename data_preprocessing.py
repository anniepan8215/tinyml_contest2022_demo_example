import os

from help_code_demo import ToTensor, IEGM_DataSET, txt_to_numpy, loadCSV, stats_report
import numpy as np
import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def load_data_to_dict(root_dir, names_list, names_dict, idx):
    text_path = root_dir + names_list[idx]  # local path of target data

    if not os.path.isfile(text_path):
        print(text_path + 'does not exist')
        return None

    IEGM_seg = txt_to_numpy(text_path, 1250).reshape(1, 1250, 1)
    label = int(names_dict[names_list[idx]])
    sample = {'IEGM_seg': IEGM_seg, 'label': label}

    return sample  # return a dictionary with data in numpy.array and label


def load_data_to_df(root_dir, names_list, names_dict, mode, idx):
    # for pandas
    name = names_list[idx]
    text_path = root_dir + name  # local path of target data

    if not os.path.isfile(text_path):
        print(text_path + 'does not exist')
        return None

    IEGM_seg = txt_to_numpy(text_path, 1250).squeeze()
    # print(type(IEGM_seg))
    # print(IEGM_seg)
    c = int(names_dict[name])
    l = name.split("-")[1]
    # sample = np.append(IEGM_seg, label)
    df = pd.DataFrame([[name, IEGM_seg, mode, c, l]], columns=['File Name', 'Data', 'Mode', 'Class', 'Label'])

    return df  # return dataframe type dataset with column name


def data_plot(data, SIZE, title):
    # plot in both time and frequency domain
    t = np.arange(0, SIZE, 1)
    # y_time = data['IEGM_seg'].squeeze()
    y_time = data
    # grating = np.sin(2 * np.pi * t / 200)
    # print(grating.shape)
    y_freq = np.fft.fft(y_time)
    freq = np.fft.fftfreq(t.shape[-1])
    plt.subplot(211)
    plt.plot(t, y_time)
    plt.subplot(212)
    plt.plot(freq, y_freq)
    plt.suptitle(title)
    plt.show()


def data_list_plot(data_list, name_list, SIZE, title):
    # plot in both time and frequency domain
    t = np.arange(0, SIZE, 1)
    # y_time = data['IEGM_seg'].squeeze()
    plt.figure(figsize=(20, 10))
    for i, y_time in enumerate(data_list):
        name = name_list[i]
        y_freq = np.fft.fft(y_time)
        freq = np.fft.fftfreq(t.shape[-1])
        plt.subplot(211)
        plt.plot(t, y_time, label=name)
        plt.subplot(212)
        plt.plot(freq, y_freq, label=name)
    plt.suptitle(title)
    plt.legend(loc='best')
    plt.show()


def fft_transfer(y_time, SIZE):
    t = np.arange(0, SIZE, 1)
    y_freq = np.fft.fft(y_time)  # calculate fft on series
    freq = np.fft.fftfreq(t.shape[-1])  # frequency
    return (freq, y_freq)


def status(x):
    return pd.Series([x.count(), x.min(), x.idxmin(), x.quantile(.25), x.median(),
                      x.quantile(.75), x.mean(), x.max(), x.idxmax(), x.mad(), x.var(),
                      x.std(), x.skew(), x.kurt()], index=['总数', '最小值', '最小值位置', '25%分位数',
                                                           '中位数', '75%分位数', '均值', '最大值', '最大值位数', '平均绝对偏差', '方差', '标准差',
                                                           '偏度', '峰度'])


def load_name(datapaths):
    names_dict = {}
    for i, (data_name, label) in enumerate(datapaths.items()):
        names_dict[str(data_name)] = label[0]  # store data name and its index as dictionary

    return names_dict


def log_reg(data_all):
    """
    Build logistic regression model
    :param data_all: data in dataframe
    :return: model
    """
    X_train = np.array([data for data in data_all.loc[data_all['Mode'] == 'train']['Data']])
    y_train = data_all.loc[data_all['Mode'] == 'train']['Class'].to_numpy(dtype=float)
    print(X_train.shape)
    print(y_train.shape)
    X_test = np.array([data for data in data_all.loc[data_all['Mode'] == 'test']['Data']])
    y_test = data_all.loc[data_all['Mode'] == 'test']['Class'].to_numpy()
    print(X_test.shape)
    print(y_test.shape)

    print("Logistic Regression model training and testing")
    model = LogisticRegression(random_state=0).fit(X_train, y_train)
    filename = 'log_reg_model.pkl'
    pickle.dump(model, open('./saved_models/' + filename, 'wb'))
    print("Logistic Regression model training and saving complete")
    total = 5625
    y_pred = model.predict(X_test)
    print(np.sum(y_pred == 1))

    correct = (y_pred == y_test).sum()
    print('test acc = ', correct / total)
    print('test correct = ', correct)
    print('train acc = ', model.score(X_train, y_train))

    segs_TN, segs_FN, segs_FP, segs_TP = confusion_matrix(y_test, y_pred).reshape(-1)

    # report metrics
    stats_file = open('./records/' + 'seg_stat_log_reg.txt', 'w')
    stats_file.write('segments: TP, FN, FP, TN\n')
    output_segs = stats_report([segs_TP, segs_FN, segs_FP, segs_TN])
    stats_file.write(output_segs + '\n')

    return model


def main():
    # Hyperparameters
    SIZE = args.size  # data point number per data
    path_data = args.path_data
    path_indices = args.path_indices
    # names_list = []
    names_dict = {}

    # loading data files from test_indics.csv and train_indice.csv
    csvdata_train = loadCSV(os.path.join(path_indices, 'train' + '_indice.csv'))
    csvdata_test = loadCSV(os.path.join(path_indices, 'test' + '_indice.csv'))
    # csvdata_all = csvdata_train | csvdata_test

    print("Loading csv file and indices complete")
    path_test = load_name(csvdata_test)
    path_train = load_name(csvdata_train)
    # path_all = path_test | path_train

    data_all = pd.DataFrame()  # 5 columns: 'File Name', 'Data', 'Mode', 'Class', 'Label'
    for i in range(len(list(path_test.keys()))):
        # for i in range(5):
        df = load_data_to_df(path_data, list(path_test.keys()), path_test, 'test', i)
        data_all = pd.concat([data_all, df], ignore_index=True)

    for i in range(len(list(path_train.keys()))):
        # for i in range(5):
        df = load_data_to_df(path_data, list(path_train.keys()), path_train, 'train', i)
        data_all = pd.concat([data_all, df], ignore_index=True)

    print("loading data complete")

    '''
       for all given data:
           total 30213
           0    15987
           1    14226
           train    24588
           test      5625
       for all train data:
           0    12751
           1    11837
       for all test data:
           0    3236
           1    2389
       '''

    print(data_all['Mode'].value_counts())
    print((data_all.loc[data_all['Mode'] == 'train'])['Label'].value_counts(ascending=True))
    print((data_all.loc[data_all['Mode'] == 'test'])['Label'].value_counts(ascending=True))
    print((data_all.loc[data_all['Class'] == 0])['Label'].value_counts(ascending=True))
    print((data_all.loc[data_all['Class'] == 1])['Label'].value_counts(ascending=True))

    # t = np.arange(0,SIZE,1)
    # y_time = data_1['IEGM_seg'].squeeze()
    # y_freq = np.fft.fft(y_time)  # calculate fft on series
    # freq = np.fft.fftfreq(t.shape[-1])  # frequency
    # # how to use np.fft.fft2()?

    return data_all


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

    data_all = main()
    AFt = np.array([data for data in data_all.loc[data_all['Label'] == 'AFt']['Data']])
    SVT = np.array([data for data in data_all.loc[data_all['Label'] == 'SVT']['Data']])
    VPD = np.array([data for data in data_all.loc[data_all['Label'] == 'VPD']['Data']])
    AFb = np.array([data for data in data_all.loc[data_all['Label'] == 'AFb']['Data']])
    SR = np.array([data for data in data_all.loc[data_all['Label'] == 'SR']['Data']])

    VFt = np.array([data for data in data_all.loc[data_all['Label'] == 'VFt']['Data']])
    VFb = np.array([data for data in data_all.loc[data_all['Label'] == 'VFb']['Data']])
    VT = np.array([data for data in data_all.loc[data_all['Label'] == 'VT']['Data']])

    list_0 = [AFt[0], SVT[0], VPD[0], AFb[0], SR[0]]
    list_1 = [VFt[0], VFb[0], VT[0]]
    # data_plot(AFt[0],SIZE)

    data_list_plot(list_0, ['AFT','SVT','VPD','AFb','SR'], 1250, 'AFT,SVT,VPD,AFb,SR')
    data_list_plot(list_1, ['VFt', 'VFb', 'VT'], 1250, 'VFt,VFb,VT')