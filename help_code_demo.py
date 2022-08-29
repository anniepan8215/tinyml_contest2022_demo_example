import csv, torch, os
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sg


def ACC(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    total = sum(mylist)
    acc = (tp + tn) / total
    return acc


def PPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then ppv should be 1
    if tp + fn == 0:
        ppv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tp + fp == 0 and tp + fn != 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    return ppv


def NPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then npv should be 1
    if tn + fp == 0:
        npv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tn + fn == 0 and tn + fp != 0:
        npv = 0
    else:
        npv = tn / (tn + fn)
    return npv


def Sensitivity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then sen should be 1
    if tp + fn == 0:
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity


def Specificity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then spe should be 1
    if tn + fp == 0:
        specificity = 1
    else:
        specificity = tn / (tn + fp)
    return specificity


def BAC(mylist):
    sensitivity = Sensitivity(mylist)
    specificity = Specificity(mylist)
    b_acc = (sensitivity + specificity) / 2
    return b_acc


def F1(mylist):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def FB(mylist, beta=2):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (1 + beta ** 2) * (precision * recall) / ((beta ** 2) * precision + recall)
    return f1


def stats_report(mylist):
    # segs_TP, segs_FN, segs_FP, segs_TN
    f1 = round(F1(mylist), 5)
    fb = round(FB(mylist), 5)
    se = round(Sensitivity(mylist), 5)
    sp = round(Specificity(mylist), 5)
    bac = round(BAC(mylist), 5)
    acc = round(ACC(mylist), 5)
    ppv = round(PPV(mylist), 5)
    npv = round(NPV(mylist), 5)

    output = str(mylist) + '\n' + \
             "F-1 = " + str(f1) + '\n' + \
             "F-B = " + str(fb) + '\n' + \
             "SEN = " + str(se) + '\n' + \
             "SPE = " + str(sp) + '\n' + \
             "BAC = " + str(bac) + '\n' + \
             "ACC = " + str(acc) + '\n' + \
             "PPV = " + str(ppv) + '\n' + \
             "NPV = " + str(npv) + '\n'

    print("F-1 = ", F1(mylist))
    print("F-B = ", FB(mylist))
    print("SEN = ", Sensitivity(mylist))
    print("SPE = ", Specificity(mylist))
    print("BAC = ", BAC(mylist))
    print("ACC = ", ACC(mylist))
    print("PPV = ", PPV(mylist))
    print("NPV = ", NPV(mylist))

    return output


def loadCSV(csvf):
    """
    return a dict saving the information of csv
    :param splitFile: csv file name
    :return: {label:[file1, file2 ...]}
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels


def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat


class ToTensor(object):
    def __call__(self, sample):
        text = sample['IEGM_seg']
        return {
            'IEGM_seg': torch.from_numpy(text),
            'label': sample['label']
        }


class IEGM_DataSET():
    def __init__(self, root_dir, indice_dir, mode, size, transform=None):
        self.root_dir = root_dir  # argparser.add_argument('--path_data', type=str, default=...)
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]

        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'IEGM_seg': IEGM_seg, 'label': label}

        return sample


def pytorch2onnx(net_path, net_name, size):
    net = torch.load(net_path, map_location=torch.device('cpu'))

    dummy_input = torch.randn(1, 1, size, 1)

    optName = str(net_name) + '.onnx'
    torch.onnx.export(net, dummy_input, optName, verbose=True)


def fft_transfer(ys_time, SIZE=1250):
    ys_freq = []
    ys_time = ys_time.squeeze()
    if len(list(ys_time.size())) == 1:
        ys_freq.append(np.fft.fft(ys_time))
    else:
        for i in range(ys_time.size(dim=0)):
            y_freq = np.fft.fft(ys_time[i, :])  # calculate fft on series
            ys_freq.append(y_freq)
    return torch.tensor(np.array(ys_freq).reshape((-1, 1, SIZE, 1)))


def plot_against_epoch_numbers(train_epoch_and_value_pairs, validation_epoch_and_value_pairs=None, train_label=None,
                               val_label=None, title=None, result_reg_path='./records/'):
    """
    Helper to reduce code duplication when plotting quantities that vary over training epochs
    epoch_and_value_pairs: An array_like consisting of pairs of the form (<epoch number>, <value of thing to plot>)
    kwargs are forwarded to matplotlib.pyplot.plot
    """

    assert train_epoch_and_value_pairs is not None
    assert train_label is not None
    assert title is not None
    if validation_epoch_and_value_pairs is None:
        array = np.array(train_epoch_and_value_pairs)
        plt.figure()
        plt.plot(array[:, 0], array[:, 1], label=train_label)
        plt.xlabel("epochs")
        plt.title(title)
    else:
        assert val_label is not None
        train_array = np.array(train_epoch_and_value_pairs)
        val_array = np.array(validation_epoch_and_value_pairs)
        plt.figure()
        plt.plot(train_array[:, 0], train_array[:, 1], label=train_label)
        plt.plot(val_array[:, 0], val_array[:, 1], label=val_label)
        plt.xlabel("epochs")
        plt.title(title)
    plt.savefig(os.path.join(result_reg_path, title + '.png'))


def get_local_max(x):
    """
    Finding indexes for all local maxima
    :param x: data with 1*1250
    :return: index list
    """
    idxs = sg.argrelmax(x)
    return idxs
