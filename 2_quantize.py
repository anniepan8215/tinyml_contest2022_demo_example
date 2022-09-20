import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import torchvision.transforms as transforms

from help_code_demo import ToTensor, stats_report, IEGM_DataSET
import time

def fft_transfer(ys_time, SIZE=1250):
    ys_freq = np.zeros((ys_time.shape[0],SIZE))
    ys_time = ys_time.squeeze()
    if len(list(ys_time.size())) == 1:
        ys_freq = np.fft.fft(ys_time)
    else:
        for i in range(ys_time.size(dim=0)):
            y_freq = np.fft.fft(ys_time[i, :])  # calculate fft on series
            ys_freq[i] = y_freq
    return torch.tensor(ys_freq.reshape((ys_time.shape[0], 1, SIZE, 1)))


def main():
    seed = 222
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Hyperparameters
    BATCH_SIZE_TEST = 1
    SIZE = args.size
    TH = args.th
    path_data = args.path_data
    path_records = args.path_record
    path_net = args.path_net
    path_indices = args.path_indices
    stats_file = open(path_records + '2_quantize.txt', 'w')

    # load trained network
    net = torch.load(path_net + 'IEGM_net_quantize.pkl', map_location='cpu')

    net.eval()
    # Dynamic PTQ
    # net = torch.quantization.quantize_dynamic(
    #     net,
    #     {torch.nn.Linear}
    # )

    # Statistic PTQ
    net.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    net.fuse_model()
    net_prepared = torch.quantization.prepare(net)

    device = torch.device('cpu')

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)

    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0
    # breakpoint()
    start = time.time()
    for data_test in testloader:
        IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
        seg_label = deepcopy(labels_test)
        IEGM_test = torch.cat((IEGM_test, fft_transfer(IEGM_test)), 1)
        IEGM_test = IEGM_test.float()

        net_prepared(IEGM_test)
        net_int8 = torch.quantization.convert(net_prepared)
        outputs_test = net_int8(IEGM_test)
        predicted_test = (outputs_test.data[:,1] > TH)

        if seg_label == 0:
            segs_FP += (labels_test.size(0) - (predicted_test == labels_test).sum()).item()
            segs_TN += (predicted_test == labels_test).sum().item()
        elif seg_label == 1:
            segs_FN += (labels_test.size(0) - (predicted_test == labels_test).sum()).item()
            segs_TP += (predicted_test == labels_test).sum().item()

    # report metrics
    stats_file.write('segments: TP, FN, FP, TN\n')
    output_segs = stats_report([segs_TP, segs_FN, segs_FP, segs_TN])
    stats_file.write(output_segs + '\n')
    stop = time.time()
    interval = stop - start
    print('interval: '+str(interval))
    torch.save(net, './saved_models/1_after_quant.pkl')
    del net


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--th', type=float, help='threshold for label smoothing', default=0.44)
    # argparser.add_argument('--path_data', type=str, default='H:/Date_Experiment/data_IEGMdb_ICCAD_Contest/segments-R250'
    #                                                         '-BPF15_55-Noise/tinyml_contest_data_training/')
    argparser.add_argument('--path_data', type=str, default='./data/')
    argparser.add_argument('--path_net', type=str, default='./saved_models/')
    argparser.add_argument('--path_record', type=str, default='./records/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')

    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))
    print("device is --------------", device)

    main()
