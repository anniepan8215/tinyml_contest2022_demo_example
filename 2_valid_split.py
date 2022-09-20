import argparse
import time
from copy import deepcopy

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from help_code_demo import ToTensor, FB, IEGM_DataSET
from models.model_1_quantize import IEGMNet_FFT

from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


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
    # Hyperparameters
    seed = 222
    torch.manual_seed(seed)
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    TH = args.th
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    validation_split = args.path_vtr
    validation_step = args.valid_step

    # Instantiating NN
    net = IEGMNet_FFT()
    net.train()
    net = net.float().to(device)

    # Start dataset loading
    trainset = IEGM_DataSET(root_dir=path_data,
                                indice_dir=path_indices,
                                mode='train',
                                size=SIZE,
                                transform=transforms.Compose([ToTensor()]))

    valid_size = int(validation_split * len(trainset))
    train_size = len(trainset) - valid_size

    train_dataset, valid_dataset = torch.utils.data.random_split(trainset, [train_size, valid_size])

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print("Training Dataset loading finish.")

    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    epoch_num = EPOCH

    Train_loss = []
    Train_acc = []
    # Test_loss = []
    # Test_acc = []
    FB_scores = []
    Valid_loss = []
    Valid_acc = []
    min_valid_loss = np.inf
    max_FB = -np.inf
    start = time.time()

    print("Start training")
    for epoch in range(epoch_num):  # loop over the dataset multiple times (specify the #epoch)

        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        for j, data in enumerate(trainloader, 0):
            inputs, labels = data['IEGM_seg'], data['label']
            inputs = torch.cat((inputs, fft_transfer(inputs)),1)
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = (outputs.data[:, 1] > TH).float()
            correct += (predicted == labels).sum()
            accuracy += correct / BATCH_SIZE
            correct = 0.0

            running_loss += loss.item()
            i += 1

        print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
              (epoch + 1, i, accuracy / i, running_loss / i))

        Train_loss.append(running_loss / i)
        Train_acc.append((accuracy / i).item())

        # running_loss = 0.0
        # accuracy = 0.0

        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_valid = 0.0

        if epoch % validation_step == 0:
            net.eval()
            segs_TP = 0
            segs_TN = 0
            segs_FP = 0
            segs_FN = 0
            with torch.no_grad():
                for data_valid in validloader:
                    IEGM_valid, labels_valid = data_valid['IEGM_seg'], data_valid['label']
                    IEGM_valid = torch.cat((IEGM_valid, fft_transfer(IEGM_valid)), 1)
                    IEGM_valid = IEGM_valid.float().to(device)
                    labels_valid = labels_valid.to(device)
                    outputs_valid = net(IEGM_valid)
                    predicted_valid = (outputs_valid.data[:, 1] > TH).float()
                    total += labels_valid.size(0)
                    correct += (predicted_valid == labels_valid).sum()

                    seg_labels = deepcopy(labels_valid)
                    for seg_label in seg_labels:
                        if seg_label == 0:
                            segs_FP += (labels_valid.size(0) - (predicted_valid == labels_valid).sum()).item()
                            segs_TN += (predicted_valid == labels_valid).sum().item()
                        elif seg_label == 1:
                            segs_FN += (labels_valid.size(0) - (predicted_valid == labels_valid).sum()).item()
                            segs_TP += (predicted_valid == labels_valid).sum().item()


                    loss_valid = criterion(outputs_valid, labels_valid)
                    running_loss_valid += loss_valid.item()
                    i += 1

                # report metrics
                print('Valid Acc: %.5f Valid Loss: %.5f' % (correct / total, running_loss_valid / i))
                FB_score = FB([segs_TP, segs_FN, segs_FP, segs_TN])
                print('FB score: %.5f' % (FB_score))

                Valid_loss.append(running_loss_valid / i)
                Valid_acc.append((correct / total).item())
                FB_scores.append(FB_score)
                # if min_valid_loss > running_loss_valid / i:
                #     min_valid_loss = running_loss_valid / i
                #     torch.save(net, './saved_models/IEGM_net_quantize.pkl')
                #     torch.save(net.state_dict(), './saved_models/IEGM_net_quantize_state_dict.pkl')
                # Save the model with highest FB score
                if max_FB < FB_score:
                    max_FB = FB_score
                    torch.save(net, './saved_models/IEGM_net_quantize.pkl')
                    torch.save(net.state_dict(), './saved_models/IEGM_net_quantize_state_dict.pkl')

    stop = time.time()
    total_time = stop - start
    print("Total training time:" + str(total_time) + 's')

    file = open('./saved_models/loss_acc_quantize.txt', 'w')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))
    file.write('\n\n')
    file.write("Valid_loss\n")
    file.write(str(Valid_loss))
    file.write('\n\n')
    file.write("Valid_acc\n")
    file.write(str(Valid_acc))
    file.write('\n\n')
    file.write("FBeta Score\n")
    file.write(str(FB_scores))
    file.write('\n\n')
    file.write("Total training time\n")
    file.write(str(total_time))
    file.write('\n\n')
    # plot_against_epoch_numbers(train_epoch_and_value_pairs=Train_loss, validation_epoch_and_value_pairs=Valid_loss,
    #                            train_label='training loss', val_label='validation loss', title='Loss Plot')
    # plot_against_epoch_numbers(train_epoch_and_value_pairs=Train_acc, validation_epoch_and_value_pairs=Valid_acc,
    #                            train_label='training accuracy', val_label='validation accuracy', title='Accuracy Plot')

    print('Finish training')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    argparser.add_argument('--th', type=float, help='threshold for label smoothing', default=0.6)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    # argparser.add_argument('--path_data', type=str, default='H:/Date_Experiment/data_IEGMdb_ICCAD_Contest/segments-R250'
    #                                                         '-BPF15_55-Noise/tinyml_contest_data_training/')
    argparser.add_argument('--path_data', type=str, default='./data/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')
    argparser.add_argument('--path_vtr', type=float, help='Validate train ratio', default=0.2)
    argparser.add_argument('--valid_step', type=int, help='number of epoch for evaluation', default=1)

    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))

    print("device is --------------", device)

    main()
