import torch.nn as nn
import torch.nn.functional as F
import torch

"""J. Cui et al., "Subject-Independent Drowsiness Recognition from Single-Channel EEG with an Interpretable CNN-LSTM 
model," 2021 International Conference on Cyberworlds (CW), 2021, pp. 201-208, doi: 10.1109/CW52790.2021.00041. """


class CNNLSTM(nn.Module):
    """
    The codes implement the CNN model proposed in the paper "Subject-Independent Drowsiness Recognition from Single-Channel EEG with an Interpretable CNN-LSTM model".
    The network is designed to classify 1D drowsy and alert EEG signals for the purposed of driver drowsiness recognition.

    """

    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.feature = 32
        self.padding = nn.ReplicationPad2d((31, 32, 0, 0))
        self.conv = nn.Conv2d(1, self.feature, (1, 64))
        self.batch = Batchlayer(self.feature)
        self.avgpool = nn.AvgPool2d((1, 8))
        self.fc = nn.Linear(32, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        self.softmax1 = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(32, 2)

    def forward(self, source):
        # print("source shape:", source.shape)
        source = source.permute(0,1,3,2)
        # print("source shape:", source.shape)
        source = self.padding(source)
        # print("padding shape:",source.shape)
        source = self.conv(source)
        # print("conv shape:" , source.shape)
        source = self.batch(source)
        # print("batchnorm shape:" , source.shape)

        source = nn.ELU()(source)
        source = self.avgpool(source)
        # print("avgpool shape:" , source.shape)
        source = source.squeeze(2)
        # print("squeezed shape:", source.shape)
        source = source.permute(2, 0, 1)
        # print("permuted shape:", source.shape)
        source = self.lstm(source)
        # print("lstm shape:", source[1][0].shape)
        source = source[1][0].squeeze(0)
        # print("squeezed shape:", source.shape)
        source = self.softmax(source)

        return source

    """
    We use the batch normalization layer implemented by ourselves for this model instead using the one provided by the Pytorch library.
    In this implementation, we do not use momentum and initialize the gamma and beta values in the range (-0.1,0.1). 
    We have got slightly increased accuracy using our implementation of the batch normalization layer.
    """


def normalizelayer(data):
    eps = 1e-05
    a_mean = data - torch.mean(data, [0, 2, 3], True).expand(int(data.size(0)), int(data.size(1)),
                                                             int(data.size(2)), int(data.size(3)))
    b = torch.div(a_mean, torch.sqrt(torch.mean((a_mean) ** 2, [0, 2, 3], True) + eps).expand(int(data.size(0)),
                                                                                              int(data.size(1)),
                                                                                              int(data.size(2)),
                                                                                              int(data.size(3))))

    return b


class Batchlayer(nn.Module):
    def __init__(self, dim):
        super(Batchlayer, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, dim, 1, 1))
        self.beta = torch.nn.Parameter(torch.Tensor(1, dim, 1, 1))
        self.gamma.data.uniform_(-0.1, 0.1)
        self.beta.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        data = normalizelayer(input)
        gammamatrix = self.gamma.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))
        betamatrix = self.beta.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))

        return data * gammamatrix + betamatrix
