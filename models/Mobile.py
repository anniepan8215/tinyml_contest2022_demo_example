import torch
import torch.nn as nn


class Depthwise_Conv(nn.Module):
    def __init__(self, in_fts, stride=(1, 1)):
        super(Depthwise_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, in_fts, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=in_fts),
            nn.BatchNorm2d(in_fts),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x


class Pointwise_Conv(nn.Module):
    def __init__(self, in_fts, out_fts):
        super(Pointwise_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, out_fts, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_fts),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x


class Depthwise_Separable_Conv(nn.Module):
    def __init__(self, in_fts, out_fts, stride=(1, 1)):
        super(Depthwise_Separable_Conv, self).__init__()
        self.dw = Depthwise_Conv(in_fts=in_fts, stride=stride)
        self.pw = Pointwise_Conv(in_fts=in_fts, out_fts=out_fts)

    def forward(self, input_image):
        x = self.pw(self.dw(input_image))
        return x
