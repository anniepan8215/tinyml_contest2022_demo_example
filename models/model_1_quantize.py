import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvReLUBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, groups=1):
        super(ConvReLUBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, 0, groups=groups, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(out_planes, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1)
        )





class IEGMNet_FFT(nn.Module):

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvReLUBN:
                torch.ao.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)

    def __init__(self):
        super(IEGMNet_FFT, self).__init__()

        self.quant = torch.quantization.QuantStub()
        self.conv1 = ConvReLUBN(2, 6, kernel_size=(6, 1), stride=(2, 1))
        self.conv2 = ConvReLUBN(6, 10, kernel_size=(5, 1), stride=(2, 1))
        self.conv3 = ConvReLUBN(10, 20, kernel_size=(4, 1), stride=(2, 1))
        self.conv4 = ConvReLUBN(20, 40, kernel_size=(4, 1), stride=(2, 1))
        self.conv5 = ConvReLUBN(40, 40, kernel_size=(4, 1), stride=(2, 1))

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=40 * 37 * 1, out_features=10)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=10, out_features=2)
        )

        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input):
        quant_output = self.quant(input)
        conv1_output = self.conv1(quant_output)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.view(-1, 40 * 37 * 1)

        fc1_output = F.relu(self.fc1(conv5_output))
        output = F.softmax(self.fc2(fc1_output))
        output_dequant = self.dequant(output)
        # output = self.global_avg(conv4_output)
        return output_dequant
