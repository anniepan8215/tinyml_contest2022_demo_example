import torch.nn as nn
import torch.nn.functional as F


def NiN_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(True))


class IEGMNet_FFT(nn.Module):
    def __init__(self):
        super(IEGMNet_FFT, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=6, kernel_size=(6, 1), stride=(2, 1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(6, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=10, kernel_size=(5, 1), stride=(2, 1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(4, 1), stride=(2, 1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(4, 1), stride=(2, 1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(40, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(4, 1), stride=(2, 1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(40, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=40*37*1, out_features=10)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=10, out_features=2)
        )

        self.global_avg = nn.Sequential(
            nn.Dropout(0.5),
            NiN_block(in_channels=40, out_channels=2, kernel_size=(4, 1), stride=(2, 1), padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # self.net = nn.Sequential(
        #     NiN_block(in_channels=1, out_channels=3, kernel_size=(6, 1), stride=(2,1), padding=(1,1)),
        #     nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        #     nn.MaxPool2d((3,1),stride=(2,1)),
        #     NiN_block(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(2,1), padding=(1,1)),
        #     nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        #     nn.MaxPool2d((3, 1), stride=(2, 1)),
        #     NiN_block(in_channels=5, out_channels=10, kernel_size=(4, 1), stride=(2,1), padding=(1,1)),
        #     nn.BatchNorm2d(10, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        #     nn.MaxPool2d((3, 1), stride=(2, 1)),
        #     NiN_block(in_channels=10, out_channels=20, kernel_size=(4, 1), stride=(2,1), padding=(1,1)),
        #     nn.BatchNorm2d(20, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
        #     nn.MaxPool2d((3, 1), stride=(2, 1)),
        #     nn.Dropout(0.5),
        #     NiN_block(in_channels=20, out_channels=2, kernel_size=(4, 1), stride=(2, 1), padding=(1,1)),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten()
        # )

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        conv5_output = conv5_output.view(-1, 40*37*1)

        fc1_output = F.relu(self.fc1(conv5_output))
        output = self.fc2(fc1_output)
        # output = self.global_avg(conv4_output)
        return output
