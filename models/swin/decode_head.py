import torch
from torch import nn


class UpSamplingModule(nn.Module):

    def __init__(self, scale_factor=1, channel_in=2048, channel_out=512, kernel_size=3, middle_dilation=2):
        super(UpSamplingModule, self).__init__()
        padding_norm = kernel_size // 2
        padding_dilation = ((kernel_size * middle_dilation) - (middle_dilation - 1)) // 2
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=1,
                      padding=padding_norm, groups=1, bias=False, dilation=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
            nn.Conv2d(channel_out, channel_out, kernel_size=kernel_size, stride=1,
                      padding=padding_dilation, groups=1, bias=False, dilation=middle_dilation),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
            nn.Conv2d(channel_out, channel_out, kernel_size=kernel_size, stride=1,
                      padding=padding_norm, groups=1, bias=False, dilation=1),
        )

    def forward(self, x):
        x = self.up_conv(x)
        return x


class DecodeHead(nn.Module):

    def __init__(self, channel_in_list, kernel_size_list=None, class_num=2, output_up_scale=4):
        super(DecodeHead, self).__init__()

        if not kernel_size_list:
            kernel_size_list = [3, 5, 5, 1]

        assert len(channel_in_list) == len(kernel_size_list)

        up_network = list()
        for i in range(len(channel_in_list) - 1):
            print(channel_in_list[i], channel_in_list[i + 1])
            atom = nn.Sequential(
                UpSamplingModule(2, channel_in_list[i], channel_in_list[i + 1], kernel_size_list[i]),
                nn.BatchNorm2d(channel_in_list[i + 1]),
                nn.ReLU()
            )
            up_network.append(atom)
        self.up_network = nn.ModuleList(up_network)
        self.output = nn.Sequential(
            nn.Upsample(scale_factor=output_up_scale, mode='bilinear', align_corners=True),
            nn.Conv2d(channel_in_list[-1], class_num, kernel_size=kernel_size_list[-1],
                      stride=1, padding=kernel_size_list[-1] // 2, groups=1, bias=False, dilation=1)
        )

    def forward(self, feature_list):
        x = feature_list[-1]
        net_idx = 0
        for idx in range(len(self.up_network) - 1, -1, -1):
            x = self.up_network[net_idx](x) + feature_list[idx]
            net_idx += 1
        x = self.output(x)
        return x
