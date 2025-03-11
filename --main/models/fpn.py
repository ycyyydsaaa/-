import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list: 各层输入通道数列表
            out_channels: FPN 输出通道数
        """
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, features):
        lateral_feats = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]
        for i in range(len(lateral_feats) - 1, 0, -1):
            lateral_feats[i - 1] += F.interpolate(lateral_feats[i],
                                                  size=lateral_feats[i - 1].shape[-2:],
                                                  mode='nearest')
        out_feats = [o_conv(f) for f, o_conv in zip(lateral_feats, self.output_convs)]
        return out_feats