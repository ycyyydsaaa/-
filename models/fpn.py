import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        self.spatial_attentions = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            self.se_blocks.append(SEBlock(out_channels))
            self.spatial_attentions.append(SpatialAttention())

        self.gradients = None


    def forward(self, features):
        lateral_feats = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]
        for i in range(len(lateral_feats) - 1, 0, -1):
            lateral_feats[i - 1] += F.interpolate(lateral_feats[i],
                                                  size=lateral_feats[i - 1].shape[-2:],
                                                  mode='nearest')
        out_feats = [o_conv(f) for f, o_conv in zip(lateral_feats, self.output_convs)]
        out_feats = [self.se_blocks[i](out) for i, out in enumerate(out_feats)]
        out_feats = [self.spatial_attentions[i](out) for i, out in enumerate(out_feats)]
        # 在 forward 中注册钩子到最终特征图
        if out_feats[-1].requires_grad:
            out_feats[-1].register_hook(self.activations_hook)
        return out_feats

    def activations_hook(self, grad):
        self.gradients = grad

    def register_hook(self):
        for conv in self.output_convs:
            conv.weight.register_hook(self.activations_hook)  # 注册到权重上