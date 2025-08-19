import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for in_c, out_c in zip(in_channels_list, out_channels_list):
            self.upconvs.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(out_c * 2, out_c))

    def forward(self, x, skip_connections, laplacian=False):
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]

            if laplacian:
                edge = self.laplacian_filter(skip)
                skip = skip + edge

            if x.shape != skip.shape:
                x = TF.resize(x, skip.shape[2:])

            x = torch.cat((skip, x), dim=1)
            x = self.decoders[idx](x)
        return x

    def laplacian_filter(self, x):
        kernel = torch.tensor([[[[-1, -1, -1],
                                 [-1,  8, -1],
                                 [-1, -1, -1]]]], dtype=torch.float32, device=x.device)
        kernel = kernel.expand(x.size(1), 1, 3, 3)
        edge = F.conv2d(x, kernel, padding=1, groups=x.size(1))
        return edge


class LearnableFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, feat1, feat2):
        combined = torch.cat([feat1, feat2], dim=1)
        weights = self.attn(combined)
        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]
        return w1 * feat1 + w2 * feat2