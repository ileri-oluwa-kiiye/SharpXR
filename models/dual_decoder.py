import torch.nn as nn
from .components import DoubleConv, DecoderBlock, LearnableFusion


class DualDecoderHybrid(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.encoder = nn.ModuleList()
        enc_in_channels = in_channels
        for feature in features:
            self.encoder.append(DoubleConv(enc_in_channels, feature))
            enc_in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Two decoders
        in_ch_list = [features[-1]*2] + list(reversed(features))[:-1]
        out_ch_list = list(reversed(features))

        self.decoder_denoise = DecoderBlock(in_ch_list, out_ch_list)
        self.decoder_edge = DecoderBlock(in_ch_list, out_ch_list)

        # Learnable fusion
        self.learnable_fusion = LearnableFusion(features[0])
        self.final_out = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        denoise_out = self.decoder_denoise(x, skip_connections)
        edge_out = self.decoder_edge(x, skip_connections, laplacian=True)

        fused = self.learnable_fusion(denoise_out, edge_out)
        return self.final_out(fused)