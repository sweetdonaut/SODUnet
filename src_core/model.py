import torch
import torch.nn as nn


class SPPF(nn.Module):
    def __init__(self, channels, pool_size=5):
        super().__init__()
        mid = channels // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid * 4, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.conv2(torch.cat([x, p1, p2, p3], dim=1))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class EncoderSegmentation(nn.Module):
    def __init__(self, in_channels, base_width):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))

        self.block1 = conv_block(in_channels, base_width)
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = conv_block(base_width, base_width * 2)
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = conv_block(base_width * 2, base_width * 4)
        self.mp3 = nn.MaxPool2d(2)
        self.block4 = conv_block(base_width * 4, base_width * 8)
        self.mp4 = nn.MaxPool2d(2)
        self.block5 = conv_block(base_width * 8, base_width * 8)
        self.mp5 = nn.MaxPool2d(2)
        self.block6 = conv_block(base_width * 8, base_width * 8)
        self.sppf = SPPF(base_width * 8)

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(self.mp1(b1))
        b3 = self.block3(self.mp2(b2))
        b4 = self.block4(self.mp3(b3))
        b5 = self.block5(self.mp4(b4))
        b6 = self.sppf(self.block6(self.mp5(b5)))
        return b1, b2, b3, b4, b5, b6


class DecoderSegmentation(nn.Module):
    def __init__(self, base_width, out_channels=2):
        super().__init__()

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))

        def fuse_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))

        bw = base_width

        # up: reduce channels, fuse: single conv after SE(cat)
        self.up_b = up_block(bw * 8, bw * 8)
        self.se_b = SEBlock(bw * 16)
        self.db_b = fuse_block(bw * 16, bw * 8)

        self.up1 = up_block(bw * 8, bw * 4)
        self.se1 = SEBlock(bw * 12)
        self.db1 = fuse_block(bw * 12, bw * 4)

        self.up2 = up_block(bw * 4, bw * 2)
        self.se2 = SEBlock(bw * 6)
        self.db2 = fuse_block(bw * 6, bw * 2)

        self.up3 = up_block(bw * 2, bw)
        self.se3 = SEBlock(bw * 3)
        self.db3 = fuse_block(bw * 3, bw)

        self.up4 = up_block(bw, bw)
        self.se4 = SEBlock(bw * 2)
        self.db4 = fuse_block(bw * 2, bw)

        self.fin_out = nn.Conv2d(bw, out_channels, 3, padding=1)

    def forward(self, b1, b2, b3, b4, b5, b6):
        d5 = self.db_b(self.se_b(torch.cat([self.up_b(b6), b5], dim=1)))
        d4 = self.db1(self.se1(torch.cat([self.up1(d5), b4], dim=1)))
        d3 = self.db2(self.se2(torch.cat([self.up2(d4), b3], dim=1)))
        d2 = self.db3(self.se3(torch.cat([self.up3(d3), b2], dim=1)))
        d1 = self.db4(self.se4(torch.cat([self.up4(d2), b1], dim=1)))
        return self.fin_out(d1)


class SegmentationNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, base_channels=64):
        super().__init__()
        self.encoder = EncoderSegmentation(in_channels, base_channels)
        self.decoder = DecoderSegmentation(base_channels, out_channels=out_channels)

    def forward(self, x):
        b1, b2, b3, b4, b5, b6 = self.encoder(x)
        return self.decoder(b1, b2, b3, b4, b5, b6)
