import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, channels, rates=(1, 2, 3)):
        super().__init__()
        mid = channels // 2
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True))
        self.atrous = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, mid, 3, padding=r, dilation=r),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True))
            for r in rates])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True))
        num_branches = 1 + len(rates) + 1  # 1x1 + atrous branches + GAP
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * num_branches, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [self.conv1x1(x)]
        for atrous_conv in self.atrous:
            feats.append(atrous_conv(x))
        gap = self.gap(x)
        feats.append(F.interpolate(gap, size=(h, w), mode='bilinear', align_corners=True))
        return self.fuse(torch.cat(feats, dim=1))


class CoordAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mid, 1)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, 1)
        self.conv_w = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        ah = self.pool_h(x)
        aw = self.pool_w(x).permute(0, 1, 3, 2)
        combined = torch.cat([ah, aw], dim=2)
        combined = self.act(self.bn(self.conv1(combined)))
        ah, aw = combined.split([h, w], dim=2)
        ah = self.conv_h(ah).sigmoid()
        aw = self.conv_w(aw.permute(0, 1, 3, 2)).sigmoid()
        return x * ah * aw


class LargeKernelBlock(nn.Module):
    def __init__(self, channels, kernel_size=13):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size,
                            padding=kernel_size // 2, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x)))) + x


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
        self.down1 = nn.MaxPool2d(2)
        self.block2 = conv_block(base_width, base_width * 2)
        self.down2 = nn.MaxPool2d(2)
        self.block3 = conv_block(base_width * 2, base_width * 4)
        self.down3 = nn.MaxPool2d(2)
        self.block4 = conv_block(base_width * 4, base_width * 8)
        self.lk4 = LargeKernelBlock(base_width * 8, kernel_size=13)
        self.down4 = nn.MaxPool2d(2)
        self.block5 = conv_block(base_width * 8, base_width * 8)
        self.lk5 = LargeKernelBlock(base_width * 8, kernel_size=7)
        self.down5 = nn.MaxPool2d(2)
        self.block6 = conv_block(base_width * 8, base_width * 8)
        self.aspp = ASPP(base_width * 8, rates=(1, 2, 3))

        # Encoder CoordAttention on skip connections
        self.ca1 = CoordAttention(base_width)
        self.ca2 = CoordAttention(base_width * 2)
        self.ca3 = CoordAttention(base_width * 4)
        self.ca4 = CoordAttention(base_width * 8)
        self.ca5 = CoordAttention(base_width * 8)

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(self.down1(b1))
        b3 = self.block3(self.down2(b2))
        b4 = self.lk4(self.block4(self.down3(b3)))
        b5 = self.lk5(self.block5(self.down4(b4)))
        b6 = self.aspp(self.block6(self.down5(b5)))

        b1 = self.ca1(b1)
        b2 = self.ca2(b2)
        b3 = self.ca3(b3)
        b4 = self.ca4(b4)
        b5 = self.ca5(b5)

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

        self.up_b = up_block(bw * 8, bw * 8)
        self.se_b = CoordAttention(bw * 16)
        self.db_b = fuse_block(bw * 16, bw * 8)

        self.up1 = up_block(bw * 8, bw * 4)
        self.se1 = CoordAttention(bw * 12)
        self.db1 = fuse_block(bw * 12, bw * 4)

        self.up2 = up_block(bw * 4, bw * 2)
        self.se2 = CoordAttention(bw * 6)
        self.db2 = fuse_block(bw * 6, bw * 2)

        self.up3 = up_block(bw * 2, bw)
        self.se3 = CoordAttention(bw * 3)
        self.db3 = fuse_block(bw * 3, bw)

        self.up4 = up_block(bw, bw)
        self.se4 = CoordAttention(bw * 2)
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
