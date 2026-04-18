import torch
import torch.nn as nn
import torch.nn.functional as F
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
def align_to(x, ref):
    th, tw = ref.shape[-2], ref.shape[-1]
    x = x[:, :, :th, :tw]
    if x.shape[-2:] != (th, tw):
        x = F.interpolate(x, size=(th, tw), mode="nearest")
    return x
class Conv(nn.Module):
    default_act = nn.SiLU(inplace=True)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
class PConv(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        pads = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=pads[g]) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.fuse = Conv(c2, c2, k=1, s=1)
    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.fuse(torch.cat([yw0, yw1, yh0, yh1], dim=1))
class SCDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = Conv(in_channels, mid, k=1, s=1)
        self.conv2 = Conv(mid, out_channels, k=3, s=2, g=mid)
    def forward(self, x):
        return self.conv2(self.conv1(x))
def haar(x):
    h, w = x.shape[-2], x.shape[-1]
    if (h & 1) or (w & 1):
        x = F.pad(x, (0, w & 1, 0, h & 1), mode="constant", value=0.0)
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh
class MultiDomainFusion(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.spa_proj = Conv(c, c, k=1, s=1)
        self.freq_proj = Conv(c, c, k=1, s=1)
        self.str_proj = Conv(c, c, k=1, s=1)
        self.fuse = Conv(3 * c, c, k=1, s=1, act=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x_spa, x_freq, x_str):
        x_spa = self.spa_proj(x_spa)
        x_freq = self.freq_proj(x_freq)
        x_str = self.str_proj(x_str)
        y = self.fuse(torch.cat([x_spa, x_freq, x_str], dim=1))
        return y * self.gate(y)
class WaveletFusion(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ll_proj1 = Conv(c, c, k=1, s=1)
        self.ll_proj2 = Conv(c, c, k=1, s=1)
        self.detail_proj = Conv(c, c, k=1, s=1)
        self.recon_proj = Conv(c, c, k=1, s=1)
        self.mix = Conv(4 * c, c, k=1, s=1, act=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        ll1, lh1, hl1, hh1 = haar(x)
        ll2, lh2, hl2, hh2 = haar(ll1)
        detail = torch.abs(lh1) + torch.abs(hl1) + torch.abs(hh1)
        ll1 = align_to(self.ll_proj1(ll1), x)
        ll2 = align_to(self.ll_proj2(ll2), x)
        detail = align_to(self.detail_proj(detail), x)
        fused = self.mix(torch.cat([ll1, ll2, detail], dim=1))
        fused = fused * self.gate(fused)
        return fused
class DCB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.local = nn.Sequential(
            Conv(c, c, k=3, s=1),
            Conv(c, c, k=3, s=1, g=c),
            Conv(c, c, k=1, s=1, act=False),
        )
        self.wave = WaveletFusion(c)
        self.mdf = MultiDomainFusion(c)
        self.norm = nn.InstanceNorm2d(c, affine=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y_local = self.local(x)
        y_wave = self.wave(x)
        y_mdf = self.mdf(x, y_local, y_wave)
        y = self.norm(y_mdf)
        return self.act(y + x)
class CSPA(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3):
        super().__init__()
        assert out_channels % 2 == 0
        hidden = out_channels // 2
        self.down = SCDown(in_channels, out_channels)
        self.pre = Conv(out_channels, out_channels, k=1, s=1)
        self.blocks = nn.ModuleList([DCB(hidden) for _ in range(num_blocks)])
        self.post = Conv(hidden * (num_blocks + 2), out_channels, k=1, s=1)
    def forward(self, x):
        x = self.pre(self.down(x))
        x1, x2 = x.chunk(2, dim=1)
        feats = [x1, x2]
        for block in self.blocks:
            x2 = block(x2)
            feats.append(x2)
        return self.post(torch.cat(feats, dim=1))
class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = PConv(c1=1, c2=16, k=3, s=2)
        self.layer1 = CSPA(16, 64, num_blocks=3)
        self.layer2 = CSPA(64, 96, num_blocks=3)
        self.layer3 = CSPA(96, 128, num_blocks=2)
        self.head = Conv(128, 128, k=1, s=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.25, inplace=False)
        self.classifier = nn.Linear(128, 18)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.head(x)
        features = x
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.classifier(x), features