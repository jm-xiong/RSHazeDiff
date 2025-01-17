import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), -1)

        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        real_part = ffted[..., 0]
        imaginary_part = ffted[..., 1]
        complex_tensor = real_part + 1j * imaginary_part
        output = torch.fft.irfft2(complex_tensor, dim=(-2, -1), s=r_size[2:], norm="ortho")

        return output

##########################################################################
##---------- Fast Fourier Block-----------------------
class FFB(nn.Module):
    def __init__(self, nc):
        super(FFB, self).__init__()
        self.block = nn.Sequential(
                        nn.Conv2d(nc,nc,3,1,1),
                        nn.LeakyReLU(0.1),
                        nn.Conv2d(nc, nc, 3, 1, 1)
        )
        self.glocal = FourierUnit(nc, nc)
        self.cat = nn.Conv2d(2*nc, nc, 1, 1, 0)

    def forward(self, x):
        conv = self.block(x)
        conv = conv + x
        glocal = self.glocal(x)
        out = torch.cat([conv, glocal], 1)
        out = self.cat(out) + x
        return out


class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [FFB(nc=dim) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

##########################################################################
##---------- Resizing Modules ----------
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        return self.bot(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()

        modules_body = []
        modules_body.append(Down(in_channels, out_channels))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        )

    def forward(self, x):
        return self.bot(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()

        modules_body = []
        modules_body.append(Up(in_channels, out_channels))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out

class Global_Local_Fusion(nn.Module):
    def __init__(self, in_chans=3, feat=24, out_chans=3):
        super(Global_Local_Fusion, self).__init__()

        self.conv_in = nn.Conv2d(in_chans, feat, 1, 1)

        self.pa_att = PALayer(channel=feat)
        self.cha_att = CALayer(channel=feat*2)
        self.post = nn.Conv2d(feat*2, feat, 1, 1)
        self.conv_out = nn.Conv2d(feat, out_chans, kernel_size=1, padding=0)


    def forward(self, diff_img, global_img):

        diff_img = self.conv_in(diff_img)
        global_img = self.conv_in(global_img)
        res_img = diff_img + global_img
        pa_map = self.pa_att(res_img)
        pa_res = pa_map

        cat_f = torch.cat([diff_img, global_img], 1)
        cha_res = self.post(self.cha_att(cat_f)) + pa_res
        out_img = self.conv_out(cha_res)   #ours

        return out_img

class GlobalNet_Fusion(nn.Module):
    def __init__(self, in_chans=3, out_chans=3,
                 embed_dims=[24, 48, 96, 48, 24], #24, 48, 96, 48, 24
                 depths=[1, 1, 2, 1, 1]):   #1, 1, 2, 1, 1
        super(GlobalNet_Fusion, self).__init__()

        self.conv1 = nn.Conv2d(in_chans, embed_dims[0], kernel_size=3, padding=1)

        # backbone
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])

        self.down1 = DownSample(embed_dims[0], embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])

        self.down2 = DownSample(embed_dims[1], embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])

        self.up1 = UpSample(embed_dims[2], embed_dims[3])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.up2 = UpSample(embed_dims[3], embed_dims[4])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        self.conv2 = nn.Conv2d(embed_dims[4], out_chans, kernel_size=3, padding=1)

        self.global_local_fusion = Global_Local_Fusion(feat=96)

    def forward_features(self, x):
        x = self.conv1(x)   # 3*3Conv
        x = self.layer1(x)
        skip1 = x

        x = self.down1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.down2(x)
        x = self.layer3(x)
        x = self.up1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.up2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.conv2(x)
        return x

    def forward(self, x, diff_img):

        global_img = self.forward_features(x)

        out_img = self.global_local_fusion(diff_img, global_img)

        return out_img

