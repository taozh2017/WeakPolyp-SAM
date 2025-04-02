import torch
import torch.nn as nn
import torch.nn.functional as F

from .res2net_v1b_base import Res2Net_model


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Cross_fusion(nn.Module):

    def __init__(self, in_channels1, in_channels2, out_channels):
        super(Cross_fusion, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels1, out_channels, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(out_channels), self.relu)
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels2, out_channels, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(out_channels), self.relu)

        self.layer_3 = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(out_channels), self.relu)
        self.layer_4 = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(out_channels), self.relu)

        self.conv_1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        ###+
        x2 = self.up_2(x2)

        x11 = self.layer_1(x1)
        x21 = self.layer_2(x2)

        x12 = self.conv_1_1(x11)
        x22 = self.conv_2_1(x21)

        x1_w = x11.mul(nn.Sigmoid()(x22))
        x2_w = x21.mul(nn.Sigmoid()(x12))

        xw = self.layer_3(torch.cat((x1_w, x2_w), dim=1))
        out = self.layer_4(torch.cat((x11 + xw, x21 + xw), dim=1))

        return out


class Fea_fusion(nn.Module):
    def __init__(self, channel=64):
        super(Fea_fusion, self).__init__()
        out_channels = channel // 2
        self.conv3_i = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv3_o = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.cat = BasicConv2d(2 * channel, channel, kernel_size=1, stride=1, padding=0, relu=True)
        # self.cat1 = BasicConv2d(2 * channel, out_channels, kernel_size=3, stride=1, padding=1, relu=True)
        self.att0 = nn.Sequential(BasicConv2d(channel, out_channels, kernel_size=3, stride=1, padding=1, relu=True),
                                  nn.Conv2d(out_channels, channel, kernel_size=1, stride=1, padding=0),
                                  nn.BatchNorm2d(channel))
        self.att1 = nn.Sequential(BasicConv2d(channel, out_channels, kernel_size=3, stride=1, padding=1, relu=True),
                                  nn.Conv2d(out_channels, channel, kernel_size=1, stride=1, padding=0),
                                  nn.BatchNorm2d(channel))
        # self.global_ = nn.AvgPool2d((2, 2), stride=2)
        # self.global_ = nn.AdaptiveAvgPool2d(1)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        if y == None:
            f_in = self.att0(x)
            w_f = self.sig(f_in)
            s_out = self.conv3_o(w_f * x + x)
        else:
            y = self.up_2(y)
            xy = self.cat(torch.cat((x, y), dim=1))
            f_in = self.att1(xy)
            w_f = self.sig(f_in)
            s_out = self.conv3_o(w_f * xy + xy)
        return s_out


### 2024/04/02
###############################################################################
class PolypNet_v2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=64, opt=None):
        super(PolypNet_v2, self).__init__()
        self.resnet = Res2Net_model(50)
        self.downSample = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.Cross_fusion4 = Cross_fusion(1024, 2048, channel)
        self.Cross_fusion3 = Cross_fusion(512, 1024, channel)
        self.Cross_fusion2 = Cross_fusion(256, 512, channel)
        self.Cross_fusion1 = Cross_fusion(64, 256, channel)

        self.Fea_fusion4 = Fea_fusion(channel)
        self.Fea_fusion3 = Fea_fusion(channel)
        self.Fea_fusion2 = Fea_fusion(channel)
        self.Fea_fusion1 = Fea_fusion(channel)

        self.layer_dil_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), self.relu)
        self.layer_dil1_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), self.relu)
        self.layer_dil1_3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), self.relu)
        self.layer_dil1_4 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), self.relu)

        self.sal_out4 = nn.Conv2d(channel, 1, kernel_size=1)
        self.sal_out3 = nn.Conv2d(channel, 1, kernel_size=1)
        self.sal_out2 = nn.Conv2d(channel, 1, kernel_size=1)
        self.sal_out1 = nn.Conv2d(channel, 1, kernel_size=1)

        self.cat_all = nn.Sequential(nn.Conv2d(4 * channel, channel, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(channel), self.relu)
        self.global_cat = nn.AdaptiveAvgPool2d(1)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xx):
        # ---- feature abstraction -----
        x0, x1, x2, x3, x4 = self.resnet(xx)
        fuse_fea1 = self.Cross_fusion1(self.up_2(x0), x1)
        fuse_fea2 = self.Cross_fusion2(x1, x2)
        fuse_fea3 = self.Cross_fusion3(x2, x3)
        fuse_fea4 = self.Cross_fusion4(x3, x4)
        fuse_all = self.cat_all(
            torch.cat((fuse_fea1, self.up_2(fuse_fea2), self.up_4(fuse_fea3), self.up_8(fuse_fea4)), dim=1))
        f_g = self.sigmoid(self.global_cat(fuse_all))

        #### layer 4
        fu_out41 = self.Fea_fusion4(fuse_fea4, None)
        out4 = self.sal_out4(fu_out41 + fu_out41 * f_g)
        fu_out31 = self.Fea_fusion3(fuse_fea3, fu_out41)
        out3 = self.sal_out3(fu_out31 + fu_out31 * f_g)
        fu_out21 = self.Fea_fusion2(fuse_fea2, fu_out31)
        out2 = self.sal_out2(fu_out21 + fu_out21 * f_g)
        fu_out11 = self.Fea_fusion1(fuse_fea1, fu_out21)
        out1 = self.sal_out1(fu_out11 + fu_out11 * f_g)

        pred_sal1 = F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=False)
        pred_sal1 = torch.sigmoid(pred_sal1)
        pred_sal2 = F.interpolate(out2, scale_factor=4, mode='bilinear', align_corners=False)
        pred_sal2 = torch.sigmoid(pred_sal2)
        pred_sal3 = F.interpolate(out3, scale_factor=8, mode='bilinear', align_corners=False)
        pred_sal3 = torch.sigmoid(pred_sal3)
        pred_sal4 = F.interpolate(out4, scale_factor=16, mode='bilinear', align_corners=False)
        pred_sal4 = torch.sigmoid(pred_sal4)

        pred_sal1 = torch.cat((1 - pred_sal1, pred_sal1), 1)
        pred_sal2 = torch.cat((1 - pred_sal2, pred_sal2), 1)
        pred_sal3 = torch.cat((1 - pred_sal3, pred_sal3), 1)
        pred_sal4 = torch.cat((1 - pred_sal4, pred_sal4), 1)

        return pred_sal1, pred_sal2, pred_sal3, pred_sal4
