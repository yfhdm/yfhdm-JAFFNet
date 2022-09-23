
# Spatial attention abalation
import torch
from torchvision import models

import torch.nn.functional as F
from .resnet_model import *

class BasicBlockV1b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MRF(nn.Module):
    def __init__(self, in_channels, out_channels,rate=[1,2,4,8]):
        super(MRF, self).__init__()

        inner_channels=in_channels//4

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv1= nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, 3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, 3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, 3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            )


        self.pro = nn.Sequential(
            nn.Conv2d(inner_channels * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):

        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)


        out = torch.cat([feat1, feat2, feat3], dim=1)
        return self.pro(out)

class DRF(nn.Module):
    def __init__(self, in_channels):
        super(DRF, self).__init__()
        out_channels =512
        self.mrf1 = MRF(in_channels,out_channels)
        self.mrf2 = MRF(in_channels,out_channels)
        self.mrf3 = MRF(in_channels,out_channels)
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.relu= nn.ReLU(inplace=True)
        self.pro = nn.Sequential(
            nn.Conv2d(in_channels , out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):

        size = x.size()[2:]
        feat1 = self.mrf1(x)
        input = x + feat1
        feat2 = self.mrf2(input)
        input = x+feat1+feat2
        feat3 = self.mrf3(input)
        feat4 = F.interpolate(self.conv4(self.gap(x)), size, mode='bilinear', align_corners=True)
        feat = x+feat1+feat2+feat3+feat4
        out = self.pro(feat)
        return out

class JAFFM(nn.Module):
    def __init__(self, in_channels):
        super(JAFFM, self).__init__()
        in_planes = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_a = nn.Sequential(
            nn.Conv2d(in_channels, in_planes, 1, bias=False),
            nn.ReLU(True)
        )
        self.conv_b = nn.Sequential(
            nn.Conv2d(in_channels, in_planes, 1, bias=False),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )


        self.spatial_net = nn.Sequential(
            nn.Conv2d(2, 64, 3,dilation=2 ,padding=2,bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, dilation=4, padding=4,bias=False),
            nn.Sigmoid()
        )

        self.pro = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 3,dilation=2, padding=2, bias=False, groups=in_planes),

        )
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(1))


    def forward(self, x, y):

        # channal attetion
        bs,c,h,w=x.size()
        a = self.conv_a(x)
        avg_out = self.conv3(self.conv1(self.avg_pool(a)))
        max_out = self.conv3(self.conv2(self.max_pool(a)))
        c_out = self.sigmoid(avg_out + max_out)


        # spacial attetion
        b = self.conv_b(x)
        avg_out = torch.mean(b, dim=1, keepdim=True)
        max_out, _ = torch.max(b, dim=1, keepdim=True)
        s_in = torch.cat([avg_out, max_out], dim=1)
        s_out = self.spatial_net(s_in)



        c_out=c_out.view(bs,c,1)
        s_out=s_out.view(bs,1,h*w)
        atten=self.pro(torch.bmm(c_out,s_out).view(bs,c,h,w))


        new_y = torch.mul(y, atten)



        return new_y * self.alpha + y

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)

class JAFFNet(nn.Module):
    def __init__(self, n_classes=1):
        super(JAFFNet, self).__init__()

        resnet = models.resnet18(pretrained=True)
        ## -------------Encoder--------------

        self.inconv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # stage 1
        self.encoder1 = resnet.layer1
        # stage 2
        self.encoder2 = resnet.layer2
        # stage 3
        self.encoder3 = resnet.layer3
        # stage 4
        self.encoder4 = resnet.layer4
        # stage 5
        self.encoder5 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
        )
        self.pool=nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 5g
        self.decoder5_g = nn.Sequential(

            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # stage 4g
        self.decoder4_g = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # stage 3g
        self.decoder3_g = nn.Sequential(

            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # stage 2g
        self.decoder2_g = nn.Sequential(

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconv6 = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, n_classes, 3, padding=1)


        self.jaff1 = JAFFM(512)
        self.jaff2 = JAFFM(512)
        self.jaff3 = JAFFM(256)
        self.jaff4 = JAFFM(128)

        self.bridge = DRF(512)

    def forward(self, x):
        hx = x
        ## -------------Encoder-------------
        hx = self.inconv(hx)

        h1 = self.encoder1(hx)
        h2 = self.encoder2(h1)
        h3 = self.encoder3(h2)
        h4 = self.encoder4(h3)

        h5 = self.encoder5(h4)
        #
        bg=self.bridge(self.pool(h5))

        bgx =  self.upscore2(bg)

        ## -------------Decoder5-------------
        hd5 = self.decoder5_g(torch.cat((bgx, self.jaff1(bgx, h5)), 1))  # 1024-512
        hx5 =  self.upscore2(hd5)

        # -------------Decoder4-------------
        hd4 = self.decoder4_g(torch.cat((hx5, self.jaff2(hx5, h4)), 1))  # 1024->256
        hx4 =  self.upscore2(hd4)

        ## -------------Decoder3-------------
        hd3 = self.decoder3_g(torch.cat((hx4, self.jaff3(hx4, h3)), 1))
        hx3 = self.upscore2(hd3)

        ## -------------Decoder2-------------
        hd2 = self.decoder2_g(torch.cat((hx3, self.jaff4(hx3, h2)), 1))  # 256->64


        out_b = self.outconv6(bgx)
        out_b = self.upscore5(out_b)
        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)
        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)
        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)
        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)


        return  F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5),F.sigmoid(out_b)


