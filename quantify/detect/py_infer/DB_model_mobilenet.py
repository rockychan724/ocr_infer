#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d


class SegDetector_Mobile_Cut(nn.Module):
    def __init__(
        self,
        in_channels=[16, 32],
        inner_channels=16,
        k=10,
        bias=False,
        adaptive=False,
        smooth=False,
        serial=False,
    ):
        """
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        """
        super(SegDetector_Mobile_Cut, self).__init__()
        self.k = k
        self.serial = serial
        # self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")

        # self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        # self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)

        # self.out5 = nn.Sequential(
        #     nn.Conv2d(inner_channels, inner_channels //
        #               4, 3, padding=1, bias=bias),
        #     nn.Upsample(scale_factor=8, mode='nearest'))
        # self.out4 = nn.Sequential(
        #     nn.Conv2d(inner_channels, inner_channels //
        #               4, 3, padding=1, bias=bias),
        #     nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 2, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels // 2, 3, padding=1, bias=bias
        )

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 2, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 2, inner_channels // 2, 2, 2),
            BatchNorm2d(inner_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 2, 1, 2, 2),
            nn.Sigmoid(),
        )
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                inner_channels, serial=serial, smooth=smooth, bias=bias
            )
            self.thresh.apply(self.weights_init)

        # self.in5.apply(self.weights_init)
        # self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        # self.out5.apply(self.weights_init)
        # self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 2, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels // 2),
            nn.ReLU(inplace=True),
            self._init_upsample(
                inner_channels // 2, inner_channels // 2, smooth=smooth, bias=bias
            ),
            BatchNorm2d(inner_channels // 2),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 2, 1, smooth=smooth, bias=bias),
            nn.Sigmoid(),
        )
        return self.thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias),
            ]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=1,
                        bias=True,
                    )
                )

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features):
        c2, c3 = features
        # in5 = self.in5(c5)
        # in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)  # 64

        # out4 = self.up5(in5) + in4  # 1/16
        # out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(in3) + in2  # 1/4

        # p5 = self.out5(in5)
        # p4 = self.out4(out4)
        # p3 = self.out3(out3)
        p3 = self.out3(in3)
        p2 = self.out2(out2)

        fuse = torch.cat((p3, p2), 1)
        # this is the pred module, not binarization module;
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        # if self.training:
        #     result = OrderedDict(binary=binary)
        # else:
        return binary

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class BottleNeck(nn.Module):
    def __init__(self, inchannles, outchannels, expansion=1, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        # 1*1
        self.conv1 = nn.Conv2d(inchannles, inchannles * expansion, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(inchannles * expansion)
        # 3*3 可分离卷积　groups设置
        self.conv2 = nn.Conv2d(
            inchannles * expansion,
            inchannles * expansion,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=inchannles * expansion,
        )
        self.bn2 = nn.BatchNorm2d(inchannles * expansion)
        # 1*1
        self.conv3 = nn.Conv2d(inchannles * expansion, outchannels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residul = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residul = self.downsample(x)
        out += residul
        out = self.relu(out)
        return out


class MobileNetV2(nn.Module):
    def __init__(self, n, numclasses=1000):
        super(MobileNetV2, self).__init__()
        self.inchannels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(n[0], outchannels=16, stride=1, expansion=1)
        self.layer2 = self.make_layer(n[1], outchannels=16, stride=2, expansion=6)
        self.layer3 = self.make_layer(n[2], outchannels=32, stride=2, expansion=6)
        # self.layer4 = self.make_layer(n[3], outchannels=64, stride=2, expansion=6)
        # self.layer5 = self.make_layer(n[4], outchannels=64, stride=1, expansion=6)
        # self.layer6 = self.make_layer(n[5], outchannels=128, stride=2, expansion=6)
        # self.layer7 = self.make_layer(n[6], outchannels=128, stride=1, expansion=1)
        # self.conv8 = nn.Conv2d(320, 1280, kernel_size=1, stride=1)
        # self.avegpool = nn.AvgPool2d(7, stride=1)
        # self.conv9 = nn.Conv2d(1280, numclasses, kernel_size=1, stride=1)

    def make_layer(self, blocks_num, outchannels, stride, expansion):
        downsample_ = nn.Sequential(
            nn.Conv2d(self.inchannels, outchannels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(outchannels),
        )
        layers = []
        # 下采样的shortcut有downsample
        temp = BottleNeck(
            self.inchannels,
            outchannels,
            expansion=expansion,
            stride=stride,
            downsample=downsample_,
        )
        layers.append(temp)
        # 剩下的shortcut干净
        self.inchannels = outchannels
        for i in range(1, blocks_num):
            layers.append(
                BottleNeck(self.inchannels, outchannels, expansion=expansion, stride=1)
            )
        return nn.Sequential(*layers)  # 取出每一层

    def forward(self, x):
        # conv1.shape: torch.Size([4, 32, 320, 320])
        # layer1.shape: torch.Size([4, 64, 320, 320])
        # layer2.shape: torch.Size([4, 64, 160, 160])
        # layer3.shape: torch.Size([4, 128, 80, 80])
        # layer4.shape: torch.Size([4, 256, 40, 40])
        # layer5.shape: torch.Size([4, 256, 40, 40])
        # layer6.shape: torch.Size([4, 512, 20, 20])
        # layer7.shape: torch.Size([4, 512, 20, 20])
        # conv1.shape: torch.Size([4, 32, 320, 320])
        # layer1.shape: torch.Size([4, 16, 320, 320])
        # layer2.shape: torch.Size([4, 24, 160, 160])
        # layer3.shape: torch.Size([4, 32, 80, 80])
        # layer4.shape: torch.Size([4, 64, 40, 40])
        # layer5.shape: torch.Size([4, 96, 40, 40])
        # layer6.shape: torch.Size([4, 160, 20, 20])
        # layer7.shape: torch.Size([4, 320, 20, 20])

        x = self.conv1(x)
        # print('conv1.shape:', x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        # print('layer1.shape:', x.shape)
        x = self.layer2(x)
        # print('layer2.shape:', x.shape)
        c2 = x
        x = self.layer3(x)
        # print('layer3.shape:', x.shape)
        c3 = x
        # x = self.layer4(x)
        # # print('layer4.shape:', x.shape)
        # x = self.layer5(x)
        # # print('layer5.shape:', x.shape)
        # c4 = x
        # x = self.layer6(x)
        # # print('layer6.shape:', x.shape)
        # x = self.layer7(x)
        # # print('layer7.shape:', x.shape)
        # c5 = x
        # x = self.conv8(x)
        # print('conv8.shape:', x.shape)
        # x = self.avegpool(x)
        # print('avegpool:', x.shape)
        # x = self.conv9(x)
        # print('conv9.shape:', x.shape)
        # x = x.view(x.size(0), -1)
        return c2, c3


def mobilenetv2_cut():
    # model = MobileNetV2(n=[1, 2, 3, 4, 3, 3, 1], numclasses=10)
    model = MobileNetV2(n=[1, 2, 3], numclasses=10)
    return model


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        # nn.Module.__init__(self)
        # print(args['backbone'], args['decoder'])
        # self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        # self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))
        self.backbone = mobilenetv2_cut()
        self.decoder = SegDetector_Mobile_Cut(adaptive=True, in_channels=[16, 32], k=50)

    def forward(self, data):
        return self.decoder(self.backbone(data))


# for tensorrt
class BasicModel_for_trt(nn.Module):
    def __init__(self):
        super(BasicModel_for_trt, self).__init__()
        # nn.Module.__init__(self)
        # print(args['backbone'], args['decoder'])
        # self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        # self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))
        self.RGB_MEAN = (
            torch.FloatTensor([122.67891434, 116.66876762, 104.00698793])
            .reshape(1, 3, 1, 1)
            .cuda()
        )
        self.backbone = mobilenetv2_cut()
        self.decoder = SegDetector_Mobile_Cut(adaptive=True, in_channels=[16, 32], k=50)

    def forward(self, data):
        data = (data - self.RGB_MEAN) / 255.0
        return self.decoder(self.backbone(data))
