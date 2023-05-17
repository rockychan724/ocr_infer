import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual_block(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super(Residual_block, self).__init__()
        self.downsample = None
        flag = False
        if isinstance(stride, tuple):
            if stride[0] > 1:
                self.downsample = nn.Sequential(
                    nn.Conv2d(c_in, c_out, 3, stride, 1),
                    nn.BatchNorm2d(c_out, momentum=0.01),
                )
                flag = True
        else:
            if stride > 1:
                self.downsample = nn.Sequential(
                    nn.Conv2d(c_in, c_out, 3, stride, 1),
                    nn.BatchNorm2d(c_out, momentum=0.01),
                )
                flag = True
        if flag:
            self.conv1 = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride, 1),
                nn.BatchNorm2d(c_out, momentum=0.01),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, 0),
                nn.BatchNorm2d(c_out, momentum=0.01),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, 1, 1), nn.BatchNorm2d(c_out, momentum=0.01)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(residual + conv2)


class CNN_ResNet18_for_cnc(nn.Module):
    def __init__(self, c_in):
        super(CNN_ResNet18_for_cnc, self).__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(c_in, 64, 7, 1, 1), nn.BatchNorm2d(64)
        )  # 48*480 -> 44*476
        self.block1 = self._make_layer(64, 128, (2, 2), 1)  # -> 22*238
        self.block2 = self._make_layer(128, 256, (2, 2), 1)  # -> 11*119
        self.block3 = self._make_layer(256, 512, (2, 2), 1)  # -> 6*60
        self.block4 = self._make_layer(512, 512, (2, 1), 1)  # -> 3*60
        self.block5 = self._make_layer(512, 512, (3, 1), 0)  # -> 1*60

    def _make_layer(self, c_in, c_out, stride, repeat=2):
        layers = []
        layers.append(Residual_block(c_in, c_out, stride))
        for i in range(repeat):
            layers.append(Residual_block(c_out, c_out, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        block0 = self.block0(x)
        block1 = self.block1(block0)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block5 = block5.squeeze(2).permute(0, 2, 1).contiguous()
        return block5


class CRNN_OCR_for_cnc(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(CRNN_OCR_for_cnc, self).__init__()
        self.CNN_encoder = CNN_ResNet18_for_cnc(c_in=in_channel)
        self.cls_op = nn.Linear(512, num_classes)

    def forward(self, inputs):  # bs x 1 x 48 x 480
        features = self.CNN_encoder(inputs)
        logits = self.cls_op(features)
        return logits  # bs x 60 x nclass


# for gpu's tensorrt
class CRNN_OCR_for_cnc_trt(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(CRNN_OCR_for_cnc_trt, self).__init__()
        self.CNN_encoder = CNN_ResNet18_for_cnc(c_in=in_channel)
        self.cls_op = nn.Linear(512, num_classes)

    def forward(self, inputs):  # bs x 1 x 48 x 480
        inputs = (inputs - 127.0) / 128.0
        features = self.CNN_encoder(inputs)
        logits = self.cls_op(features)  # bs x 60 x nclass
        preds = logits.argmax(dim=2).reshape(logits.size(0), -1)
        return preds  # bs x 60


# for mlu's cnrt
class CRNN_OCR_for_cnc_e2e(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(CRNN_OCR_for_cnc_e2e, self).__init__()
        self.CNN_encoder = CNN_ResNet18_for_cnc(c_in=in_channel)
        self.cls_op = nn.Linear(512, num_classes)

    def forward(self, inputs):  # bs x 1 x 48 x 480
        features = self.CNN_encoder(inputs)
        logits = self.cls_op(features)  # bs x 60 x nclass
        logits = logits.unsqueeze(1)  # bs x 1 x 60 x nclass
        b, _, word_len, class_num = logits.shape
        preds = torch.ops.torch_mlu.mlu_ctc_greed_decoder(
            logits, b, word_len, class_num, True
        )
        preds = preds.reshape(b, -1)
        return preds  # bs x 60
