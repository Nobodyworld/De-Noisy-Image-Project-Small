# model_parts/unet.py
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
# Encoder
        self.enc_conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.res_enc1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(16), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(16))
        self.enc_conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.res_enc2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(32), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(32))
        self.enc_conv3 = nn.Sequential(nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.res_enc3 = nn.Sequential(nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(48), nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(48))
        self.enc_conv4 = nn.Sequential(nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool4 = nn.MaxPool2d(2, 2)
        self.res_enc4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64))
        self.enc_conv5 = nn.Sequential(nn.Conv2d(64, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True), nn.Conv2d(80, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True))
        self.pool5 = nn.MaxPool2d(2, 2)
        self.res_enc5 = nn.Sequential(nn.Conv2d(80, 80, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(80), nn.Conv2d(80, 80, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(80))
        self.enc_conv6 = nn.Sequential(nn.Conv2d(80, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.pool6 = nn.MaxPool2d(2, 2)
        self.res_enc6 = nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(96), nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(96))
        self.enc_conv7 = nn.Sequential(nn.Conv2d(96, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True), nn.Conv2d(112, 112, 3, padding=1), nn.BatchNorm2d(112), nn.ReLU(inplace=True))

        # Decoder
        self.dec_conv7 = nn.Sequential(nn.Conv2d(112, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True), nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv6 = nn.Sequential(nn.Conv2d(96, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True), nn.Conv2d(80, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True))
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv5 = nn.Sequential(nn.Conv2d(80, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv4 = nn.Sequential(nn.Conv2d(64, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = nn.Sequential(nn.Conv2d(48, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Modify the last decoder layer to have 32 output channels
        self.dec_conv1 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        # Keep the output layer's input channels at 32
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid()  # or nn.Sigmoid()
        )


    def forward(self, x):
        # Encoder
        x1 = self.enc_conv1(x)
        x2 = self.pool1(x1)
        x2 = x2 + self.res_enc1(x2)
        x3 = self.enc_conv2(x2)
        x4 = self.pool2(x3)
        x4 = x4 + self.res_enc2(x4)
        x5 = self.enc_conv3(x4)
        x6 = self.pool3(x5)
        x6 = x6 + self.res_enc3(x6)
        x7 = self.enc_conv4(x6)
        x8 = self.pool4(x7)
        x8 = x8 + self.res_enc4(x8)
        x9 = self.enc_conv5(x8)
        x10 = self.pool5(x9)
        x10 = x10 + self.res_enc5(x10)
        x11 = self.enc_conv6(x10)
        x12 = self.pool6(x11)
        x12 = x12 + self.res_enc6(x12)
        x13 = self.enc_conv7(x12)

        # Decoder
        x14 = self.dec_conv7(x13)
        x14 = x14 + x12
        x15 = self.up6(x14)
        x16 = self.dec_conv6(x15)
        x16 = x16 + x10
        x17 = self.up5(x16)
        x18 = self.dec_conv5(x17)
        x18 = x18 + x8
        x19 = self.up4(x18)
        x20 = self.dec_conv4(x19)
        x20 = x20 + x6
        x21 = self.up3(x20)
        x22 = self.dec_conv3(x21)
        x22 = x22 + x4
        x23 = self.up2(x22)
        x24 = self.dec_conv2(x23)
        x24 = x24 + x2
        x25 = self.up1(x24)
        x26 = self.dec_conv1(x25)

        # Output layer
        x27 = self.out_conv(x26)

        return x27
