import torch
import torch.nn as nn


class UNet1(nn.Module):
    def __init__(self, kernel, num_filters, num_colours, num_in_channels):
        super(UNet1, self).__init__()
        self.n_channels = num_in_channels
        self.n_classes = num_colours

        padding = kernel // 2
        self.downconv1 = nn.Sequential(
            nn.Conv2d(num_in_channels, num_filters, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2), )
        self.downconv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(2), )

        self.rfconv = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU())

        self.upconv1 = nn.Sequential(
            nn.Conv2d(num_filters * 2 * 2, num_filters, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), )
        self.upconv2 = nn.Sequential(
            nn.Conv2d(num_filters * 2, 3, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), )

        self.finalconv = nn.Conv2d(num_in_channels + 3, num_colours, kernel_size=kernel, padding=padding)

    def forward(self, x):
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.rfconv(self.out2)
        self.out4 = self.upconv1(torch.cat((self.out3, self.out2), dim=1))
        self.out5 = self.upconv2(torch.cat((self.out4, self.out1), dim=1))
        self.out_final = self.finalconv(torch.cat((self.out5, x), dim=1))
        return self.out_final
