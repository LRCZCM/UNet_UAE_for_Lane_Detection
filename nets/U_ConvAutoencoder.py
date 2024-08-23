import torch
import torch.nn as nn
class U_ConvAutoencoder(nn.Module):
    def __init__(self):
        super(U_ConvAutoencoder, self).__init__()
        # 编码器
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # output: 16 x 1692 x 855
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # output: 32 x 846 x 428
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # output: 64 x 423 x 214
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # output: 128 x 212 x 107
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # output: 256 x 106 x 54
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # 解码器
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # output: 128 x 212 x 107
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: 64 x 423 x 214
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: 32 x 846 x 428
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: 16 x 1692 x 855
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # output: 1 x 3384 x 1710
            nn.Sigmoid()  # 使用Sigmoid以确保输出在[0, 1]范围内
        )

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        # 解码器
        dec5 = self.decoder5(enc5)
        dec4 = self.decoder4(torch.cat([dec5, enc4], dim=1))
        dec3 = self.decoder3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))

        return dec1