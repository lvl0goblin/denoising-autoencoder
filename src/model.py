import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 x 28 x 28 -> 16 x 28 x 28 -> pooled -> 16 x 14 x 14
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # 16 x 14 x 14 -> 32 x 14 x 14 -> pooled -> 32 x 7 x 7
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        return x  # (B, 32, 7, 7) latent representation


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.deconv1(x))
        x = self.sigmoid(self.deconv2(x))
        return x


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out
