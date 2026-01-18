import torch.nn as nn


class MAEDecoder(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

