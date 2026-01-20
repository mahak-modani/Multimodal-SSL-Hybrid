import torch
import torch.nn as nn
import torch.nn.functional as F


class MAEDecoder(nn.Module):
    """
    Lightweight decoder that reconstructs coarse RGB patches
    from ResNet feature maps.
    """

    def __init__(self, in_dim=512):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=1)
        )

    def forward(self, x):
        """
        x: [B, C, H, W] feature map
        returns: [B, 3, H, W]
        """
        return self.decoder(x)
