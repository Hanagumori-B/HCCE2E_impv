import torch
import torch.nn as nn
import torch.nn.functional as F


class HcceDecodeNet(nn.Module):
    def __init__(self, in_channels=24, hidden_dim=64):
        """
        input: [B, 24, H, W] \n
        output: [B, 3, H ,W] norm(0~1)
        """
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim * 3, kernel_size=1, groups=3),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 3, 3, kernel_size=1, groups=3),
            nn.Sigmoid()
        )
        last_conv = self.decoder[2]
        nn.init.normal_(last_conv.weight, mean=0.0, std=0.01)
        if last_conv.bias is not None:
            nn.init.constant_(last_conv.bias, 0.0)
        
    def forward(self, prob):
        coords = self.decoder(prob)
        return coords
