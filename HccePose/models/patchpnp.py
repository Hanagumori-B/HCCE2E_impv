import torch
import torch.nn as nn
import torch.nn.functional as F
from ..tools.dropblock import DropBlock2D, LinearScheduler
from mmengine.model import normal_init, constant_init


class PatchPnPNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 feat_dim=128, 
                 num_gn_groups=32,
                 rot_dim=6,
                 drop_prob=0.25,
                 mask_attention_type="none",
                 denormalize_by_extent=True,
                ):
        super().__init__()
        self.mask_attention_type = mask_attention_type
        self.denormalize_by_extent = denormalize_by_extent
        self.drop_prob = drop_prob
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=5),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=5000,
        )
        if self.mask_attention_type == "concat":
            in_channels += 1
    
        self.features = nn.Sequential( # input 128 x 128
            nn.Conv2d(in_channels, feat_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_gn_groups, feat_dim),
            nn.LeakyReLU(0.1, True),
            # 64 x 64
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_gn_groups, feat_dim),
            nn.LeakyReLU(0.1, True),
            # 32 x 32
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_gn_groups, feat_dim),
            nn.LeakyReLU(0.1, True),
            # 16 x 16
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_gn_groups, feat_dim),
            nn.LeakyReLU(0.1, True),
            # 8 x 8
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_gn_groups, feat_dim),
            nn.LeakyReLU(0.1, True),
            # 8 x 8
        )

        self.flatten_dim = feat_dim * 8 * 8
        self.mlp = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True)
        )

        self.fc_r = nn.Linear(256, rot_dim) # 6D Rotation
        self.fc_t = nn.Linear(256, 3)       # t_size = (delta_x, delta_y, delta_z)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_r, std=0.01)
        normal_init(self.fc_t, std=0.01)

    def forward(self, x: torch.Tensor, region=None, extents=None, mask_attention=None):
        bs, in_c, fh, fw = x.shape
        if in_c in [3, 5] and self.denormalize_by_extent and extents is not None:
            x[:, :3, :, :] = (x[:, :3, :, :] - 0.5) * extents.view(bs, 3, 1 ,1)
        elif in_c in [6, 8] and self.denormalize_by_extent and extents is not None:
            x[:, :6, :, :] = (x[:, :6, :, :] - 0.5) * extents.view(bs, 3, 1, 1).repeat(1, 2, 1, 1)
        else:
            raise ValueError('Wrong input shape!')
        if region is not None:
            x = torch.cat([x, region], dim=1)
            
        if self.mask_attention_type != "none":
            assert mask_attention is not None
            if self.mask_attention_type == "mul":
                x = x * mask_attention  # 元素乘法，抑制背景噪声    
            elif self.mask_attention_type == "concat":
                x = torch.cat([x, mask_attention], dim=1)    
            else:
                raise ValueError(f"Wrong mask attention type: {self.mask_attention_type}")
                
        # DropBlock 正则化
        if self.training:
            if self.drop_prob > 0:
                self.dropblock.step()  # 更新步数以调整 drop 概率
                x = self.dropblock(x)
                
        x = self.features(x)
        x = x.flatten(1)
        x = self.mlp(x)
        
        rot_6d = self.fc_r(x)
        t_site = self.fc_t(x)
        
        return rot_6d, t_site