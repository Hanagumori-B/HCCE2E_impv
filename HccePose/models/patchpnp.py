import torch
import torch.nn as nn
import torch.nn.functional as F
from ..tools.dropblock import DropBlock2D, LinearScheduler
from mmengine.model import normal_init, constant_init


class PnPTransformer(nn.Module):
    def __init__(self, feat_dim, num_heads=8, dropout=0.05):
        super().__init__()
        self.attention = nn.MultiheadAttention(feat_dim, num_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(feat_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, feat_dim)
        )
        self.norm2 = nn.LayerNorm(feat_dim)
        
    def forward(self, x, pose_embed):
        x = x + pose_embed
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        norm_x2 = self.norm2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + ffn_out
        return x


class QueryPoseDecoder(nn.Module):
    def __init__(self, feat_dim, rot_dim, num_heads=8):
        super().__init__()
        self.pose_queries = nn.Parameter(torch.randn(3, feat_dim))
        self.cross_attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(feat_dim)
        self.norm_kv = nn.LayerNorm(feat_dim)
        self.norm_ffn = nn.LayerNorm(feat_dim)
        self.norm_final = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Linear(feat_dim * 4, feat_dim)
        )
        # 6D Rotation
        self.fc_r = nn.Linear(feat_dim, rot_dim)
        # t_size = (delta_x, delta_y, delta_z)
        self.fc_xy = nn.Linear(feat_dim, 2)
        self.fc_z = nn.Linear(feat_dim, 1)
        
    def forward(self, attn_tokens): 
        B = attn_tokens.shape[0] # input: [B, 64, 128]
        queries = self.pose_queries.unsqueeze(0).expand(B ,-1, -1) # [B, 3, 128]
        q_norm = self.norm_q(queries)
        kv_norm = self.norm_kv(attn_tokens)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        queries = queries + attn_out
        ffn_in = self.norm_ffn(queries)
        queries = queries + self.ffn(ffn_in)
        out_queries = self.norm_final(queries)
        rot_6d = self.fc_r(out_queries[:, 0, :])
        xy_site = self.fc_xy(out_queries[:, 1, :])
        z_site = self.fc_z(out_queries[:, 2, :])
        
        t_site = torch.concat([xy_site, z_site], dim=1)
        return rot_6d, t_site

    
class PatchPnPNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 feat_dim=128, 
                 num_gn_groups=16,
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
            nn.GELU(),
            # 64 x 64
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_gn_groups, feat_dim),
            nn.GELU(),
            # 32 x 32
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_gn_groups, feat_dim),
            nn.GELU(),
            # 16 x 16
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_gn_groups, feat_dim),
            nn.GELU(),
            # 8 x 8
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_gn_groups, feat_dim),
            nn.GELU(),
        )

        # self.flatten_dim = feat_dim * 8 * 8
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.flatten_dim, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 256),
        #     nn.ReLU(inplace=True)
        # )

        # self.features = MultiScaleFeatureExtractor(in_channels, feat_dim, num_gn_groups)
        self.pnp_transformer = PnPTransformer(feat_dim, 8)
        self.pose_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, feat_dim)
        )
        # # 6D Rotation
        # self.fc_r = nn.Linear(feat_dim, rot_dim)
        # # t_size = (delta_x, delta_y, delta_z)
        # self.fc_xy = nn.Linear(feat_dim, 2)
        # self.fc_z = nn.Linear(feat_dim, 1)
        self.query_pose = QueryPoseDecoder(feat_dim, rot_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.query_pose.fc_r, std=0.01)
        normal_init(self.query_pose.fc_xy, std=0.01)
        normal_init(self.query_pose.fc_z, std=0.01)

    def forward(self, x: torch.Tensor, region=None, extents=None, mask_attention=None):
        bs, in_c, fh, fw = x.shape
        if in_c == 8:
            coord_2d = x[:, 6:8, :, :]
            coords_3d = (x[:, :6, :, :] - 0.5) * extents.view(bs, 3, 1, 1).repeat(1, 2, 1, 1)
            x = torch.cat([coords_3d, coord_2d], dim=1)
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
        # x = x.flatten(1)
        # # x = self.mlp(x)
        B, C, H, W = x.shape
        coord2d_low_res = F.adaptive_avg_pool2d(coord_2d, (H, W))
        coord_seq = coord2d_low_res.view(B, 2, H * W).permute(0, 2, 1)
        # avg_mask = F.adaptive_avg_pool2d(mask_attention, (H, W)).view(B, -1, 1)
        pos_embed = self.pose_proj(coord_seq)
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = self.pnp_transformer(x, pos_embed)
        # masked_x = x * avg_mask
        # x = masked_x.sum(dim=1) / (avg_mask.sum(dim=1) + 1e-6) # Masked Pooling
        # x = x.mean(dim=1) # Mean Pooling
        # rot_6d = self.fc_r(x)
        # xy_site = self.fc_xy(x)
        # z_site = self.fc_z(x)
        # t_site = torch.concat([xy_site, z_site], dim=1)
        rot_6d, t_site = self.query_pose(x)
        return rot_6d, t_site