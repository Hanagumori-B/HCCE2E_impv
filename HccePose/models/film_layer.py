import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    def __init__(self, emb_dim, feature_dim):
        super().__init__()
        # 从 Embedding 映射到 gamma 和 beta
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
        self.feature_dim = feature_dim

    def forward(self, x, emb):
        # emb: [B, emb_dim]
        # x: [B, C, H, W]
        stats = self.mlp(emb).view(-1, self.feature_dim * 2, 1, 1)
        gamma, beta = torch.split(stats, self.feature_dim, dim=1)
        
        # 核心变换逻辑
        # 这里的 1 + gamma 是为了让初始化更稳定（初始 gamma 为 0 时，变换为恒等变换）
        return x * (1 + gamma) + beta

