import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    '''
    - data_format="channels_last": 输入 (N, H, W, C) -> 类似于 nn.Linear 的输入
    - data_format="channels_first": 输入 (N, C, H, W) -> 传统的 Conv 输入
    '''
    def __init__(self, normalized_shape: int | tuple, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape, )
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim), requires_grad=True)
        
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity() if drop_path <= 0 else DropPath(drop_path)
        
    def forward(self, x: torch.Tensor):
        input_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input_x + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    def __init__(self, in_c=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.):
        super().__init__()
        self.depths = depths
        
        self.downsample_layers = nn.ModuleList()
        # (4x下采样)
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_c, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            ))
            
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(
                *[Block(dims[i], dp_rates[cur + j]) for j in range(depths[i])]
            ))
            cur += depths[i]
        
        self.out_c = dims
    
    def forward(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        # features[0]: stride 4  (Low-level)
        # features[1]: stride 8
        # features[2]: stride 16
        # features[3]: stride 32 (High-level)
        return features
    
    def load_pretrained_weights(self, weight_path):
        checkpoint = torch.load(weight_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # 简单的 key 匹配逻辑
        model_dict = self.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            # 官方权重没有 'downsample_layers' 前缀，可能需要调整
            # 这里假设你是手动加载或者权重 key 已经匹配
            if k in model_dict and v.shape == model_dict[k].shape:
                new_state_dict[k] = v
        
        msg = self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded Backbone weights: {msg}")


# ==========================================================================


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))
        
        # Atrous convolutions
        rates = [6, 12, 18]
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))

        # Image Pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        self.convs = nn.ModuleList(modules)
        
        # Project after concatenation
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            # Check if it is the image pooling branch
            if isinstance(conv[0], nn.AdaptiveAvgPool2d):
                # AdaptiveAvgPool -> Conv -> BN -> ReLU -> Upsample
                feat = conv(x)
                feat = F.interpolate(feat, size=x.size()[2:], mode='bilinear', align_corners=False)
            else:
                feat = conv(x)
            res.append(feat)
        res = torch.cat(res, dim=1)
        return self.project(res)
    

class DecoderBlock(nn.Module):
    """标准的卷积-BN-ReLU块,用于特征融合后的处理"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class FPN_Fusion(nn.Module):
    """
    FPN 风格的特征融合模块
    将 C1, C2, C3, C4 全部映射到同一通道，并上采样到 C1 的尺寸进行拼接融合
    """
    def __init__(self, dims, out_channels=64):
        """
        Args:
            dims: Backbone 输出的各层通道数列表，例如 [96, 192, 384, 768]
            out_channels: 每一层投影后的通道数 (建议 32 或 64)
        """
        super().__init__()
        
        # 1. 侧向连接 (Lateral Connections) - 1x1 卷积降维
        self.lat_c4 = nn.Conv2d(dims[3], out_channels, 1)
        self.lat_c3 = nn.Conv2d(dims[2], out_channels, 1)
        self.lat_c2 = nn.Conv2d(dims[1], out_channels, 1)
        self.lat_c1 = nn.Conv2d(dims[0], out_channels, 1)
        
        # 2. 融合层 (Feature Aggregation)
        # 输入通道数 = 4 * out_channels (因为拼接了4层)
        # 使用 GroupNorm 代替 BatchNorm，对 ConvNeXt 和小 Batch 训练更友好
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, 256, 3, padding=1, bias=False),
            nn.GroupNorm(32, 256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        c1, c2, c3, c4 = features
        
        # 1. 降维投影
        p4 = self.lat_c4(c4) # Stride 32
        p3 = self.lat_c3(c3) # Stride 16
        p2 = self.lat_c2(c2) # Stride 8
        p1 = self.lat_c1(c1) # Stride 4
        
        # 2. 统一上采样到 C1 的尺寸 (Stride 4)
        target_size = p1.shape[-2:]
        
        p4_up = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=target_size, mode='bilinear', align_corners=False)
        p2_up = F.interpolate(p2, size=target_size, mode='bilinear', align_corners=False)
        # p1 不需要上采样
        
        # 3. 拼接 (Concat)
        # [B, 4*out_ch, H/4, W/4]
        cat_features = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)
        
        # 4. 融合
        out = self.fusion(cat_features)
        
        return out


# ==========================================================================


class ConvNeXtV2_DL(nn.Module):
    def __init__(self, num_classes, input_channels=3, variant='tiny'):
        """
        Args:
            num_classes: 输出的总通道数 (mask + binary_code = 1 + 48)
            input_channels: 输入图片通道数 (通常为3)
            variant: ConvNeXtV2 的规格 ('tiny', 'base', 'large')
        """
        super().__init__()
        self.num_classes = num_classes
        
        # 1. 配置 Backbone (默认为 Tiny 配置，可根据显存调整)
        configs = {
            'tiny':  {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
            'base':  {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
            'large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]},
            # 可以根据需要添加其他规格
        }
        
        cfg = configs.get(variant, configs['tiny'])
        self.backbone = ConvNeXtV2(
            input_channels, 
            depths=cfg['depths'], 
            dims=cfg['dims']
        )
        
        # Backbone 输出通道数
        # c1 (stride 4), c2 (stride 8), c3 (stride 16), c4 (stride 32)
        dims = cfg['dims']
        
        # 2. ASPP Head (处理 High-level features)
        aspp_out_channels = 256
        self.aspp = ASPP(in_channels=dims[3], out_channels=aspp_out_channels)
        
        # 3. Low-level feature projection (DeepLabV3+ 结构)
        self.lat_c3 = nn.Conv2d(dims[2], 128, 1) # Stride 16
        self.lat_c2 = nn.Conv2d(dims[1], 64, 1)  # Stride 8
        self.lat_c1 = nn.Conv2d(dims[0], 32, 1)  # Stride 4
        
        # 4. Decoder (融合层)
        # Fusion 1: ASPP(256) + C3(128) -> 256
        self.dec_block3 = DecoderBlock(256 + 128, 256)
        # Fusion 2: Prev(256) + C2(64) -> 128
        self.dec_block2 = DecoderBlock(256 + 64, 128)
        # Fusion 3: Prev(128) + C1(32) -> 64
        self.dec_block1 = DecoderBlock(128 + 32, 64)
        # 5. Final Head
        self.final_conv = nn.Conv2d(64, num_classes, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: [B, 3, H, W]
        input_shape = x.shape[-2:]
        
        # features = [c1, c2, c3, c4]
        # c1: 64x64 (Stride 4)
        # c2: 32x32 (Stride 8)
        # c3: 16x16 (Stride 16)
        # c4: 8x8   (Stride 32)
        features = self.backbone(x)
        c1, c2, c3, c4 = features
        
        x = self.aspp(c4) # [B, 256, H/32, W/32]
        
        # --- Decoder (Step-wise Upsampling) ---
        
        # Step 1: Fuse C3 (S16)
        x = F.interpolate(x, size=c3.shape[-2:], mode='bilinear', align_corners=False) # Up to S16
        _c3 = self.lat_c3(c3)
        x = torch.cat([x, _c3], dim=1) # Concat
        x = self.dec_block3(x)         # Process
        
        # Step 2: Fuse C2 (S8)
        x = F.interpolate(x, size=c2.shape[-2:], mode='bilinear', align_corners=False) # Up to S8
        _c2 = self.lat_c2(c2)
        x = torch.cat([x, _c2], dim=1)
        x = self.dec_block2(x)
        
        # Step 3: Fuse C1 (S4) - 最重要的空间信息层
        x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False) # Up to S4
        _c1 = self.lat_c1(c1)
        x = torch.cat([x, _c1], dim=1)
        x = self.dec_block1(x) # [B, 64, H/4, W/4]
        
        # --- Head ---
        output = self.final_conv(x)
        
        # --- Final Resizing ---
        # 目标是输出 stride=2 (即 1/2 原图大小)，以匹配 GT (128x128 for 256x256 input)
        target_shape = (input_shape[0] // 2, input_shape[1] // 2)
        output = F.interpolate(output, size=target_shape, mode='bilinear', align_corners=False)
        
        # Split
        mask, binary_code = torch.split(output, [1, self.num_classes - 1], 1)
        
        return mask, binary_code


class ConvNeXtV2_FPN(nn.Module):
    def __init__(self, num_classes, input_channels=3, variant='tiny', return_features=False):
        """
        Args:
            num_classes: 输出的总通道数 (mask + binary_code = 1 + 48)
            input_channels: 输入图片通道数
            variant: 规格
        """
        super().__init__()
        self.num_classes = num_classes
        self.return_features = return_features
        
        # 1. 配置 Backbone
        configs = {
            'tiny':  {'depths': [3, 3, 9, 3], 'dims': [96, 192, 384, 768]},
            'base':  {'depths': [3, 3, 27, 3], 'dims': [128, 256, 512, 1024]},
            'large': {'depths': [3, 3, 27, 3], 'dims': [192, 384, 768, 1536]},
        }
        
        cfg = configs.get(variant, configs['tiny'])
        self.backbone = ConvNeXtV2(
            input_channels, 
            depths=cfg['depths'], 
            dims=cfg['dims']
        )
        
        # 2. 使用 FPN 模块替换原来的 ASPP + DecoderBlock
        # 这里 out_channels 设为 64，意味着融合前有 64*4=256 通道
        self.neck = FPN_Fusion(dims=cfg['dims'], out_channels=64)
        
        # # Final Head (从 FPN 输出的 256 通道映射到 num_classes)
        # self.final_conv = nn.Sequential(
        #     nn.Conv2d(256, num_classes, 1),
        #     # 这里不需要 Activation，因为 Mask 需要 Sigmoid，Code 需要 Linear/Identity
        #     # 具体的激活在 Loss 计算或推理时处理
        # )
        self.scale_branch = nn.Linear(256, 2)
        
        if self.return_features:
            # HCCE Head: 负责 Mask(1) + Code(48)
            self.hcce_head = nn.Sequential(
                nn.Conv2d(256, self.num_classes - 4, kernel_size=1)
            )
            # Uncertainty Head: 负责 w2d (4个通道：正反面各2)
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(256, 4, kernel_size=1)
            )
        else:
            self.hcce_head = nn.Sequential(
                nn.Conv2d(256, self.num_classes, kernel_size=1)
            )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        neck_out = self.neck(features) # [B, 256, H/4, W/4]
        
        # 分别投影
        hcce_out = self.hcce_head(neck_out)
        
        # 上采样到目标分辨率 (128x128)
        target_shape = (input_shape[0] // 2, input_shape[1] // 2)
        hcce_out = F.interpolate(hcce_out, size=target_shape, mode='bilinear', align_corners=False)
        
        
        if self.return_features:
            w2d_out = self.uncertainty_head(neck_out)
            
            mask, binary_code = torch.split(hcce_out, [1, self.num_classes - 5], 1)
            w2d_out = F.interpolate(w2d_out, size=target_shape, mode='bilinear', align_corners=False)
            # 计算 scale (EPro-PnP 特有)
            # 这里的 neck_out_features.flatten(2).mean(dim=-1) 是常用的全局池化特征
            scale_feat = neck_out.flatten(2).mean(dim=-1)
            scale = self.scale_branch(scale_feat).exp()
            scale = torch.clamp(scale, min=1e-4, max=6.0)

            return mask, binary_code, w2d_out, scale, neck_out
        
        mask, binary_code = torch.split(hcce_out, [1, self.num_classes - 1], 1)
        return mask, binary_code
        
        
        # # x shape: [B, 3, H, W]
        # input_shape = x.shape[-2:]
        
        # # 1. Backbone
        # # features: [c1(S4), c2(S8), c3(S16), c4(S32)]
        # features = self.backbone(x)
        
        # # 2. FPN Neck (替代了原来的 ASPP 和 级联 Decoder)
        # # 输出尺寸为 Stride 4 (H/4, W/4)
        # neck_out_features = self.neck(features)
        
        # # 3. Final Projection
        # output = self.final_conv(neck_out_features)
        
        # # 4. Final Upsample to Target Resolution
        # # 目标是输入的一半 (H/2, W/2)，匹配 GT (128x128 for 256x256 input)
        # target_shape = (input_shape[0] // 2, input_shape[1] // 2)
        # output = F.interpolate(output, size=target_shape, mode='bilinear', align_corners=False)
        
        # # 5. Split
        # mask, binary_code = torch.split(output, [1, self.num_classes - 1], 1)
        
        # if self.return_features:
        #     hcce_out = F.interpolate(hcce_out, size=target_shape, mode='bilinear', align_corners=False)
        #     w2d_out = F.interpolate(w2d_out, size=target_shape, mode='bilinear', align_corners=False)
        
        #     return mask, binary_code, neck_out_features
        
        # return mask, binary_code
