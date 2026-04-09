from .network_model import *
from .models.film_layer import FiLMLayer

class Multi_HccePose_EPro_Net(HccePose_BF_Net):
    def __init__(
        self,
        net='convnext', 
        input_channels=3, 
        output_channels=53,  # mask 1 + front_code 24 + back_code 24 + front_uncertainty 2 + back_uncertainty 2
        num_classes_all=5,
        **kwargs
    ):
        super().__init__(net, input_channels, output_channels, None, None, return_features=True, **kwargs)
        self.epropnp = EProPnP6DoF(
            mc_samples=512,
            num_iter=4,
            solver=LMSolver(
                dof=6,
                num_iter=5,
                init_solver=RSLMSolver(
                    dof=6,
                    num_points=16,
                    num_proposals=4,
                    num_iter=3)))
        self.camera = PerspectiveCamera
        self.emb_dim = 64
        self.obj_emb = nn.Embedding(num_classes_all, self.emb_dim)
        self.film = FiLMLayer(self.emb_dim, 256) # 特征图通道 256
        
    def forward(self, inputs, obj_ids):
        """
        inputs:\n
        obj_ids: torch.Tensor[torch.int64]
        """
        
        input_shape = inputs.shape[-2:]
        
        features = self.net.backbone(inputs)
        neck_out = self.net.neck(features) # [B, 256, H/4, W/4]
        
        # FiLM 
        emb = self.obj_emb(obj_ids)
        modulated_feat = self.film(neck_out, emb)
        
        hcce_out = self.net.hcce_head(modulated_feat)
        target_shape = (input_shape[0] // 2, input_shape[1] // 2)
        hcce_out = F.interpolate(hcce_out, size=target_shape, mode='bilinear', align_corners=False)
        
        if self.net.return_features:
            w2d_out = self.net.uncertainty_head(modulated_feat)
            mask, binary_code = torch.split(hcce_out, [1, self.net.num_classes - 5], 1)
            w2d_out = F.interpolate(w2d_out, size=target_shape, mode='bilinear', align_corners=False)
            scale_feat = modulated_feat.flatten(2).mean(dim=-1)
            scale = self.net.scale_branch(scale_feat).exp()
            scale = torch.clamp(scale, min=1e-4, max=6.0)

            return mask, binary_code, w2d_out, scale
        
        mask, binary_code = torch.split(hcce_out, [1, self.net.num_classes - 1], 1)
        return mask, binary_code
    
    def get_diff_coords(self, pred_code, batch_size_xyz, batch_min_xyz):
        """
        根据当前 Batch 中每个样本的尺寸还原坐标
        batch_size_xyz: [B, 3]
        batch_min_xyz: [B, 3]
        """
        prob_code = torch.sigmoid(pred_code)
        mid = prob_code.shape[1] // 2
        prob_f = prob_code[:, :mid, :, :].permute(0, 2, 3, 1) # [B, H, W, 24]
        prob_b = prob_code[:, mid:, :, :].permute(0, 2, 3, 1) # [B, H, W, 24]
        # 解码得到 0~1 的相对坐标 [B, H, W, 3]
        coords_f = self.hcce_decode(prob_f) / 255.0
        coords_b = self.hcce_decode(prob_b) / 255.0
        # 还原 [B, H, W, 3] * [B, 1, 1, 3] + [B, 1, 1, 3]
        x3d_f = coords_f * batch_size_xyz[:, None, None, :] + batch_min_xyz[:, None, None, :]
        x3d_b = coords_b * batch_size_xyz[:, None, None, :] + batch_min_xyz[:, None, None, :]
        
        return x3d_f, x3d_b
    
    def inference_batch(self, inputs, Bbox, obj_ids, batch_size_xyz, batch_min_xyz, threshold=0.5):
        # 推理时也需要传入该类别对应的 size/min
        pred_mask, pred_fb_code, w2d, scale = self.forward(inputs, obj_ids)
        pred_mask = self.activation_function(pred_mask)
        pred_mask[pred_mask > threshold] = 1.0
        pred_mask[pred_mask <= threshold] = 0.0
        pred_mask = pred_mask[:, 0, ...]
        
        pred_front_code_raw = ((pred_fb_code.permute(0, 2, 3, 1)+1)/2).clone().clamp(0,1)[...,:24]
        pred_back_code_raw = ((pred_fb_code.permute(0, 2, 3, 1)+1)/2).clone().clamp(0,1)[...,24:]
        
        # 解码并还原
        pred_fb_code_sig = self.activation_function(pred_fb_code).permute(0, 2, 3, 1)
        pred_front_norm = self.hcce_decode(pred_fb_code_sig[..., :24]) / 255.0
        pred_back_norm = self.hcce_decode(pred_fb_code_sig[..., 24:]) / 255.0
        
        pred_front_obj = pred_front_norm * batch_size_xyz[:, None, None, :] + batch_min_xyz[:, None, None, :]
        pred_back_obj = pred_back_norm * batch_size_xyz[:, None, None, :] + batch_min_xyz[:, None, None, :]
        
        x = torch.arange(pred_front_norm.shape[2] , device=pred_front_norm.device).to(torch.float32) / pred_front_norm.shape[2] 
        y = torch.arange(pred_front_norm.shape[1] , device=pred_front_norm.device).to(torch.float32) / pred_front_norm.shape[1] 
        X, Y = torch.meshgrid(x, y, indexing='xy')  
        coord_image = torch.cat([X[..., None], Y[..., None]], dim=-1) 
        coord_image = coord_image[None,...].repeat(pred_front_norm.shape[0],1,1,1)
        coord_image[..., 0] = coord_image[..., 0] * Bbox[:, None, None, 2] + Bbox[:, None, None, 0]
        coord_image[..., 1] = coord_image[..., 1] * Bbox[:, None, None, 3] + Bbox[:, None, None, 1]
        
        w2d_f = w2d[:, :2 ,: ,:]
        w2d_b = w2d[:, 2: ,: ,:]
        w2d_f = w2d_f.permute(0, 2, 3, 1)
        w2d_b = w2d_b.permute(0, 2, 3, 1)
        
        return {
            'pred_mask' : pred_mask,
            'coord_2d_image' : coord_image,
            'pred_front_code_obj' : pred_front_obj,
            'pred_back_code_obj' : pred_back_obj,
            'pred_front_code' : pred_front_norm,
            'pred_back_code' : pred_back_norm,
            'pred_front_code_raw' : pred_front_code_raw,
            'pred_back_code_raw' : pred_back_code_raw,
            'w2d_front': w2d_f,
            'w2d_back': w2d_b,
            'scale': scale,
        }
    
    
    
class MultiHead_HccePose_EPro_Net(HccePose_BF_Net):
    def __init__(
        self,
        net='convnext', 
        input_channels=3, 
        output_channels=53,  # mask 1 + front_code 24 + back_code 24 + front_uncertainty 2 + back_uncertainty 2
        target_obj_ids:list=None,
        **kwargs
    ):
        super().__init__(net, input_channels, output_channels, None, None, return_features=True, **kwargs)
        self.epropnp = EProPnP6DoF(
            mc_samples=512,
            num_iter=4,
            solver=LMSolver(
                dof=6,
                num_iter=5,
                init_solver=RSLMSolver(
                    dof=6,
                    num_points=16,
                    num_proposals=4,
                    num_iter=3)))
        self.camera = PerspectiveCamera
        
        self.obj_ids = target_obj_ids
        self.hcce_heads = nn.ModuleDict({
            str(o_id): nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, output_channels - 4, kernel_size=1)
            )for o_id in target_obj_ids
            # nn.Conv2d(256, 4, kernel_size=1) for o_id in target_obj_ids
        })
        self.uncertainty_heads = nn.ModuleDict({
            str(o_id): nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 4, kernel_size=1)
            )for o_id in target_obj_ids
            # nn.Conv2d(256, 4, kernel_size=1) for o_id in target_obj_ids
        })
        self.scale_branches = nn.ModuleDict({
            str(o_id): nn.Linear(256, 2) for o_id in target_obj_ids
        })
        
    def forward(self, x, obj_ids):
        """
        x:\n
        obj_ids: torch.Tensor[torch.int64]
        """
        
        input_shape = x.shape[-2:]
        target_shape = (input_shape[0] // 2, input_shape[1] // 2)
        
        features = self.net.backbone(x)
        neck_out = self.net.neck(features) # [B, 256, H/4, W/4]
        
        B, C, H, W = neck_out.shape
        final_hcce = torch.zeros((B, 49, H, W), device=x.device, dtype=x.dtype)
        final_w2d = torch.zeros((B, 4, H, W), device=x.device, dtype=x.dtype)
        final_scale = torch.zeros((B, 2), device=x.device, dtype=x.dtype)
        
        unique_ids = torch.unique(obj_ids)
        for o_id in unique_ids:
            mask = (obj_ids == o_id)
            id_str = str(int(o_id))
            
            # 局部特征
            sub_feat = neck_out[mask]
            
            # 使用对应类别的 Head 处理
            final_hcce[mask] = self.hcce_heads[id_str](sub_feat).to(final_hcce.dtype)
            final_w2d[mask] = self.uncertainty_heads[id_str](sub_feat).to(final_hcce.dtype)
            
            # Scale 处理
            scale_feat = sub_feat.flatten(2).mean(dim=-1)
            final_scale[mask] = self.scale_branches[id_str](scale_feat).exp().to(final_hcce.dtype)
            
        final_hcce = F.interpolate(final_hcce, size=target_shape, mode='bilinear', align_corners=False)
        final_w2d = F.interpolate(final_w2d, size=target_shape, mode='bilinear', align_corners=False)
        
        pred_mask, pred_code = torch.split(final_hcce, [1, 48], 1)
        final_scale = torch.clamp(final_scale, min=1e-4, max=6.0)

        return pred_mask, pred_code, final_w2d, final_scale
    
    def get_diff_coords(self, pred_code, batch_size_xyz, batch_min_xyz):
        """
        根据当前 Batch 中每个样本的尺寸还原坐标
        batch_size_xyz: [B, 3]
        batch_min_xyz: [B, 3]
        """
        prob_code = torch.sigmoid(pred_code)
        mid = prob_code.shape[1] // 2
        prob_f = prob_code[:, :mid, :, :].permute(0, 2, 3, 1) # [B, H, W, 24]
        prob_b = prob_code[:, mid:, :, :].permute(0, 2, 3, 1) # [B, H, W, 24]
        # 解码得到 0~1 的相对坐标 [B, H, W, 3]
        coords_f = self.hcce_decode(prob_f) / 255.0
        coords_b = self.hcce_decode(prob_b) / 255.0
        # 还原 [B, H, W, 3] * [B, 1, 1, 3] + [B, 1, 1, 3]
        x3d_f = coords_f * batch_size_xyz[:, None, None, :] + batch_min_xyz[:, None, None, :]
        x3d_b = coords_b * batch_size_xyz[:, None, None, :] + batch_min_xyz[:, None, None, :]
        
        return x3d_f, x3d_b
    
    def inference_batch(self, inputs, Bbox, obj_ids, batch_size_xyz, batch_min_xyz, threshold=0.5):
        # 推理时也需要传入该类别对应的 size/min
        pred_mask, pred_fb_code, w2d, scale = self.forward(inputs, obj_ids)
        pred_mask = self.activation_function(pred_mask)
        pred_mask[pred_mask > threshold] = 1.0
        pred_mask[pred_mask <= threshold] = 0.0
        pred_mask = pred_mask[:, 0, ...]
        
        pred_front_code_raw = ((pred_fb_code.permute(0, 2, 3, 1)+1)/2).clone().clamp(0,1)[...,:24]
        pred_back_code_raw = ((pred_fb_code.permute(0, 2, 3, 1)+1)/2).clone().clamp(0,1)[...,24:]
        
        # 解码并还原
        pred_fb_code_sig = self.activation_function(pred_fb_code).permute(0, 2, 3, 1)
        pred_front_norm = self.hcce_decode(pred_fb_code_sig[..., :24]) / 255.0
        pred_back_norm = self.hcce_decode(pred_fb_code_sig[..., 24:]) / 255.0
        
        pred_front_obj = pred_front_norm * batch_size_xyz[:, None, None, :] + batch_min_xyz[:, None, None, :]
        pred_back_obj = pred_back_norm * batch_size_xyz[:, None, None, :] + batch_min_xyz[:, None, None, :]
        
        x = torch.arange(pred_front_norm.shape[2] , device=pred_front_norm.device).to(torch.float32) / pred_front_norm.shape[2] 
        y = torch.arange(pred_front_norm.shape[1] , device=pred_front_norm.device).to(torch.float32) / pred_front_norm.shape[1] 
        X, Y = torch.meshgrid(x, y, indexing='xy')  
        coord_image = torch.cat([X[..., None], Y[..., None]], dim=-1) + 0.5
        coord_image = coord_image[None,...].repeat(pred_front_norm.shape[0],1,1,1)
        coord_image[..., 0] = coord_image[..., 0] * Bbox[:, None, None, 2] + Bbox[:, None, None, 0]
        coord_image[..., 1] = coord_image[..., 1] * Bbox[:, None, None, 3] + Bbox[:, None, None, 1]
        
        w2d_f = w2d[:, :2 ,: ,:]
        w2d_b = w2d[:, 2: ,: ,:]
        w2d_f = w2d_f.permute(0, 2, 3, 1)
        w2d_b = w2d_b.permute(0, 2, 3, 1)
        
        return {
            'pred_mask' : pred_mask,
            'coord_2d_image' : coord_image,
            'pred_front_code_obj' : pred_front_obj,
            'pred_back_code_obj' : pred_back_obj,
            'pred_front_code' : pred_front_norm,
            'pred_back_code' : pred_back_norm,
            'pred_front_code_raw' : pred_front_code_raw,
            'pred_back_code_raw' : pred_back_code_raw,
            'w2d_front': w2d_f,
            'w2d_back': w2d_b,
            'scale': scale,
        }
    