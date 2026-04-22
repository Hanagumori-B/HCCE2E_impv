import os
import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .models import *
from .network_model import HccePose_BF_Net, HccePose_Loss
from .tools.losses import angular_distance_quat, PyPMLoss
from epropnp.epropnp import EProPnP6DoF
from epropnp.levenberg_marquardt import LMSolver, RSLMSolver
from epropnp.cost_fun import AdaptiveHuberPnPCost
from epropnp.camera import PerspectiveCamera
from epropnp.monte_carlo_pose_loss import MonteCarloPoseLoss
from epropnp.common import evaluate_pnp
from .tools.rot_reps import rot6d_to_mat_batch, mat2quat_batch
from .tools.t_site_tools import trans_to_site_batch, site_to_trans_batch


@dataclass
class ModelPred:
    pred_front: torch.Tensor
    pred_back: torch.Tensor
    pred_mask: torch.Tensor
    pred_rot_6d: torch.Tensor
    pred_t_site: torch.Tensor
    pred_coords_f: torch.Tensor
    pred_coords_b: torch.Tensor

@dataclass
class GroundTruth:
    gt_front: torch.Tensor
    gt_back: torch.Tensor
    gt_mask: torch.Tensor
    gt_rot_mat: torch.Tensor
    gt_t_site: torch.Tensor


class HccePose_PatchPnP_EPro_Loss(HccePose_Loss):
    def __init__(self, size_xyz=None, symmetric=False):
        super().__init__()
        self.size_xyz=size_xyz
        self.pm_loss = PyPMLoss(r_only=True, norm_by_extent=True, symmetric=symmetric)
        self.coord_front_loss = nn.L1Loss(reduction='none')
        self.coord_back_loss = nn.L1Loss(reduction='none')
        self.monte_carlo_pose_loss = MonteCarloPoseLoss()
        
    def forward(self, epropnp:EProPnP6DoF, model_out, 
                coords_3d:tuple[torch.Tensor, torch.Tensor], 
                coords_2d: torch.Tensor, 
                targets, cam_K, bbox, 
                preds: ModelPred, gts: GroundTruth,
                model_points, net_ref,
                out_res=128, num_sample=1024, 
                weight_sample=None, pose_perturb_std=None,
                sym_infos=None
                ):...


class HccePose_PatchPnP_EPro_Net(HccePose_BF_Net):
    def __init__(self, 
                net='resnet', 
                input_channels=3, 
                output_channels=53,  # mask 1 + front_code 24 + back_code 24 + front_uncertainty 2 + back_uncertainty 2
                feat_dim=128,
                fb_type = "fb", # fb | f(b-f)
                **kwargs):
        # HCCEPose 输出: 1(mask) + 24(front) + 24(back) = 49 channels
        super().__init__(net, input_channels, output_channels, return_features=True, **kwargs)
        self.decode_net = HcceDecodeNet()
        self.pnp_net = PatchPnPNet(
            in_channels=8,  # Front + Back + Coord2d
            feat_dim=feat_dim, 
            rot_dim=6, 
            mask_attention_type="concat", # none | mul | concat
            denormalize_by_extent=True,
        )
        self.fb_type = fb_type
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
        self.cost_func = AdaptiveHuberPnPCost
    
    def _forward_regression(self, inputs, cam_K, Bbox, img_sz, s_zoom):
        """ Backbone 特征提取与 PatchPnP 姿态回归 """
        # 网络基础输出
        pred_mask_logits, pred_codes_logits, w2d, scale, _ = self.net(inputs)
        
        # 3D 坐标解码
        prob_codes = torch.sigmoid(pred_codes_logits)
        prob_f, prob_b = prob_codes[:, :24, ...], prob_codes[:, 24:, ...]
        norm_coords_f = self.decode_net(prob_f)
        norm_coords_b = self.decode_net(prob_b)
        
        # 2D 坐标计算 (用于 PatchPnP 输入和后续 EProPnP)
        B, _, H, W = norm_coords_f.shape
        device = norm_coords_f.device
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        coord2d = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        bbox_origin, bbox_size = Bbox[:, :2, None, None], Bbox[:, 2:, None, None]
        x2d_pixel = coord2d * bbox_size + bbox_origin
        img_wh = img_sz[:, [1, 0], None, None] 
        x2d_norm = x2d_pixel / img_wh

        # PatchPnPNet 输入准备与回归
        # 拼接前后层坐标 [B, 6, H, W]
        if self.fb_type == "fb":
            pnp_input = torch.cat([norm_coords_f, norm_coords_b], dim=1)
        elif self.fb_type == "f(b-f)":
            pnp_input = torch.cat([norm_coords_f, norm_coords_b - norm_coords_f], dim=1)
        pnp_input = torch.cat([pnp_input, x2d_norm], dim=1)
        
        pred_mask_prob = torch.sigmoid(pred_mask_logits)
        batch_extents = self.size_xyz.unsqueeze(0).expand(B, -1)
        
        pred_rot_6d, pred_t_site = self.pnp_net(
            pnp_input, 
            extents=batch_extents,
            mask_attention=pred_mask_prob.detach()
        )
        
        # 格式转换
        pred_rot_quat = mat2quat_batch(rot6d_to_mat_batch(pred_rot_6d))
        pred_trans = site_to_trans_batch(pred_t_site, cam_K, Bbox, s_zoom)
        
        return {
            'pred_mask_prob': pred_mask_prob,
            'norm_coords_f': norm_coords_f,
            'norm_coords_b': norm_coords_b,
            'x2d_pixel': x2d_pixel,
            'w2d': w2d,
            'scale': scale,
            'pred_pose': torch.cat([pred_trans, pred_rot_quat], dim=1),
            'patch_pnp_raw': (pred_rot_6d, pred_t_site), # 用于辅助 Loss
            'prob_codes': (prob_f, prob_b) # 用于可视化
        }
        
    def _forward_probabilistic(self, reg_data, cam_K, Bbox):
        """ EPro-PnP 数据准备与求解 """
        B, _, H, W = reg_data['norm_coords_f'].shape
        
        # 1. 3D 反归一化
        x3d_f = reg_data['norm_coords_f'] * self.size_xyz[None, :, None, None] + self.min_xyz[None, :, None, None]
        x3d_b = reg_data['norm_coords_b'] * self.size_xyz[None, :, None, None] + self.min_xyz[None, :, None, None]
        
        # 2. 展平与拼接坐标
        x3d_all = torch.cat([
            x3d_f.permute(0, 2, 3, 1).reshape(B, -1, 3),
            x3d_b.permute(0, 2, 3, 1).reshape(B, -1, 3)
        ], dim=1)
        
        x2d_flat = reg_data['x2d_pixel'].permute(0, 2, 3, 1).reshape(B, -1, 2)
        x2d_all = torch.cat([x2d_flat, x2d_flat], dim=1)
        
        # 3. 权重处理
        mask_flat = reg_data['pred_mask_prob'].reshape(B, -1, 1)
        num_pts = float(x2d_flat.size(1))
        log_N = math.log(num_pts)
        
        w2d_raw = reg_data['w2d']
        w2d_f_logit = w2d_raw[:, :2, ...].permute(0, 2, 3, 1).reshape(B, -1, 2)
        w2d_b_logit = w2d_raw[:, 2:4, ...].permute(0, 2, 3, 1).reshape(B, -1, 2)
        
        # 注意：正面用 scale[0], 背面用 scale[1]
        w2d_f = self.process_weight(w2d_f_logit, log_N, reg_data['scale'][:, 0:1], mask_flat)
        w2d_b = self.process_weight(w2d_b_logit, log_N, reg_data['scale'][:, 1:2], mask_flat)
        w2d_all = torch.cat([w2d_f, w2d_b], dim=1)
        
        # 4. 配置 Camera 和 Cost Function
        wh_begin = Bbox[:, 0:2]
        wh_unit = (Bbox[:, 2] / float(W)).clamp(min=1e-5)
        allowed_border = 30 * wh_unit[:, None]
        
        camera = self.camera_class(
            cam_mats=cam_K, 
            z_min=0.01, 
            lb=wh_begin - allowed_border, 
            ub=wh_begin + Bbox[:, 2:4] + allowed_border
        )
        cost_func = self.cost_func_class(relative_delta=0.1)
        cost_func.set_param(x2d_all, w2d_all)
        
        return x3d_all, x2d_all, w2d_all, camera, cost_func

    def forward(self, inputs, cam_K, Bbox, img_sz, s_zoom=128.0, out_vis=False):
        # 回归
        reg_data = self._forward_regression(inputs, cam_K, Bbox, img_sz, s_zoom)
        
        preds = ModelPred(
            pred_front=reg_data['norm_coords_f'], # 归一化3D坐标
            pred_back=reg_data['norm_coords_b'],
            pred_mask=reg_data['pred_mask_prob'],
            pred_rot_6d=reg_data['patch_pnp_raw'][0], # R6D 预测值
            pred_t_site=reg_data['patch_pnp_raw'][1], # SITE 预测值
            pred_coords_f=reg_data['prob_codes'][0], # HCCE 表面预测值
            pred_coords_b=reg_data['prob_codes'][1]
        )
        
        # 准备 PnP 数据
        x3d_all, x2d_all, w2d_all, camera, cost_func = self._forward_probabilistic(reg_data, cam_K, Bbox)
        
        epro_data = {
            'x3d': x3d_all, 'x2d': x2d_all, 'w2d': w2d_all,
            'camera': camera, 'cost_func': cost_func,
            'pose_init': reg_data['pred_pose'],
            'scale': reg_data['scale']
        }

        if self.training:
            # 训练模式：返回所有中间变量供 Loss 函数调用 monte_carlo_forward
            return preds, epro_data
        else:
            # 推理模式：直接求解
            pose_opt, _, _, _ = self.epropnp(
                x3d_all, x2d_all, w2d_all,
                camera, cost_func, pose_init=reg_data['pred_pose']
            )
            if out_vis:
                return pose_opt, preds
            return pose_opt

        
    @staticmethod
    def process_weight(logit, log_N, scale, mask_f):
            # 归一化 (类似 Softmax 思想但保留 scale 控制权)
            w = (logit - logit.mean(dim=1, keepdim=True) - log_N).exp()
            w = w * scale[:, None, :] 
            w = torch.nan_to_num(w, nan=0.0, posinf=1e5, neginf=0.0)
            # Mask 过滤
            w = w * mask_f.float()
            # 防止权重完全为 0 导致矩阵奇异
            return w.clamp(min=1e-12)
