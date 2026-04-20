import os
import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import *
from .tools.losses import angular_distance_quat, PyPMLoss
from epropnp.epropnp import EProPnP6DoF
from epropnp.levenberg_marquardt import LMSolver, RSLMSolver
from epropnp.cost_fun import AdaptiveHuberPnPCost
from epropnp.camera import PerspectiveCamera
from epropnp.monte_carlo_pose_loss import MonteCarloPoseLoss
from epropnp.common import evaluate_pnp
from .tools.rot_reps import rot6d_to_mat_batch, mat2quat_batch
from .tools.t_site_tools import trans_to_site_batch, site_to_trans_batch


class FixedSizeList:
    def __init__(self, size):
        self.size = size
        self.data = []
    
    def append(self, item):
        if len(self.data) >= self.size:
            self.data.pop(0)  
        self.data.append(item)
    
    def get_list(self):
        return self.data
    
    def __repr__(self):
        return repr(self.data)


# =====================================================================
#                               Losses
# =====================================================================


class HccePose_Loss(nn.Module):
    def __init__(self, ):
        
        super().__init__()
        
        self.Front_error_list = [[], [], []]
        self.Back_error_list = [[], [], []]
        
        self.current_front_error_ratio = [None, None, None]
        self.current_back_error_ratio = [None, None, None]
        
        self.weight_front_error_ratio = [None, None, None]
        self.weight_back_error_ratio = [None, None, None]
        
        self.Front_L1Loss = nn.L1Loss(reduction='none')
        self.Back_L1Loss = nn.L1Loss(reduction='none')
        
        self.mask_loss = nn.L1Loss(reduction="mean")
        
        self.activation_function = torch.nn.Sigmoid()
        
        pass
    
    def cal_error_ratio(self, pred_code, gt_code, pred_mask):
        pred_mask  = pred_mask.clone().detach().round().clamp(0,1) 
        pred_code = torch.sigmoid(pred_code).clone().detach().round().clamp(0,1)
        gt_code = gt_code.clone().detach().round().clamp(0,1)
        error = torch.abs(pred_code-gt_code)*pred_mask
        error_ratio = error.sum([0,2,3])/(pred_mask.sum()+1)
        return error_ratio
    
    def print_error_ratio(self, ):
        if self.weight_front_error_ratio[0] is not None:
            np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
            print('front(x) error: {}'.format(self.weight_front_error_ratio[0].detach().cpu().numpy()))
        if self.weight_front_error_ratio[1] is not None:
            np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
            print('front(y) error:{}'.format(self.weight_front_error_ratio[1].detach().cpu().numpy()))
        if self.weight_front_error_ratio[2] is not None:
            np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
            print('front(z) error:{}'.format(self.weight_front_error_ratio[2].detach().cpu().numpy()))

        if self.weight_back_error_ratio[0] is not None:
            np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
            print('back(x) error:{}'.format(self.weight_back_error_ratio[0].detach().cpu().numpy()))
        if self.weight_back_error_ratio[1] is not None:
            np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
            print('back(y) error:{}'.format(self.weight_back_error_ratio[1].detach().cpu().numpy()))
        if self.weight_back_error_ratio[2] is not None:
            np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
            print('back(z) error:{}'.format(self.weight_back_error_ratio[2].detach().cpu().numpy()))
        return
    
    def forward(self, pred_front, pred_back, pred_mask, gt_front, gt_back, gt_mask,):
        pred_mask_for_loss = pred_mask[:, 0, :, :]
        pred_mask_for_loss = torch.sigmoid(pred_mask_for_loss)
        mask_loss_v = self.mask_loss(pred_mask_for_loss, gt_mask)
        
        if gt_mask.dim() == 3:
            target_mask = gt_mask.unsqueeze(1)
        else:
            target_mask = gt_mask
        target_mask = target_mask.detach().float()
        
        pred_mask_prob = self.activation_function(pred_mask)
        pred_mask_binary = pred_mask_prob.detach().clone().round().clamp(0,1)
        # pred_mask = pred_mask_prob
        
        Front_L1Loss_v_l = []
        Back_L1Loss_v_l = []
        
        for k in range(3):
            front_error_ratio = self.cal_error_ratio(pred_front[:, k*8:(k+1)*8], gt_front[:, k*8:(k+1)*8], pred_mask_binary)
            self.current_front_error_ratio[k] = front_error_ratio.clone().detach()
            if self.weight_front_error_ratio[k] is None:
                self.weight_front_error_ratio[k]  = front_error_ratio.clone().detach()
                for i in range(pred_front[:, k*8:(k+1)*8].shape[1]):
                    self.Front_error_list[k].append(FixedSizeList(100))
                    self.Front_error_list[k][i].append(self.current_front_error_ratio[k][i].cpu().numpy())
            else:
                for i in range(pred_front[:, k*8:(k+1)*8].shape[1]):
                    self.Front_error_list[k][i].append(self.current_front_error_ratio[k][i].cpu().numpy())
                    self.weight_front_error_ratio[k][i] = np.mean(self.Front_error_list[k][i].data)
                
            back_error_ratio = self.cal_error_ratio(pred_back[:, k*8:(k+1)*8], gt_back[:, k*8:(k+1)*8], pred_mask_binary)
            self.current_back_error_ratio[k] = back_error_ratio.clone().detach()
            if self.weight_back_error_ratio[k] is None:
                self.weight_back_error_ratio[k]  = back_error_ratio.clone().detach()
                for i in range(pred_back[:, k*8:(k+1)*8].shape[1]):
                    self.Back_error_list[k].append(FixedSizeList(100))
                    self.Back_error_list[k][i].append(self.current_back_error_ratio[k][i].cpu().numpy())
            else:
                for i in range(pred_back[:, k*8:(k+1)*8].shape[1]):
                    self.Back_error_list[k][i].append(self.current_back_error_ratio[k][i].cpu().numpy())
                    self.weight_back_error_ratio[k][i] = np.mean(self.Back_error_list[k][i].data)
        
        
            weight_front_error_ratio = torch.exp(torch.minimum(self.weight_front_error_ratio[k],0.51-self.weight_front_error_ratio[k]) * 3).detach().clone()
            weight_back_error_ratio = torch.exp(torch.minimum(self.weight_back_error_ratio[k],0.51-self.weight_back_error_ratio[k]) * 3).detach().clone()
        
            # Front_L1Loss_v = self.Front_L1Loss(pred_front[:, k*8:(k+1)*8]*pred_mask.detach().clone(),(gt_front[:, k*8:(k+1)*8] *2 -1)*pred_mask.detach().clone())
            # Front_L1Loss_v = Front_L1Loss_v.mean([0,2,3])
            # Front_L1Loss_v = torch.sum(Front_L1Loss_v*weight_front_error_ratio)/torch.sum(weight_front_error_ratio)
        
            # Back_L1Loss_v = self.Back_L1Loss(pred_back[:, k*8:(k+1)*8]*pred_mask.detach().clone(),(gt_back[:, k*8:(k+1)*8] *2 -1)*pred_mask.detach().clone())
            # Back_L1Loss_v = Back_L1Loss_v.mean([0,2,3])
            # Back_L1Loss_v = torch.sum(Back_L1Loss_v*weight_back_error_ratio)/torch.sum(weight_back_error_ratio)
            
            Front_L1Loss_v = self.Front_L1Loss(pred_front[:, k*8:(k+1)*8]*target_mask.clone(),(gt_front[:, k*8:(k+1)*8] *2 -1)*target_mask)
            Front_L1Loss_v = Front_L1Loss_v.mean([0,2,3])
            Front_L1Loss_v = torch.sum(Front_L1Loss_v*weight_front_error_ratio)/(torch.sum(weight_front_error_ratio) + 1e-7)
        
            Back_L1Loss_v = self.Back_L1Loss(pred_back[:, k*8:(k+1)*8]*target_mask.clone(),(gt_back[:, k*8:(k+1)*8] *2 -1)*target_mask)
            Back_L1Loss_v = Back_L1Loss_v.mean([0,2,3])
            Back_L1Loss_v = torch.sum(Back_L1Loss_v*weight_back_error_ratio)/(torch.sum(weight_back_error_ratio) + 1e-7)

            Front_L1Loss_v_l.append(Front_L1Loss_v[None])
            Back_L1Loss_v_l.append(Back_L1Loss_v[None])
        
        Front_L1Loss_v_l = torch.cat(Front_L1Loss_v_l, dim = 0).view(-1)
        Back_L1Loss_v_l = torch.cat(Back_L1Loss_v_l, dim = 0).view(-1)
        
        return {
            'mask_loss' : mask_loss_v, 
            'Front_L1Losses' : Front_L1Loss_v_l,
            'Back_L1Losses' : Back_L1Loss_v_l,
        }


class HccePose_EPro_Loss(HccePose_Loss):
    def __init__(self, r_loss=False, t_loss=False):
        super(HccePose_EPro_Loss, self).__init__()
        self.monte_carlo_pose_loss = MonteCarloPoseLoss()
        self.r_loss = r_loss
        self.t_loss = t_loss
        
    def debug_single_batch(self, B, x3d_sampled, x2d_sampled, w2d_raw, w2d_norm, gt_pose, cam_K, pose_opt_plus, pose_sample_logweights, cost_tgt):
        def stats(t):
            return dict(shape=tuple(t.shape), mean=float(t.mean().cpu()), std=float(t.std().cpu()), min=float(t.min().cpu()), max=float(t.max().cpu()), has_nan=bool(torch.isnan(t).any().item()))
        print("=== DEBUG SINGLE BATCH ===")
        print("x3d_sampled stats:", stats(x3d_sampled))
        print("x2d_sampled stats:", stats(x2d_sampled))
        print("w2d_raw stats:", stats(w2d_raw))
        print("w2d_norm stats:", stats(w2d_norm))
        print("cam_K stats:", stats(cam_K))
        if pose_sample_logweights is not None:
            print("pose_sample_logweights stats:", stats(pose_sample_logweights))
        else:
            print("pose_sample_logweights is None")
        print("cost_tgt is None?", cost_tgt is None)
        if cost_tgt is not None:
            print("cost_tgt stats:", stats(cost_tgt))
        if pose_opt_plus is not None:
            print("pose_opt_plus stats:", stats(pose_opt_plus))
        print("gt_pose stats:", stats(gt_pose))
        print("=========================")
        
    def forward(self, epropnp:EProPnP6DoF, model_out, coords_3d:tuple[torch.Tensor, torch.Tensor], coords_2d: torch.Tensor, 
                targets, cam_K, bbox, out_res=128, num_sample=1024, weight_sample=None, pose_perturb_std=None):
        """
        epropnp: EProPnP6DoF 实例
        model_out: (pred_mask, pred_code, w2d, scale)
        coords_3d: result of net.get_diff_coords() ([B, H, W, 3]) front & back
        coords_2d: pred_results['coord_2d_image'] [B, H, W, 3]
        targets: (
            gt_mask, 
            gt_front,
            gt_back,
            gt_pose{torch.Tensor[[cam_t_m2c, gt_quat]]},
            obj_size, 
            obj_min
        )
        cam_K: 相机内参
        bbox: [x, y, w, h]
        out_res: 网络输出的分辨率
        """
        pred_mask, pred_code, w2d, scale = model_out
        w2d_f = w2d[:, :2, :, :].permute(0, 2, 3, 1)
        w2d_b = w2d[:, 2:, :, :].permute(0, 2, 3, 1)
        x3d_f, x3d_b = coords_3d
        gt_mask, gt_front, gt_back, gt_pose, obj_size, obj_min = targets
        
        bs = pred_mask.shape[0]
        
        # 基础 HCCE 坐标 Loss (保持原有监督)
        base_losses = super().forward(pred_code[:, :24], pred_code[:, 24:], pred_mask, gt_front, gt_back, gt_mask)
        
        B, H, W, _ = coords_2d.shape
        x2d_roi = coords_2d.reshape(B, -1, 2)
        batch_idx = torch.arange(B, device=x3d_f.device).view(-1, 1)
        
        gt_mask_flat = gt_mask.reshape(B, -1).float()
        pred_mask_flat = pred_mask.detach().reshape(B, -1).float().clamp(min=0.0)
        eps = 1e-6
        
        if not weight_sample: weight_sample = [0.1, 0.45, 0.45] # 从[均匀,gt_mask,pred_mask]采样
        p_uniform = torch.ones_like(gt_mask_flat) / (H * W)
        prob = weight_sample[0] * p_uniform + weight_sample[1] * gt_mask_flat + weight_sample[2] * pred_mask_flat + eps
        
        # prob = gt_mask_flat + eps # 从gt_mask采样
        prob = prob / (prob.sum(dim=1, keepdim=True) + eps)
        idx_f = torch.multinomial(prob, num_sample, replacement=True)
        idx_b = torch.multinomial(prob, num_sample, replacement=True)
        
        # 正面采样
        x3d_f_flat = x3d_f.reshape(B, H*W, 3)
        x2d_f_flat = x2d_roi.reshape(B, H*W, 2)
        w2d_f_flat = w2d_f.reshape(B, H*W, 2)

        # rand_f = torch.rand(B, H*W, device=x3d_f.device)
        # _, indices_f = torch.topk(rand_f, k=num_sample, dim=1)

        sampled_x3d_f = x3d_f_flat[batch_idx, idx_f, :]
        sampled_x2d_f = x2d_f_flat[batch_idx, idx_f, :]
        sampled_w2d_f = w2d_f_flat[batch_idx, idx_f, :]
        
        # 背面采样
        x3d_b_flat = x3d_b.reshape(B, H*W, 3)
        x2d_b_flat = x2d_roi.reshape(B, H*W, 2) # x2d 对背面是一样的坐标
        w2d_b_flat = w2d_b.reshape(B, H*W, 2)
        
        # rand_b = torch.rand(B, H*W, device=x3d_f.device)
        # _, indices_b = torch.topk(rand_b, k=num_sample, dim=1)

        sampled_x3d_b = x3d_b_flat[batch_idx, idx_b, :]
        sampled_x2d_b = x2d_b_flat[batch_idx, idx_b, :]
        sampled_w2d_b = w2d_b_flat[batch_idx, idx_b, :]
        
        x3d_sampled = torch.cat([sampled_x3d_f, sampled_x3d_b], dim=1).float() # [B, 2*HW, 3]
        x2d_sampled = torch.cat([sampled_x2d_f, sampled_x2d_b], dim=1).float() # [B, 2*HW, 2]
        w2d_sampled = torch.cat([sampled_w2d_f, sampled_w2d_b], dim=1).float() # [B, 2*HW, 2]
        gt_pose = gt_pose.float()
        wh_begin = bbox[:, 0:2]
        wh_unit = bbox[:, 2] / float(out_res)
        
        with torch.amp.autocast('cuda', enabled=False):
            allowed_border = 30 * wh_unit[:, None]
            
            w2d_norm = (w2d_sampled - w2d_sampled.mean(dim=1, keepdim=True) - math.log(w2d_sampled.size(1))).exp() * scale[:, None, :]
            # print(torch.where(w2d_norm.isinf()))
            # print(torch.where(w2d_norm.isnan()))
            # print(torch.where(w2d_norm<=0))
            # w2d_norm = torch.softmax(w2d_sampled, dim=1) * scale[:, None, :]
            
            camera = PerspectiveCamera(
                    cam_mats=cam_K.float(),
                    z_min=0.01,
                    lb=wh_begin - allowed_border,
                    ub=wh_begin + bbox[:, 2:4] + allowed_border)
            cost_func = AdaptiveHuberPnPCost(relative_delta=0.1)
            cost_func.set_param(x2d_sampled, w2d_norm)
            
            # 根据gt_pose计算的cost
            _, cost_gt, _ = evaluate_pnp(
                x3d_sampled, x2d_sampled, w2d_norm, gt_pose, 
                camera, cost_func, out_cost=True)
            
            pose_init = gt_pose.clone().detach()
            if pose_perturb_std:
                pose_init_perturbed = self._pose_perturb(pose_init, pose_perturb_std[:3], pose_perturb_std[3])
            else:
                pose_init_perturbed = pose_init

            # MonteCarlo 过程
            pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt = epropnp.monte_carlo_forward(
                x3d_sampled, x2d_sampled, w2d_norm, camera, cost_func,
                # pose_init=pose_init_perturbed,
                pose_init=pose_init_perturbed,
                force_init_solve=True, 
                with_pose_opt_plus=True,
                sample_conv_min_extrict=pose_perturb_std)
            
            with torch.no_grad():
                # pose_samples 形状: (mc_samples, B, 7)
                # 计算平移 (x, y, z) 在采样维度 (dim=0) 上的标准差
                # samples_std_t 形状: (B, 3)
                samples_std_t = torch.std(pose_samples[:, :, :3], dim=0) 
                
                # 计算旋转 (四元数) 的标准差作为离散度参考
                samples_std_r = torch.std(pose_samples[:, :, 3:], dim=0) # (B, 4)

                # 记录到字典中，取 Batch 的平均值作为 Scalar
                # 这里的 key 必须与主循环中的 base_losses 结构一致
                monitor_stats = {
                    'std_x': samples_std_t[:, 0].mean(),
                    'std_y': samples_std_t[:, 1].mean(),
                    'std_z': samples_std_t[:, 2].mean(),
                    'std_r': samples_std_r.mean(),
                    'cost_gt_val': cost_gt.mean() 
                }
            
            # self.debug_single_batch(B, x3d_sampled[0].unsqueeze(0), x2d_sampled[0].unsqueeze(0),
            #         w2d_sampled[0].unsqueeze(0), w2d_norm[0].unsqueeze(0),
            #         gt_pose[0].unsqueeze(0), cam_K[0].unsqueeze(0),
            #         pose_opt_plus, pose_sample_logweights, cost_tgt)
            
        # scale = w2d_sampled_raw.mean(dim=-1).detach().mean()
        mc_loss = self.monte_carlo_pose_loss(pose_sample_logweights, cost_gt, scale.mean())
        
        if self.t_loss:
            obj_diagonal = torch.norm(obj_size, dim=-1).mean()
            t_loss = F.l1_loss(pose_opt_plus[:, :3] / obj_diagonal, gt_pose[:, :3] / obj_diagonal)
        if self.r_loss:
            r_loss = angular_distance_quat(pose_opt_plus[:, 3:], gt_pose[:, 3:])
        
        base_losses['mc_loss'] = mc_loss
        if self.t_loss:
            base_losses['t_loss'] = t_loss
        if self.r_loss:
            base_losses['r_loss'] = r_loss
        base_losses['monitor'] = monitor_stats
        return base_losses
    
    def hcce_loss(self, pred_front, pred_back, pred_mask, gt_front, gt_back, gt_mask):
        return super().forward(pred_front, pred_back, pred_mask, gt_front, gt_back, gt_mask)
    
    @staticmethod
    def _pose_perturb(pose: torch.Tensor, std_t=None, std_r=2.0): 
        """
        Args:
            pose: [B, 7] (x, y, z, qw, qx, qy, qz)
            std_t: std of trans, mm (std_x, std_y, std_z)
            std_r: std of rot, deg
        """
        device = pose.device
        B = pose.shape[0]
        trans = pose[:, :3]
        quat = pose[:, 3:]
        
        if std_t: 
            std_t = torch.tensor(std_t, device=device)
        else:
            std_t = torch.tensor([5.0, 5.0, 15.0], device=device)
        
        trans_perturbed = torch.randn_like(trans) * std_t + trans
        
        std_r_rad = std_r * math.pi / 180.0
        r_noise = torch.randn(B, 3, device=device) * std_r_rad
        # 增量四元数[cos(theta/2), sin(theta/2) * axis] 
        # 在小角度下的近似[1, theta_x/2, theta_y/2, theta_z/2]
        dq = torch.cat([torch.ones(B, 1, device=device), r_noise / 2.0], dim=1)
        dq = F.normalize(dq, dim=-1)
        # 四元数乘法
        dw, dx, dy, dz = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        nw = dw*qw - dx*qx - dy*qy - dz*qz
        nx = dw*qx + qw*dx + dy*qz - dz*qy
        ny = dw*qy + qw*dy + dz*qx - dx*qz
        nz = dw*qz + qw*dz + dx*qy - dy*qx
        
        quat_perturbed= torch.stack([nw, nx, ny, nz], dim=1)
        quat_perturbed= F.normalize(quat_perturbed, dim=-1)
        
        return torch.cat([trans_perturbed, quat_perturbed], dim=1)


class HccePose_PatchPnP_Loss(HccePose_Loss):
    def __init__(self, size_xyz=None, symmetric=False):
        super().__init__()
        self.size_xyz=size_xyz
        self.pm_loss = PyPMLoss(r_only=True, norm_by_extent=True, symmetric=symmetric)
        self.coord_front_loss = nn.L1Loss(reduction='none')
        self.coord_back_loss = nn.L1Loss(reduction='none')
        
    def forward(self, pred_front, pred_back, pred_mask, 
                pred_rot_6d, pred_t_site, 
                pred_coords_f, pred_coords_b,
                gt_front, gt_back, gt_mask, 
                gt_rot_mat, gt_t_site, 
                model_points, net_ref, 
                cam_K=None, Bbox=None, s_zoom=None, 
                sym_infos=None
                ):
        base_losses = super().forward(pred_front, pred_back, pred_mask, gt_front, gt_back, gt_mask)
        
        B = pred_front.shape[0]
        
        gt_f_permuted = gt_front.permute(0, 2, 3, 1)
        gt_b_permuted = gt_back.permute(0, 2, 3, 1)
        gt_coords_f = net_ref.hcce_decode(gt_f_permuted).permute(0, 3, 1, 2).detach() / 255.0
        gt_coords_b = net_ref.hcce_decode(gt_b_permuted).permute(0, 3, 1, 2).detach() / 255.0
        target_mask = gt_mask.float()
        if target_mask.dim() == 3:
            target_mask = target_mask.unsqueeze(1)
        loss_coord_f = (self.coord_front_loss(pred_coords_f, gt_coords_f) * target_mask).sum() / (target_mask.sum() * 3 + 1e-6)
        loss_coord_b = (self.coord_back_loss(pred_coords_b, gt_coords_b) * target_mask).sum() / (target_mask.sum() * 3 + 1e-6)
        
        pred_rot_mat = rot6d_to_mat_batch(pred_rot_6d)
        batch_extents = self.size_xyz.unsqueeze(0).expand(B, -1)
        # pred_trans = site_to_trans_batch(pred_t_site, cam_K, Bbox, s_zoom)
        # gt_trans = site_to_trans_batch(gt_t_site, cam_K, Bbox, s_zoom)
        loss_pm = self.pm_loss(pred_rot_mat, gt_rot_mat, model_points, extents=batch_extents, sym_infos=sym_infos)
        loss_pm_r = loss_pm['loss_PM_R']
        # loss_pm_xy = loss_pm['loss_PM_xy']
        # loss_pm_z = loss_pm['loss_PM_z']

        # pred_r_q = mat2quat_batch(rot6d_to_mat_batch(pred_rot_6d))
        # gt_r_q = mat2quat_batch(gt_rot_mat)
        # loss_r = angular_distance_quat(pred_r_q, gt_r_q)
        
        loss_center = F.l1_loss(pred_t_site[:, :2], gt_t_site[:, :2], reduction='mean')
        loss_z = F.l1_loss(pred_t_site[:, 2], gt_t_site[:, 2], reduction='mean')
        
        base_losses.update({
            'coord_front_loss': loss_coord_f,
            'coord_back_loss': loss_coord_b,
            'pm_r_loss': loss_pm_r,
            # 'pm_xy_loss': loss_pm_xy,
            # 'pm_z_loss': loss_pm_z,
            # 'r_loss': loss_r,
            'center_loss': loss_center,
            'z_loss': loss_z
        })
        
        return base_losses
    
    def base_loss(self, pred_front, pred_back, pred_mask, gt_front, gt_back, gt_mask):
        return super().forward(pred_front, pred_back, pred_mask, gt_front, gt_back, gt_mask)
    
# =====================================================================
#                              Networks
# =====================================================================


class HccePose_BF_Net(nn.Module):
    def __init__(
        self, 
        net = 'resnet', 
        input_channels = 3,
        output_channels = 49, # mask 1 + front_code 24 + back_code 24
        min_xyz = None,
        size_xyz = None,
        return_features=False
    ):
        super(HccePose_BF_Net, self).__init__()
        if net == 'resnet':
            self.net = DeepLabV3(output_channels,  efficientnet_key=False, input_channels=input_channels, return_features=return_features)
        elif net == 'efficientnet':
            self.net = DeepLabV3(output_channels,  efficientnet_key=True, input_channels=input_channels, return_features=return_features)
        elif net == 'convnext':
            self.net = ConvNeXtV2_FPN(output_channels, input_channels=input_channels, return_features=return_features)
            self.net.backbone.load_pretrained_weights('/media/ubuntu/DISK-C/YJP/HCCEPose/HccePose/models/pretrained/convnextv2_tiny_1k_224_fcmae.pt')
        else:
            assert KeyError('Wrong Net Name.')
        self.min_xyz = min_xyz
        self.size_xyz = size_xyz
        self.powers = None
        self.coord_image = None
        self.activation_function = torch.nn.Sigmoid()

    def forward(self, inputs):
        return self.net(inputs)
    
    def hcce_decode_v0(self, class_code_images_pytorch, class_base=2):
        
        class_code_images = class_code_images_pytorch.detach().cpu().numpy()
        class_id_image_2 = np.zeros((class_code_images.shape[0], class_code_images.shape[1],class_code_images.shape[2], 3))
        codes_length = int(class_code_images.shape[3]/3) 
        
        class_id_image_2[...,0] = class_id_image_2[...,0] + class_code_images[...,0] * (class_base**(codes_length - 1 - 0))
        temp2 = class_code_images[...,0]
        for i in range(codes_length-1):
            temp2 = class_code_images[...,i+1] - temp2
            temp2 = np.abs(temp2)
            class_id_image_2[...,0] = class_id_image_2[...,0] + temp2 * (class_base**(codes_length - 1 - i - 1))
        
        class_id_image_2[...,1] = class_id_image_2[...,1] + class_code_images[...,0+codes_length] * (class_base**(codes_length - 1 - 0))
        temp2 = class_code_images[...,0+codes_length]
        for i in range(codes_length - 1):
            temp2 = class_code_images[...,i+codes_length+1] - temp2
            temp2 = np.abs(temp2)
            class_id_image_2[...,1] = class_id_image_2[...,1] + temp2 * (class_base**(codes_length - 1 - i - 1))

        class_id_image_2[...,2] = class_id_image_2[...,2] + class_code_images[...,0+codes_length*2] * (class_base**(codes_length - 1 - 0))
        temp2 = class_code_images[...,0+codes_length*2]
        for i in range(codes_length-1):
            temp2 = class_code_images[...,i+codes_length*2+1] - temp2
            temp2 = np.abs(temp2)
            class_id_image_2[...,2] = class_id_image_2[...,2] + temp2 * (class_base**(codes_length - 1 - i - 1))

        class_id_image_2 = torch.from_numpy(class_id_image_2).to(class_code_images_pytorch.device)
        return class_id_image_2

    def hcce_decode(self, class_code_images):
        class_base = 2
        
        batch_size, height, width, channels = class_code_images.shape
        codes_length = channels // 3 

        class_id_image = torch.zeros_like(class_code_images[..., :3])

        if self.powers is None:
            device = class_code_images.device
            self.powers = torch.pow(
                torch.tensor(class_base, device=device, dtype=torch.float32),
                torch.arange(codes_length-1, -1, -1, device=device)
            )
        for c in range(3):
            start_idx = c * codes_length
            end_idx = start_idx + codes_length
            codes = class_code_images[..., start_idx:end_idx]
            diffs = torch.zeros_like(codes)
            diffs[..., 0] = codes[..., 0]
            for k in range(1, codes_length):
                diffs[..., k] = torch.abs(codes[..., k] - diffs[..., k-1])
            class_id_image[..., c] = torch.sum(diffs * self.powers, dim=-1)
        
        return class_id_image
    
    # @torch.inference_mode()
    def inference_batch(self, inputs, Bbox, thershold=0.5):

        pred_mask, pred_front_back_code = self.net(inputs)
        pred_mask = self.activation_function(pred_mask)
        pred_mask[pred_mask > thershold] = 1.0
        pred_mask[pred_mask <= thershold] = 0.0
        pred_mask = pred_mask[:, 0, ...]
        
        pred_front_code_raw = ((pred_front_back_code.permute(0, 2, 3, 1)+1)/2).clone().clamp(0,1)[...,:24]
        pred_back_code_raw = ((pred_front_back_code.permute(0, 2, 3, 1)+1)/2).clone().clamp(0,1)[...,24:]
        
        pred_front_back_code = self.activation_function(pred_front_back_code)
        pred_front_back_code[pred_front_back_code > thershold] = 1.0
        pred_front_back_code[pred_front_back_code <= thershold] = 0.0
        
        pred_front_back_code = pred_front_back_code.permute(0, 2, 3, 1)
        pred_front_code = pred_front_back_code[...,:24]
        pred_back_code = pred_front_back_code[...,24:]
        pred_front_code = self.hcce_decode(pred_front_code) / 255
        pred_back_code = self.hcce_decode(pred_back_code) / 255
        if self.coord_image is None:
            x = torch.arange(pred_front_code.shape[2] , device=pred_front_code.device).to(torch.float32) / pred_front_code.shape[2] 
            y = torch.arange(pred_front_code.shape[1] , device=pred_front_code.device).to(torch.float32) / pred_front_code.shape[1] 
            X, Y = torch.meshgrid(x, y, indexing='xy')  
            self.coord_image = torch.cat([X[..., None], Y[..., None]], dim=-1) 
        coord_image = self.coord_image[None,...].repeat(pred_front_code.shape[0],1,1,1)
        coord_image[..., 0] = coord_image[..., 0] * Bbox[:, None, None, 2] + Bbox[:, None, None, 0]
        coord_image[..., 1] = coord_image[..., 1] * Bbox[:, None, None, 3] + Bbox[:, None, None, 1]
        pred_front_code_0 = pred_front_code * self.size_xyz[None,None,None] + self.min_xyz[None,None,None]
        pred_back_code_0 = pred_back_code * self.size_xyz[None,None,None] + self.min_xyz[None,None,None]
        
        return {
            'pred_mask' : pred_mask,
            'coord_2d_image' : coord_image,
            'pred_front_code_obj' : pred_front_code_0,
            'pred_back_code_obj' : pred_back_code_0,
            'pred_front_code' : pred_front_code,
            'pred_back_code' : pred_back_code,
            'pred_front_code_raw' : pred_front_code_raw,
            'pred_back_code_raw' : pred_back_code_raw,
        }


class HccePose_EPro_Net(HccePose_BF_Net):
    def __init__(self, 
            net = 'resnet', 
            input_channels=3,
            output_channels=53,  # mask 1 + front_code 24 + back_code 24 + front_uncertainty 2 + back_uncertainty 2
            **kwargs):
        super(HccePose_EPro_Net, self).__init__(net, input_channels, output_channels, return_features=True, **kwargs)
        # self.scale_branch = nn.Linear(256, 2)
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

    def forward(self, inputs):
        pred_mask, pred_code, w2d, scale, _ = self.net(inputs)
        # pred_mask, res, feat = self.net(inputs)
        # pred_code, w2d = torch.split(res, [48, 4], 1)
        # scale = self.scale_branch(feat.flatten(2).mean(dim=-1)).exp()
        # scale = torch.clamp(scale, min=1e-4, max=6.0)
        return pred_mask, pred_code, w2d, scale
    
    def get_diff_coords(self, pred_code):
        # 注意这里需要确保输入是 sigmoid 后的连续值以保留梯度
        # 不要使用 .round() 或 .clamp(0,1) 的硬操作
        prob_code = torch.sigmoid(pred_code)
        mid = prob_code.shape[1] // 2
        prob_f = prob_code[:, :mid, :, :].permute(0, 2, 3, 1) # [B, H, W, 24]
        prob_b = prob_code[:, mid:, :, :].permute(0, 2, 3, 1) # [B, H, W, 24]
        coords_f = self.hcce_decode(prob_f) / 255.0 # [B, H, W, 3]
        coords_b = self.hcce_decode(prob_b) / 255.0 # [B, H, W, 3]
        x3d_f = coords_f * self.size_xyz[None,None,None] + self.min_xyz[None,None,None]
        x3d_b = coords_b * self.size_xyz[None,None,None] + self.min_xyz[None,None,None]
        return x3d_f, x3d_b
    
    def inference_batch(self, inputs, Bbox, threshold=0.5):
        pred_mask, pred_front_back_code, w2d, scale  = self(inputs)
        pred_mask = self.activation_function(pred_mask)
        pred_mask[pred_mask > threshold] = 1.0
        pred_mask[pred_mask <= threshold] = 0.0
        pred_mask = pred_mask[:, 0, ...]
        
        pred_front_code_raw = ((pred_front_back_code.permute(0, 2, 3, 1)+1)/2).clone().clamp(0,1)[...,:24]
        pred_back_code_raw = ((pred_front_back_code.permute(0, 2, 3, 1)+1)/2).clone().clamp(0,1)[...,24:]
        
        pred_front_back_code = self.activation_function(pred_front_back_code)
        pred_front_back_code[pred_front_back_code > threshold] = 1.0
        pred_front_back_code[pred_front_back_code <= threshold] = 0.0
        
        pred_front_back_code = pred_front_back_code.permute(0, 2, 3, 1)
        pred_front_code = pred_front_back_code[...,:24]
        pred_back_code = pred_front_back_code[...,24:]
        pred_front_code = self.hcce_decode(pred_front_code) / 255
        pred_back_code = self.hcce_decode(pred_back_code) / 255
        if self.coord_image is None:
            x = torch.arange(pred_front_code.shape[2] , device=pred_front_code.device).to(torch.float32) / pred_front_code.shape[2] 
            y = torch.arange(pred_front_code.shape[1] , device=pred_front_code.device).to(torch.float32) / pred_front_code.shape[1] 
            X, Y = torch.meshgrid(x, y, indexing='xy')  
            self.coord_image = torch.cat([X[..., None], Y[..., None]], dim=-1) 
        coord_image = self.coord_image[None,...].repeat(pred_front_code.shape[0],1,1,1)
        coord_image[..., 0] = coord_image[..., 0] * Bbox[:, None, None, 2] + Bbox[:, None, None, 0]
        coord_image[..., 1] = coord_image[..., 1] * Bbox[:, None, None, 3] + Bbox[:, None, None, 1]
        pred_front_code_0 = pred_front_code * self.size_xyz[None,None,None] + self.min_xyz[None,None,None]
        pred_back_code_0 = pred_back_code * self.size_xyz[None,None,None] + self.min_xyz[None,None,None]
        w2d_f = w2d[:, :2 ,: ,:]
        w2d_b = w2d[:, 2: ,: ,:]
        w2d_f = w2d_f.permute(0, 2, 3, 1)
        w2d_b = w2d_b.permute(0, 2, 3, 1)
        return {
            'pred_mask' : pred_mask,
            'coord_2d_image' : coord_image,
            'pred_front_code_obj' : pred_front_code_0,
            'pred_back_code_obj' : pred_back_code_0,
            'pred_front_code' : pred_front_code,
            'pred_back_code' : pred_back_code,
            'pred_front_code_raw' : pred_front_code_raw,
            'pred_back_code_raw' : pred_back_code_raw,
            'w2d_front': w2d_f,
            'w2d_back': w2d_b,
            'scale': scale,
        }


class HccePose_PatchPnP_Net(HccePose_BF_Net):
    def __init__(self, 
                 net='resnet', 
                 input_channels=3, 
                 feat_dim=128,
                 fb_type = "fb", # fb | f(b-f)
                 **kwargs):
        # HCCEPose 输出: 1(mask) + 24(front) + 24(back) = 49 channels
        super(HccePose_PatchPnP_Net, self).__init__(net, input_channels, output_channels=49, **kwargs)
        self.decode_net = HcceDecodeNet()
        # 输入通道: 3(front_xyz) + 3(back_xyz) = 6
        # 加入 mask 辅助，通过 mask_attention 处理
        self.pnp_net = PatchPnPNet(
            in_channels=8,  # Front + Back + Coord2d
            feat_dim=feat_dim, 
            rot_dim=6, 
            mask_attention_type="concat", # none | mul | concat
            denormalize_by_extent=True,
            # feat_layers=[3, 4]
        )
        self.fb_type = fb_type

    def forward(self, inputs, Bbox, img_sz):
        """
        inputs: [B, 3, H, W] 图像
        """
        pred_mask_logits, pred_codes_logits = self.net(inputs)
        
        prob_codes = torch.sigmoid(pred_codes_logits)
        # [B, 24, H, W]
        prob_f = prob_codes[:, :24, ...]
        prob_b = prob_codes[:, 24:, ...]

        norm_coords_f = self.decode_net(prob_f)
        norm_coords_b = self.decode_net(prob_b)
        
        # 拼接前后层坐标 [B, 6, H, W]
        if self.fb_type == "fb":
            pnp_input = torch.cat([norm_coords_f, norm_coords_b], dim=1)
        elif self.fb_type == "f(b-f)":
            pnp_input = torch.cat([norm_coords_f, norm_coords_b - norm_coords_f], dim=1)
        B, _, H, W = pnp_input.shape
        device = pnp_input.device
        # Coord2D
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        coord2d = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        coord2d[:, 0, :, :] = (coord2d[:, 0, :, :] * Bbox[:, 2, None, None] + Bbox[:, 0, None, None]) / img_sz[:, 1, None, None]
        coord2d[:, 1, :, :] = (coord2d[:, 1, :, :] * Bbox[:, 3, None, None] + Bbox[:, 1, None, None]) / img_sz[:, 0, None, None]
        # [B, 8, H, W]
        pnp_input = torch.cat([pnp_input, coord2d], dim=1)
        
        # 准备 mask attention
        pred_mask_prob = torch.sigmoid(pred_mask_logits) # [B, 1, H, W]
        
        # PatchPnPNet 的 denormalize_by_extent 逻辑需要适配 6 通道
        batch_extents = self.size_xyz.unsqueeze(0).expand(B, -1)
        pred_rot_6d, pred_t_site = self.pnp_net(
            pnp_input, 
            extents=batch_extents,
            mask_attention=pred_mask_prob.detach()
        )
        
        return pred_mask_logits, pred_codes_logits, pred_rot_6d, pred_t_site, norm_coords_f, norm_coords_b
        

    def inference_batch(self, inputs, cam_K, Bbox, img_sz, s_zoom=128.0, thershold=0.5):
        # 推理时除了调用基类的解码，还需要返回 PnP 结果
        pred_mask, pred_front_back_code, pred_rot_6d, pred_t_site, pred_front_code, pred_back_code = self.forward(inputs, Bbox, img_sz)
        pred_trans = site_to_trans_batch(pred_t_site, cam_K, Bbox, s_zoom)
        
        # 获取基础的 HCCE 解码结果（用于可视化）
        # base_res = super().inference_batch(inputs, Bbox, thershold)
        pred_mask = self.activation_function(pred_mask)
        # pred_mask[pred_mask > thershold] = 1.0
        # pred_mask[pred_mask <= thershold] = 0.0
        pred_mask = pred_mask[:, 0, ...]

        pred_front_code_0 = pred_front_code * self.size_xyz[None,:,None,None] + self.min_xyz[None,:,None,None]
        pred_back_code_0 = pred_back_code * self.size_xyz[None,:,None,None] + self.min_xyz[None,:,None,None]
        
        base_res = {
            'pred_mask' : pred_mask,
            'pred_front_code_obj' : pred_front_code_0,
            'pred_back_code_obj' : pred_back_code_0,
            'pred_front_code' : pred_front_code,
            'pred_back_code' : pred_back_code,
            'pred_rot_6d': pred_rot_6d,
            'pred_trans': pred_trans
        }
        return base_res
    
    # def soft_hcce_decode(self, probs, codes_length=8):
    #     """
    #     logits: [B, H, W, 24] (X, Y, Z 各 8 位)
        
    #     `hcce_decode` 的递归逻辑 $diff_k = |code_k - diff_{k-1}|$ 本质上是 **Gray Code 到 Binary Code** 的转换，其数学等价于前 $k$ 个位的累积异或(Cumulative XOR)。

    #     在概率域中，如果有两个独立的随机变量 $A, B \in \{0, 1\}$，其概率分别为 $p_a, p_b$，那么它们异或结果为 1 的概率为：
    #     $$P(A \oplus B = 1) = p_a(1-p_b) + p_b(1-p_a)$$

    #     利用概率论中的 Piling-up Lemma堆叠引理, 前 $k$ 个独立位累积异或为 1 的概率（即第 $k$ 个二进制位 $\hat{b}_k$）可以表示为：
    #     $$\hat{b}_k = \frac{1 - \prod_{j=0}^{k} (1 - 2p_j)}{2}$$

    #     """
    #     device = probs.device
    #     powers = 2 ** torch.arange(codes_length - 1, -1, -1).to(device).float()
        
    #     probs = self.sigmoid(probs, self.soft_xor_k, -0.5)
    #     x = 1 - 2 * probs
    #     x = x.view(*x.shape[:-1], 3, codes_length)
    #     cum_x = torch.cumprod(x, dim=-1) # 累积乘法
    #     soft_bits = (1 - cum_x) / 2.0
    #     coords = torch.sum(soft_bits * powers, dim=-1)
    #     coords_norm = coords / (2.0 ** codes_length -1)
        
    #     return coords_norm
    
    
# =====================================================================
#                      Checkpoints Functions
# =====================================================================


def save_checkpoint(path, net, iteration_step, best_score, optimizer, max_to_keep, scheduler, keypoints_ = None, w_optimizer = True, epoch=None):
    
    if not os.path.isdir(path):
        os.makedirs(path)
    saved_ckpt = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.pt')]
    # saved_ckpt = [int(i) for i in saved_ckpt]
    # saved_ckpt.sort()
    ckpt_time = [os.path.getmtime(os.path.join(path, f)) for f in saved_ckpt]
    
    num_saved_ckpt = len(saved_ckpt)
    if num_saved_ckpt >= max_to_keep:
        remove_ckpt_id = np.argmin(ckpt_time)
        os.remove(os.path.join(path, saved_ckpt[remove_ckpt_id]))

    if isinstance(net, torch.nn.parallel.DataParallel):
        state_dict = net.module.state_dict()
    elif isinstance(net, torch.nn.parallel.DistributedDataParallel):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    save_data = {
                'model_state_dict': state_dict,
                'iteration_step': iteration_step,
                'best_score': best_score,
                'scheduler_state_dict': scheduler.state_dict(), 
                }
    save_name = str(iteration_step) + '.pt'
    if w_optimizer:
        save_data.update({'optimizer_state_dict': optimizer.state_dict()})
    if keypoints_:
        save_data.update({'keypoints_': keypoints_.tolist()})
    if isinstance(epoch, int):
        save_data.update({'epoch': epoch})
        save_name = str(epoch) + '.pt'
    torch.save(save_data, os.path.join(path, save_name))
  
def get_checkpoint(path, ext='.pt'):
    # saved_ckpt = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # saved_ckpt_s = [float(i.split('step')[0].replace('_', '.')) for i in saved_ckpt]
    # saved_ckpt_id = np.argmax(saved_ckpt_s)
    # return os.path.join(path, saved_ckpt[saved_ckpt_id])
    saved_ckpt = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1] == ext]
    ckpt_time = [os.path.getmtime(f) for f in saved_ckpt]
    saved_ckpt_id = np.argmax(ckpt_time)
    return saved_ckpt[saved_ckpt_id]

def save_best_checkpoint(best_score_path, net, optimizer, best_score, iteration_step, scheduler, keypoints_ = None, w_optimizer = True, epoch=None):
    saved_ckpt = [f for f in os.listdir(best_score_path) if os.path.isfile(os.path.join(best_score_path, f)) and f.endswith('.pt')]
    if saved_ckpt != []:
        os.remove(os.path.join(best_score_path, saved_ckpt[0]))

    best_score_file_name = '{:.4f}'.format(best_score)
    best_score_file_name = best_score_file_name.replace('.', '_')
    best_score_file_name = best_score_file_name + 'step'
    best_score_file_name = best_score_file_name + str(iteration_step)
    if isinstance(net, torch.nn.parallel.DataParallel):
        state_dict = net.module.state_dict()
    elif isinstance(net, torch.nn.parallel.DistributedDataParallel):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    save_data = {
                'model_state_dict': state_dict,
                'iteration_step': iteration_step,
                'best_score': best_score,
                'scheduler_state_dict': scheduler.state_dict(), 
                }
    if w_optimizer:
        save_data.update({'optimizer_state_dict': optimizer.state_dict()})
    if keypoints_:
        save_data.update({'keypoints_': keypoints_.tolist()})
    if isinstance(epoch, int):
        save_data.update({'epoch': epoch})
        best_score_file_name = best_score_file_name + '_' + str(epoch)
    torch.save(save_data, os.path.join(best_score_path, best_score_file_name + '.pt'))

    print("best check point saved in ", os.path.join(best_score_path, best_score_file_name))

def load_checkpoint(check_point_path, net : HccePose_BF_Net, optimizer=None, local_rank=0, CUDA_DEVICE='0'):
    best_score = 0
    iteration_step = 0
    keypoints_ = []
    scheduler = None
    epoch = None
    try:
        check_point_got=get_checkpoint(check_point_path)
        print(check_point_got)
        checkpoint = torch.load(check_point_got, map_location='cuda:'+CUDA_DEVICE, weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_score = checkpoint.get('best_score', 0.0)
        iteration_step = checkpoint.get('iteration_step', 0)
        keypoints_ = checkpoint.get('keypoints_', [])
        scheduler = checkpoint.get('scheduler_state_dict', None)
        epoch = checkpoint.get('epoch', None)
    except Exception as e:
        if local_rank == 0:
            print('no checkpoint !')
            print(e)
    return {
        'best_score' : best_score,
        'iteration_step' : iteration_step,
        'keypoints_' : keypoints_,
        'scheduler_state_dict': scheduler,
        'epoch_step': epoch,
    }
