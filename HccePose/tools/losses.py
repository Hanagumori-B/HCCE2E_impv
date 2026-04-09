import torch
import torch.nn as nn
import torch.nn.functional as F
from .rot_reps import quat2mat_batch, get_closest_rot_batch


def angular_distance_quat(pred_q, gt_q):
    dist = 1 - torch.pow(torch.bmm(pred_q.view(-1, 1, 4), gt_q.view(-1, 4, 1)), 2)
    return dist.mean()


class PyPMLoss(nn.Module):
    def __init__(self, loss_type="l1", beta=1.0, reduction="mean", loss_weight=1.0, 
                 norm_by_extent=False, disentangle_t=False, disentangle_z=False, 
                 t_loss_use_points=False, symmetric=False, r_only=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.norm_by_extent = norm_by_extent
        self.symmetric = symmetric
        self.r_only = r_only
        self.disentangle_t = disentangle_t or disentangle_z
        self.disentangle_z = disentangle_z
        self.t_loss_use_points = t_loss_use_points if self.disentangle_t else True
        
        # 损失函数映射
        loss_map = {
            "smooth_l1": lambda x, y: F.smooth_l1_loss(x, y, beta=beta, reduction=reduction),
            "l1": nn.L1Loss(reduction=reduction),
            "mse": nn.MSELoss(reduction=reduction),
            "l2": nn.MSELoss(reduction=reduction) # 通常 MSE 就是 L2 的平方
        }
        self.loss_func = loss_map.get(loss_type.lower(), nn.L1Loss(reduction=reduction))

    def _compute_loss(self, pred, target, weight=1.0, multiplier=3.0):
        """通用损失计算：带权重和系数"""
        return self.loss_func(weight * pred, weight * target) * multiplier * self.loss_weight

    def forward(self, pred_rots, gt_rots, points, pred_transes=None, gt_transes=None, extents=None, sym_infos=None):
        # 1. 旋转矩阵预处理
        if gt_rots.shape[-1] == 4:
            gt_rots = quat2mat_batch(gt_rots)
        if self.symmetric:
            gt_rots = get_closest_rot_batch(pred_rots, gt_rots, sym_infos=sym_infos)

        # 2. 变换点云 (仅旋转部分)
        pts_est = transform_pts_batch(points, pred_rots, t=None)
        pts_gt = transform_pts_batch(points, gt_rots, t=None)

        # 3. 归一化系数
        weight = 1.0 / extents.max(1)[0].view(-1, 1, 1) if self.norm_by_extent else 1.0

        # 4. 核心损失逻辑
        if self.r_only:
            return {"loss_PM_R": self._compute_loss(pts_est, pts_gt, weight)}

        # 如果解耦 T，则旋转损失只取决于旋转变换后的点云
        loss_dict = {}
        loss_dict["loss_PM_R"] = self._compute_loss(pts_est, pts_gt, weight)

        if self.disentangle_z:
            if self.t_loss_use_points:
                # 使用点云计算 xy 和 z 的损失
                loss_dict["loss_PM_xy"] = self._compute_loss(pred_transes[:, :2], gt_transes[:, :2], weight=1.0) # 简化逻辑
                loss_dict["loss_PM_z"] = self._compute_loss(pred_transes[:, 2], gt_transes[:, 2], weight=1.0)
            else:
                loss_dict["loss_PM_xy_noP"] = self.loss_func(pred_transes[:, :2], gt_transes[:, :2])
                loss_dict["loss_PM_z_noP"] = self.loss_func(pred_transes[:, 2], gt_transes[:, 2])
                
        elif self.disentangle_t:
            if self.t_loss_use_points:
                loss_dict["loss_PM_T"] = self._compute_loss(pred_transes, gt_transes, weight=1.0)
            else:
                loss_dict["loss_PM_T_noP"] = self.loss_func(pred_transes, gt_transes)
        else:
            # 不解耦：(R_pred * P + T_pred) vs (R_gt * P + T_gt)
            pts_est_rt = pts_est + pred_transes.view(-1, 1, 3)
            pts_gt_rt = pts_gt + gt_transes.view(-1, 1, 3)
            return {"loss_PM_RT": self._compute_loss(pts_est_rt, pts_gt_rt, weight)}

        return loss_dict
    
def transform_pts_batch(pts, R, t=None):
    """
    Args:
        pts: (B,P,3)
        R: (B,3,3)
        t: (B,3,1)

    Returns:

    """
    bs = R.shape[0]
    n_pts = pts.shape[1]
    assert pts.shape == (bs, n_pts, 3)
    if t is not None:
        assert t.shape[0] == bs

    pts_transformed = R.view(bs, 1, 3, 3) @ pts.view(bs, n_pts, 3, 1)
    if t is not None:
        pts_transformed += t.view(bs, 1, 3, 1)
    return pts_transformed.squeeze(-1)  # (B, P, 3)
