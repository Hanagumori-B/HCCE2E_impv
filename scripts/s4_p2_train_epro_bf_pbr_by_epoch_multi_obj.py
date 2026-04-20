import os, torch, argparse
import random
import math
import shutil
from datetime import datetime, timedelta
import itertools
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from HccePose.multi_bop_loader import BopDataset, Multi_TrainBopDatasetBFEPro, Multi_TestBopDatasetBFEPro, get_obj_meta
from HccePose.multi_network_model import MultiHead_HccePose_EPro_Net, HccePose_EPro_Loss, load_checkpoint, save_checkpoint, save_best_checkpoint
from torch.amp import GradScaler
from torch import optim
import torch.distributed as dist
from HccePose.visualization import vis_rgb_mask_Coord
from HccePose.PnP_solver import solve_PnP, solve_PnP_comb
from HccePose.metric import add_s
from kasal.bop_toolkit_lib.inout import load_ply
from HccePose.tools.rot_reps import mat2quat_batch, quat2mat_batch
from epropnp.camera import PerspectiveCamera
from epropnp.cost_fun import AdaptiveHuberPnPCost
from epropnp.monte_carlo_pose_loss import MonteCarloPoseLoss
from epropnp.common import evaluate_pnp

def test_epro(obj_meta,
              net: MultiHead_HccePose_EPro_Net, 
              test_loader: torch.utils.data.DataLoader, 
              local_rank, 
              world_size,
              device=None,
              writer=None,
              ):
    net.eval()
    local_add_list = []
    disable_tqdm = (local_rank != 0)
    monte_carlo_pose_loss = MonteCarloPoseLoss().to(device)
    local_mc_losses = []
    model_ptr = net.module if hasattr(net, 'module') else net
    for batch_idx, batch in tqdm(
        enumerate(test_loader), total=len(test_loader), desc='Validation', postfix='<' * 15, disable=disable_tqdm):
        if torch.cuda.is_available():
            rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_K_B, cam_R_m2c, cam_t_m2c, obj_ids = [
                x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch
            ]
            b_size = torch.stack([obj_meta[int(i)]['size'] for i in obj_ids])
            b_min = torch.stack([obj_meta[int(i)]['min'] for i in obj_ids])

        with torch.no_grad(): 
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_results = model_ptr.inference_batch(rgb_c, Bbox, obj_ids, b_size, b_min)
                pred_mask = pred_results['pred_mask']
                bs = pred_mask.shape[0]
                x3d_f = pred_results['pred_front_code_obj']
                x3d_b = pred_results['pred_back_code_obj']
                x2d_roi = pred_results['coord_2d_image']
                w2d_f = pred_results['w2d_front']
                w2d_b = pred_results['w2d_back']
                scale = pred_results['scale']
                cam_K_cpu = cam_K.detach().cpu().numpy()
                pred_mask_np = pred_mask.detach().cpu().numpy()
                pred_f_obj_np = x3d_f.detach().cpu().numpy()
                pred_b_obj_np = x3d_b.detach().cpu().numpy()
                x2d_roi_np = x2d_roi.detach().cpu().numpy()
                
                cam_R_m2c = cam_R_m2c.to(device, non_blocking=True)
                cam_t_m2c = cam_t_m2c.to(device, non_blocking=True).squeeze(-1)
                gt_quat = mat2quat_batch(cam_R_m2c)
                gt_pose = torch.cat([cam_t_m2c, gt_quat], dim=-1).float()
    
                pose_inits = []
                for i in range(bs):
                    pred_m_bf_c_np_i = (pred_mask_np[i], pred_f_obj_np[i], pred_b_obj_np[i], x2d_roi_np[i], cam_K_cpu[i])
                    # 获取多种 PnP 组合的结果
                    info_list = solve_PnP_comb(pred_m_bf_c_np_i, train=True)
                    
                    # 挑选内点数最多（最可靠）的一个解作为初值
                    # 若全部失败，info_list 也会返回默认的 eye(3) 和 zeros(3,1)
                    best_info = max(info_list, key=lambda x: x['num'])
                    
                    # 转换旋转矩阵为四元数 [qw, qx, qy, qz]
                    # 这里使用 torch 的工具函数更方便处理 Batch
                    r_mat = torch.from_numpy(best_info['rot']).float()
                    t_vec = torch.from_numpy(best_info['tvecs']).float().reshape(3)
                    
                    # mat2quat 结果通常是 [qw, qx, qy, qz]
                    quat = mat2quat_batch(r_mat)
                    
                    pose_init = torch.cat([t_vec, quat], dim=0)
                    pose_inits.append(pose_init)
                
                # 拼接成 Batch Tensor
                pose_init_tensor = torch.stack(pose_inits).to(x3d_f.device)
                
                B, H, W, _ = x3d_f.shape
                
                x3d_all = torch.cat([x3d_f.reshape(B, -1, 3), x3d_b.reshape(B, -1, 3)], dim=1)
                x2d_all = torch.cat([x2d_roi.reshape(B, -1, 2), x2d_roi.reshape(B, -1, 2)], dim=1)
                w2d_all = torch.cat([w2d_f.reshape(B, -1, 2), w2d_b.reshape(B, -1, 2)], dim=1)
                
                with torch.amp.autocast('cuda', enabled=False):
                    x3d_all, x2d_all, w2d_all = x3d_all.float(), x2d_all.float(), w2d_all.float()

                    cam_K = cam_K.float()
                    cam_K_B = cam_K_B.float()
                    wh_begin = Bbox[:, 0:2]
                    wh_unit = Bbox[:, 2] / float(W)
                    wh_unit = torch.clamp(wh_unit, min=1e-5)
                    
                    mask_flat = pred_mask.reshape(B, -1)
                    mask_all = torch.cat([mask_flat, mask_flat], dim=1).unsqueeze(-1)

                    w2d_all = (w2d_all - w2d_all.mean(dim=1, keepdim=True) - math.log(w2d_all.size(1))).exp() * scale[:, None, :]

                    w2d_all = torch.nan_to_num(w2d_all, nan=0.0, posinf=1e5, neginf=0.0)
                    w2d_all = w2d_all * mask_all.float()
                    w2d_all = w2d_all.clamp(min=1e-12)

                    allowed_border = 30 * wh_unit[:, None]

                    camera = PerspectiveCamera(
                        cam_mats=cam_K, 
                        z_min=0.01, 
                        lb=wh_begin - allowed_border, 
                        ub=wh_begin + Bbox[:, 2:4] + allowed_border
                        )
                    cost_func = AdaptiveHuberPnPCost(relative_delta=0.1)
                    cost_func.set_param(x2d_all, w2d_all)
                    
                    _, cost_gt, _ = evaluate_pnp(
                        x3d_all, x2d_all, w2d_all, gt_pose,
                        camera, cost_func, out_cost=True)
                    
                    pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt = model_ptr.epropnp.monte_carlo_forward(
                        x3d_all, x2d_all, w2d_all, camera, cost_func,
                        pose_init=pose_init_tensor, 
                        force_init_solve=False,
                        fast_mode=False 
                        )
                    local_mc_losses.append(monte_carlo_pose_loss(pose_sample_logweights, cost_gt, scale.detach().mean()).detach())
                pred_rot = quat2mat_batch(pose_opt[:, 3:]).detach().cpu().numpy()
                pred_t = pose_opt[:, :3].detach().cpu().numpy()
                for i in range(B):
                    o_id = int(obj_ids[i]) # 获取当前样本的物体 ID
                    cur_ply = obj_meta[o_id]['ply']
                    cur_info = obj_meta[o_id]['info']
                    # 将预测的 rot 和 t 包装成原有格式
                    add_val = add_s(cur_ply, cur_info, 
                        [[cam_R_m2c[i].cpu().numpy(), cam_t_m2c[i].cpu().numpy()]], 
                        [[pred_rot[i], pred_t[i].reshape(3,1)]])[0]
                    local_add_list.append(np.array([add_val])) 
                '''
                vis_rgb_mask_Coord(rgb_c, pred_mask, pred_front_code, pred_back_code, img_path='show_vis.jpg')
                '''
            if batch_idx == 0:
                print(f"GT Translation: {cam_t_m2c[0].cpu().numpy().flatten()}")
                print(f"Pi Translation: {pose_inits[0][:3].cpu().numpy().flatten()}")
                print(f"Pr Translation: {pred_t[0].flatten()}")
                print(f"GT Rotation: {cam_R_m2c[0].cpu().numpy().flatten()}")
                print(f"Pi Rotation: {quat2mat_batch(pose_inits[0][3:]).cpu().numpy().flatten()}")
                print(f"Pr Rotation: {pred_rot[0].flatten()}")
                # 如果 GT 是 [100.5, 20.1, 500.3]，而 Pred 是 [0.1, 0.02, 0.5]，
                if local_rank == 0 and writer is not None:
                    random_idx = random.randint(0, bs - 1)
                    gt_m = mask_vis_c[random_idx].detach().cpu().unsqueeze(0)
                    pred_m = pred_mask[random_idx].detach().cpu().unsqueeze(0)
                    img_vis = rgb_c[random_idx].detach().cpu().unsqueeze(0)
                    gt_m_3c = gt_m.repeat(3, 1, 1)
                    pred_m_3c = pred_m.repeat(3, 1, 1)
                    pred_front = x3d_f[random_idx].detach().cpu().permute(2, 0, 1)
                    pred_back = x3d_b[random_idx].detach().cpu().permute(2, 0, 1)
                    gt_front_hcce = GT_Front_hcce[random_idx].unsqueeze(0).permute(0, 2, 3, 1).to(device)
                    gt_back_hcce = GT_Back_hcce[random_idx].unsqueeze(0).permute(0, 2, 3, 1).to(device)
                    with torch.no_grad():
                        gt_front_decoded = model_ptr.hcce_decode(gt_front_hcce) / 255.0
                        gt_back_decoded = model_ptr.hcce_decode(gt_back_hcce) / 255.0
                    gt_front_vis = gt_front_decoded.squeeze(0).permute(2, 0, 1).cpu()
                    gt_back_vis = gt_back_decoded.squeeze(0).permute(2, 0, 1).cpu()
                    img_vis = torch.nn.functional.interpolate(img_vis, size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
                    mask_comparison = torch.cat([img_vis, gt_m_3c, pred_m_3c], dim=2)
                    front_comparison = torch.cat([img_vis, gt_front_vis, pred_front], dim=2)
                    back_comparison = torch.cat([img_vis, gt_back_vis, pred_back], dim=2)
                    writer.add_image('Visual/Mask_Comparison', mask_comparison, global_step=0)
                    writer.add_image('Visual/Front_Comparison', front_comparison, global_step=0)
                    writer.add_image('Visual/Back_Comparison', back_comparison, global_step=0)
    torch.cuda.empty_cache()
    
    local_add_list = np.concatenate(local_add_list, axis=0) 
    local_tensor = torch.from_numpy(local_add_list).to(f'cuda:{local_rank}')
    local_mc_losses = torch.stack(local_mc_losses).flatten()
    
    if world_size > 1:
        # 先同步各卡的 Tensor 长度，防止长度不一导致 all_gather 死锁
        local_size = torch.tensor([local_tensor.shape[0]], dtype=torch.long, device=local_tensor.device)
        gathered_sizes =[torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_sizes, local_size)
        
        # 找到最大长度，将所有卡的 Tensor 用 0 补齐到相同最大长度
        max_size = max([s.item() for s in gathered_sizes])
        padded_local = torch.zeros(max_size, dtype=local_tensor.dtype, device=local_tensor.device)
        padded_local[:local_tensor.shape[0]] = local_tensor
        
        # 安全地执行 all_gather
        gathered_padded =[torch.zeros_like(padded_local) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_padded, padded_local)
        
        # 剔除补齐的 0 废数据，合并真实结果
        valid_tensors =[]
        for i, size in enumerate(gathered_sizes):
            valid_tensors.append(gathered_padded[i][:size.item()].cpu().numpy())
        all_add_list = np.concatenate(valid_tensors, axis=0)
    else:
        all_add_list = local_add_list
        # mc_losses = local_mc_losses.cpu().numpy()
        
    if np.isnan(all_add_list).any():
        print("Warning: all_add_list contains nan, fixing...")
        all_add_list = np.nan_to_num(all_add_list, nan=0.0)
        
    add_list_l = np.mean(all_add_list, axis=0)
    max_acc_id = np.argmax(add_list_l)
    max_acc = np.max(add_list_l)
    # mc_loss:np.ndarray = mc_losses.mean()
    if local_rank == 0:
        # 使用 \n 确保另起一行，不受进度条残余影响
        print("="*30)
        print(f"EPRO Validation Results")
        print(f"Max Accuracy ID: {max_acc_id}")
        print(f"Max Accuracy:    {max_acc:.4f}")
        # print(f'MonteCarloLoss:  {mc_loss:.4f}')
        print("="*30)
    net.train()
    return max_acc_id, float(max_acc), add_list_l

def test_ransac(obj_meta,
                net: MultiHead_HccePose_EPro_Net, 
                test_loader: torch.utils.data.DataLoader, 
                local_rank, 
                world_size, 
                device=None,
                writer=None,
                ):
    net.eval()
    local_add_list = []
    model_ptr = net.module if hasattr(net, 'module') else net
    disable_tqdm = (local_rank != 0)
    for batch_idx, batch in tqdm(
        enumerate(test_loader), total=len(test_loader), desc='Validation', postfix='<' * 15, disable=disable_tqdm):
        if torch.cuda.is_available():
            rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_K_B, cam_R_m2c, cam_t_m2c, obj_ids = [
                x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch
            ]
            b_size = torch.stack([obj_meta[int(i)]['size'] for i in obj_ids])
            b_min = torch.stack([obj_meta[int(i)]['min'] for i in obj_ids])

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_results = model_ptr.inference_batch(rgb_c, Bbox, obj_ids, b_size, b_min)
                pred_mask = pred_results['pred_mask']
                coord_image = pred_results['coord_2d_image']
                pred_front_code_0 = pred_results['pred_front_code_obj']
                pred_back_code_0 = pred_results['pred_back_code_obj']
                pred_front_code = pred_results['pred_front_code']
                pred_back_code = pred_results['pred_back_code']
                pred_front_code_raw = pred_results['pred_front_code_raw'].reshape((-1,128,128,3,8)).permute((0,1,2,4,3)).reshape((-1,128,128,24))
                pred_back_code_raw = pred_results['pred_back_code_raw'].reshape((-1,128,128,3,8)).permute((0,1,2,4,3)).reshape((-1,128,128,24))
                pred_front_code = torch.cat([pred_front_code, pred_front_code_raw], dim=-1)
                pred_back_code = torch.cat([pred_back_code, pred_back_code_raw], dim=-1)

                pred_mask_np = pred_mask.detach().cpu().numpy()
                pred_front_code_0_np = pred_front_code_0.detach().cpu().numpy()
                pred_back_code_0_np = pred_back_code_0.detach().cpu().numpy()
                coord_image_np = coord_image.detach().cpu().numpy()
                cam_K = cam_K.detach().cpu().numpy()
                pred_m_bf_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], pred_back_code_0_np[i], coord_image_np[i], cam_K[i]) for i in range(pred_mask_np.shape[0])]
                bs = pred_mask.shape[0]
                for i in range(bs):
                    o_id = int(obj_ids[i])
                    cur_ply = obj_meta[o_id]['ply']
                    cur_info = obj_meta[o_id]['info']
                    pred_m_bf_c_np_i = pred_m_bf_c_np[i]
                # for (cam_R_m2c_i, cam_t_m2c_i, pred_m_bf_c_np_i) in zip(cam_R_m2c.detach().cpu().numpy(), cam_t_m2c.detach().cpu().numpy(), pred_m_bf_c_np):
                    info_list = solve_PnP_comb(pred_m_bf_c_np_i, train=True)
                    for info_id_, info_i in enumerate(info_list):
                        info_list[info_id_]['add'] = add_s(
                            cur_ply, cur_info, 
                            [[cam_R_m2c.detach().cpu().numpy()[i], cam_t_m2c.detach().cpu().numpy()[i]]],
                            [[info_i['rot'], info_i['tvecs']]])[0]
                    add_list = []
                    for i_ in range(len(info_list)):
                        info_list_i = itertools.combinations(info_list, len(info_list) - i_)
                        for info_list_i_j in info_list_i:
                            best_add = 0
                            best_s = 0
                            for info_list_i_j_k in info_list_i_j:
                                if info_list_i_j_k['num'] > best_s:
                                    best_s = info_list_i_j_k['num']
                                    best_add = info_list_i_j_k['add']
                            add_list.append(best_add)
                    add_list = np.array(add_list)
                    local_add_list.append(add_list)
            if batch_idx == 0:
                print(f"GT Translation: {cam_t_m2c[0].cpu().numpy().flatten()}")
                print(f"Pi Translation: {info_list[0]['tvecs'].flatten()}")
                print(f"GT Rotation: {cam_R_m2c[0].cpu().numpy().flatten()}")
                print(f"Pi Rotation: {info_list[0]['rot'].flatten()}")
                if local_rank == 0 and writer is not None:
                    random_idx = random.randint(0, bs - 1)
                    gt_m = mask_vis_c[random_idx].detach().cpu().unsqueeze(0)
                    pred_m = pred_mask[random_idx].detach().cpu().unsqueeze(0)
                    img_vis = rgb_c[random_idx].detach().cpu().unsqueeze(0)
                    gt_m_3c = gt_m.repeat(3, 1, 1)
                    pred_m_3c = pred_m.repeat(3, 1, 1)
                    pred_front = pred_front_code_0[random_idx].detach().cpu().permute(2, 0, 1)
                    pred_back = pred_back_code_0[random_idx].detach().cpu().permute(2, 0, 1)
                    gt_front_hcce = GT_Front_hcce[random_idx].unsqueeze(0).permute(0, 2, 3, 1).to(device)
                    gt_back_hcce = GT_Back_hcce[random_idx].unsqueeze(0).permute(0, 2, 3, 1).to(device)
                    with torch.no_grad():
                        gt_front_decoded = model_ptr.hcce_decode(gt_front_hcce) / 255.0
                        gt_back_decoded = model_ptr.hcce_decode(gt_back_hcce) / 255.0
                    gt_front_vis = gt_front_decoded.squeeze(0).permute(2, 0, 1).cpu()
                    gt_back_vis = gt_back_decoded.squeeze(0).permute(2, 0, 1).cpu()
                    img_vis = torch.nn.functional.interpolate(img_vis, size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
                    mask_comparison = torch.cat([img_vis, gt_m_3c, pred_m_3c], dim=2)
                    front_comparison = torch.cat([img_vis, gt_front_vis, pred_front], dim=2)
                    back_comparison = torch.cat([img_vis, gt_back_vis, pred_back], dim=2)
                    writer.add_image('Visual/Mask_Comparison', mask_comparison, global_step=0)
                    writer.add_image('Visual/Front_Comparison', front_comparison, global_step=0)
                    writer.add_image('Visual/Back_Comparison', back_comparison, global_step=0)
    torch.cuda.empty_cache()
    
    local_add_list = np.concatenate(local_add_list, axis=0) 
    local_tensor = torch.from_numpy(local_add_list).to(f'cuda:{local_rank}')
    
    if world_size > 1:
        # 先同步各卡的 Tensor 长度，防止长度不一导致 all_gather 死锁
        local_size = torch.tensor([local_tensor.shape[0]], dtype=torch.long, device=local_tensor.device)
        gathered_sizes =[torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_sizes, local_size)
        
        # 找到最大长度，将所有卡的 Tensor 用 0 补齐到相同最大长度
        max_size = max([s.item() for s in gathered_sizes])
        padded_local = torch.zeros(max_size, dtype=local_tensor.dtype, device=local_tensor.device)
        padded_local[:local_tensor.shape[0]] = local_tensor
        
        # 安全地执行 all_gather
        gathered_padded =[torch.zeros_like(padded_local) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_padded, padded_local)
        
        # 剔除补齐的 0 废数据，合并真实结果
        valid_tensors =[]
        for i, size in enumerate(gathered_sizes):
            valid_tensors.append(gathered_padded[i][:size.item()].cpu().numpy())
        all_add_list = np.concatenate(valid_tensors, axis=0)
    else:
        all_add_list = local_add_list
        
    add_list_l = np.mean(all_add_list, axis=0)
    max_acc_id = np.argmax(add_list_l)
    max_acc = np.max(add_list_l)
    if local_rank == 0:
        print("="*30)
        print(f"RANSAC Validation Results")
        print(f"Max Accuracy ID: {max_acc_id}")
        print(f"Max Accuracy:    {max_acc:.4f}")
        print("="*30)
    net.train()
    return max_acc_id, float(max_acc), add_list_l

def set_train_phase(net, phase):
    """
    phase 1: 训练全网 (Backbone + HCCE Heads + Uncertainty Heads)
    phase 2: 冻结 Backbone 和 HCCE Heads, 只训练 EProPnP 的不确定性分支 (w2d, scale)
    """
    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
        model = net.module
    else:
        model = net
        
    if phase == 1:
        if dist.get_rank() == 0: print(">>> [PHASE 1] <<<")
        for param in model.parameters():
            param.requires_grad = True
    elif phase == 2:
        if dist.get_rank() == 0: print(">>> [PHASE 2] <<<")
        for param in model.parameters():
            param.requires_grad = False
        for o_id in model.target_obj_ids:
            id_str = str(o_id)
            for param in model.hcce_heads[id_str].parameters():
                param.requires_grad = True
            for param in model.uncertainty_heads[id_str].parameters():
                param.requires_grad = True
            for param in model.scale_branches[id_str].parameters():
                param.requires_grad = True
        
        # 3. 处理 BN 层：将不需要训练的部分设为 eval，防止 running stats 改变
        net.train() # 先设为 train
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

if __name__ == '__main__':
    '''
    When `ide_debug` is set to True, single-GPU mode is used, allowing IDE debugging.  
    When `ide_debug` is set to False, DDP (Distributed Data Parallel) training is enabled.  

    DDP Training:  
    screen -S train_ddp
    nohup python -u -m torch.distributed.launch --nproc_per_node 6 /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    
    Single-GPU Training:  
    nohup python -u /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    
    ------------------------------------------------------    
    
    当 `ide_debug` 为 True 时，仅使用单卡，可在 IDE 中进行调试。  
    当 `ide_debug` 为 False 时，启用 DDP分布式数据并行训练。  

    DDP 训练：  
    screen -S train_ddp
    nohup python -u -m torch.distributed.launch --nproc_per_node 2 /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    python 前可通过`CUDA_VISIBLE_DEVICES=1`指定使用显卡的序号
    
    单卡训练：
    nohup python -u /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    '''
    
    # TODO: 用test的代码去跑train的数据集,检查accuracy和loss的情况; 保证训练和验证中关于EProPnP的代码相同.
    
    ide_debug = False
    
    # Specify the path to the dataset folder.
    # 指定数据集文件夹的路径。
    dataset_name = 'grab'
    dataset_path = '/media/ubuntu/DISK-C/YJP/HCCEPose/datasets/'
    dataset_path = os.path.join(dataset_path, dataset_name)
    
    # Specify the name of the subfolder in the dataset used for loading training data.
    # 指定数据集中用于加载训练数据的子文件夹名称。
    # train_folder_name = 'train_pbr'
    train_folder_name = 'train'
    val_folder_name = 'val'
    
    # The range of object IDs for training.  
    # 训练的物体 ID 范围。  
    target_obj_ids = [1, 2, 3]
    
    # 主干网络类型
    net_name = 'convnext'
    
    # Total number of training batches.
    # 总训练批次。
    total_epochs = [50, 80]
    # total_epochs = 40
    
    # Learning rate.
    # 学习率。
    lr = 8e-4
    
    # Number of samples per training epoch.
    # 每轮训练的样本数量。
    batch_size = 32
    
    # Number of worker processes used by the DataLoader.
    # DataLoader 的进程数量。
    num_workers = 16
    
    # The number of epochs between saving checkpoints.
    # 保存检查点的间隔轮数。
    log_freq_per_epoch = 10
    
    # Scaling ratio for 2D bounding boxes.
    # 2D 包围盒的缩放比例。
    padding_ratio = 1.5
    

    # Whether to enable EfficientNet.
    # 是否启用 EfficientNet。
    # efficientnet_key = None
    
    
    # Whether to enable load_breakpoint.
    # 是否启用 load_breakpoint 加载断点。
    load_breakpoint = False
    manual_load_path = '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grab/pose_estimation3/2026-03-24_14:01:36'
    validate_after_load = False
    
    # 配置学习率衰减
    warmup_epochs = 3
    
    # 配置开始位姿训练 0从开始就训练， -1不使用epro
    # epro_start_epoch = total_epochs // 6
    epro_start_epoch = -1
    if isinstance(total_epochs, list):
        if len(total_epochs) == 2:
            epro_start_epoch = total_epochs[1]
        
    
    # 备份存储位置
    output_save = '/media/ubuntu/DISK-C/YJP/HCCEPose/output/'
    
    # 计算mc_loss时均匀采样，真值掩码采样，预测掩码采样的权重
    # weight_sample = [0.8, 0.1, 0.1]
    weight_sample = [0.1, 0.45, 0.45]
    
    # Loss 权重因子
    loss_factors = {
        'Front_L1Losses': 3.0,
        'Back_L1Losses': 3.0,
        'mask_loss': 1.0,
        'mc_loss': 0.02,
        't_loss': 0.2,
        'r_loss': 0.2,
    }
    
    # 训练时初始位姿扰动 [std_x, std_y, std_z, std_r]
    start_perturb_epoch = 0
    # pose_perturb_std = [3.0, 3.0, 10.0, 1.0]
    pose_perturb_std = None
    
    # 梯度裁剪
    clip_grad = 1.0
    
    parser = argparse.ArgumentParser()
    if ide_debug:
        parser.add_argument("--local-rank", default=0, type=int)
    else:
        parser.add_argument("--local-rank", default=-1, type=int)
    args = parser.parse_args()
    
    if output_save :
        now_stamp = datetime.now()
        output_save = os.path.join(output_save, dataset_name, 'pose_estimation3', now_stamp.strftime('%Y-%m-%d_%H:%M:%S'))
        if args.local_rank == 0:
            os.makedirs(output_save, exist_ok=True)
            print('-'*100)
            print(output_save)
            print('-'*100)
    
    if not ide_debug:
        torch.distributed.init_process_group(
            backend='nccl',
            timeout=timedelta(seconds=1800)
            )
        torch.distributed.barrier() 
        world_size = torch.distributed.get_world_size()
    local_rank = args.local_rank
    if local_rank != 0:
        if ide_debug is True:
            pass
    CUDA_DEVICE = str(local_rank)
    # np.random.seed(local_rank)
    
    bop_dataset_item = BopDataset(dataset_path, local_rank=local_rank)
    train_bop_dataset_back_front_item = Multi_TrainBopDatasetBFEPro(bop_dataset_item, train_folder_name, target_obj_ids, padding_ratio=padding_ratio)
    test_bop_dataset_back_front_item = Multi_TestBopDatasetBFEPro(bop_dataset_item, val_folder_name, target_obj_ids, padding_ratio=padding_ratio, ratio=0.5)
    object_meta = get_obj_meta(bop_dataset_item, target_obj_ids, device='cuda:'+CUDA_DEVICE)
    
    if not ide_debug:
        dist.barrier()
    
    # Create the save path.
    # 创建保存路径。
    writer = None
    save_path = os.path.join(dataset_path, 'HccePose', 'obj_all')
    best_save_path = os.path.join(save_path, 'best_score')
    
    if args.local_rank == 0:
        os.makedirs(os.path.join(dataset_path, 'HccePose'), exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(best_save_path, exist_ok=True)
        
        if output_save:
            output_save_path_base = os.path.join(output_save, 'obj_all')
            # output_best_save_path = os.path.join(output_save_path, 'best_score')
            os.makedirs(output_save_path_base, exist_ok=True)
            # os.makedirs(output_best_save_path, exist_ok=True)
            
            tb_log_dir = os.path.join(output_save_path_base, 'runs') 
            os.makedirs(tb_log_dir, exist_ok=True)
            
            if load_breakpoint:
                potential_old_runs = None
                base_dir = os.path.join(manual_load_path, 'obj_all')
                if os.path.exists(os.path.join(base_dir, 'runs')):
                    potential_old_runs = os.path.join(base_dir, 'runs')
                if potential_old_runs and os.path.exists(potential_old_runs):
                    print(f"Copying old TensorBoard logs from {potential_old_runs} to {tb_log_dir}...")
                    for file in os.listdir(potential_old_runs):
                        if "events.out.tfevents" in file:
                            print(f'copy {file} to {tb_log_dir}')
                            shutil.copy(os.path.join(potential_old_runs, file), tb_log_dir)
                            
            writer = SummaryWriter(log_dir=tb_log_dir)
            print(f"TensorBoard log dir: {tb_log_dir}")
            if not load_breakpoint:
                writer.add_scalar('Test/ADD-S_Accuracy', 0.0, 0)
        
    # Define the loss function and neural network.
    # 定义损失函数和神经网络。
    loss_net = HccePose_EPro_Loss(r_loss=True, t_loss=True)
    scaler = GradScaler()
    net = MultiHead_HccePose_EPro_Net(
            net = net_name,
            input_channels=3,
            target_obj_ids=target_obj_ids,
        )
    net_test = MultiHead_HccePose_EPro_Net(
            net = net_name,
            input_channels=3, 
            target_obj_ids=target_obj_ids,
        )
    if torch.cuda.is_available():
        net=net.to('cuda:'+CUDA_DEVICE)
        net_test=net_test.to('cuda:'+CUDA_DEVICE)
        loss_net=loss_net.to('cuda:'+CUDA_DEVICE)
        
    optimizer=optim.Adam(net.parameters(), lr=lr)   
    if net_name == 'convnext':
        optimizer=torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    
    # Update the training and testing data loaders respectively.
    # 分别更新训练和测试数据加载器。
    if not ide_debug:
        train_sampler = torch.utils.data.DistributedSampler(
            train_bop_dataset_back_front_item, 
            shuffle=True # 训练集需要打乱
            )
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_bop_dataset_back_front_item, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        drop_last=True,
        sampler=train_sampler,
        shuffle=(train_sampler is None)
            )
    
    
    if not ide_debug:
        test_sampler = torch.utils.data.DistributedSampler(
            test_bop_dataset_back_front_item, 
            shuffle=True
        )
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_bop_dataset_back_front_item, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        drop_last=False,
        sampler=test_sampler,
        shuffle=(test_sampler is None)
        ) 
    
    # scheduler
    iter_per_epoch = len(train_loader)
    if epro_start_epoch <= 0:
        if isinstance(total_epochs, list):
            total_epoch = total_epochs[-1]
        else:
            total_epoch = total_epochs
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.01, 
            end_factor=1.0, 
            total_iters=warmup_epochs * iter_per_epoch
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=(total_epoch - warmup_epochs) * iter_per_epoch, 
            eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[scheduler_warmup, scheduler_cosine], 
            milestones=[warmup_epochs * iter_per_epoch]
        )
    else:
        phase1_epochs = total_epochs[0]
        phase2_epochs = total_epochs[1]

        # 第一阶段：Warmup(3) + Cosine
        sch1 = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs * iter_per_epoch)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=(phase1_epochs - warmup_epochs) * iter_per_epoch,
            eta_min=1e-6)

        # 第二阶段：Warmup(3) + Cosine
        sch3 = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.01, 
            end_factor=1.0,
            total_iters=warmup_epochs * iter_per_epoch)
        sch4 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=(phase2_epochs - warmup_epochs) * iter_per_epoch,
            eta_min=1e-6)

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[sch1, sch2, sch3, sch4],
            milestones=[
                warmup_epochs * iter_per_epoch, 
                phase1_epochs * iter_per_epoch, 
                (phase1_epochs + warmup_epochs) * iter_per_epoch
            ]
        )

    # Attempt to load weights from an interrupted training session.
    # 尝试加载中断训练时保存的权重。
    
    best_score = 0
    iteration_step = 0
    start_epoch = 0
    
    if load_breakpoint:
        try:
            load_path = os.path.join(manual_load_path, 'obj_all') if manual_load_path else save_path
            if os.path.exists(os.path.join(load_path, 'phase2')):
                load_path = os.path.join(load_path, 'phase2')
            elif os.path.exists(os.path.join(load_path, 'phase1')):
                load_path = os.path.join(load_path, 'phase1')
            checkpoint_info = load_checkpoint(load_path, net, optimizer, local_rank=local_rank, CUDA_DEVICE=CUDA_DEVICE)
            best_score = checkpoint_info['best_score']
            iteration_step = checkpoint_info['iteration_step']
            start_epoch = checkpoint_info.get('epoch_step', iteration_step // len(train_loader))
            if start_epoch is None:
                start_epoch = iteration_step // len(train_loader)
            if 'scheduler_state_dict' in checkpoint_info and checkpoint_info['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint_info['scheduler_state_dict'])
                print(f"Successfully resumed scheduler from step {iteration_step}")
            else:
                # 备选方案：如果之前的模型没存 scheduler 状态，手动快进
                print("Scheduler state not found, fast-forwarding...")
                for _ in range(iteration_step - 1):
                    scheduler.step()
                print(f"Successfully resumed scheduler from step {iteration_step}")
        except Exception as e:
            print('no checkpoint', e)
            raise e
    
    if not ide_debug:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], find_unused_parameters=True)
        dist.barrier() 

        
        # if args.local_rank == 0:
        #     pbar = tqdm(total=1, desc=f"Training", unit="step", dynamic_ncols=True)
            
    if args.local_rank == 0 and output_save:
        with open(os.path.join(output_save_path_base, 'config.yaml'), 'w') as f:
            yaml.dump({
                'ide_debug': ide_debug,
                'dataset_path': dataset_path,
                'target_obj_ids': target_obj_ids,
                'net_name': net_name,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'lr': lr,
                'padding_ratio': padding_ratio,
                'load_breakpoint': load_breakpoint,
                'total_epochs': total_epochs,
                'warmup_epochs': warmup_epochs,
                'output_save': output_save,
                'loss_factors': loss_factors,
                'weight_sample': weight_sample,
                'pose_perturb_std': pose_perturb_std,
                'start_perturb_epoch': start_perturb_epoch,
                'clip_grad': clip_grad,
            }, f, sort_keys=False)

        total_params = 0
        for name, module in net.named_modules():
            params = sum(p.numel() for p in module.parameters())
            total_params += params
        with open(os.path.join(output_save_path_base, 'model.txt'), 'w') as f:
            print(f'total_params:{int(total_params)}', net, file=f)
                
    
    if load_breakpoint and validate_after_load:
        if start_epoch < epro_start_epoch or epro_start_epoch < 0:
            if args.local_rank == 0: 
                print(f"Epoch {start_epoch} is running RANSAC")
            max_acc_id, max_acc, add_list_l = test_ransac(
                object_meta, net_test, test_loader, 
                local_rank=args.local_rank, 
                world_size=world_size if not ide_debug else 1, 
                device='cuda:'+CUDA_DEVICE,
                writer=writer,
                )
        else:
            if args.local_rank == 0: 
                print(f"Epoch {start_epoch} is running EPro")
            max_acc_id, max_acc, add_list_l = test_epro(
                object_meta, net_test, test_loader, 
                local_rank=args.local_rank, 
                world_size=world_size if not ide_debug else 1, 
                device='cuda:'+CUDA_DEVICE,
                writer=writer,
                )
        
    # Train
    # 训练
    for epoch in range(start_epoch, total_epochs[-1]):
        num_batches = len(train_loader)
        log_interval = max(num_batches // log_freq_per_epoch, 1) # 每隔 10% 的进度记录一次
        logs_done_this_epoch = 0
        if epoch == 0 or (load_breakpoint and 0 < epoch < epro_start_epoch):
            set_train_phase(net, phase=1)
            phase_dir = "phase1"
            current_phase_best_score = 0.0
            if args.local_rank == 0 and output_save:
                output_save_path = os.path.join(output_save_path_base, phase_dir)
                output_best_save_path = os.path.join(output_save_path, 'best_score')
                os.makedirs(output_save_path, exist_ok=True)
                os.makedirs(output_best_save_path, exist_ok=True)
        elif epoch == epro_start_epoch or (load_breakpoint and epoch >= epro_start_epoch):
            set_train_phase(net, phase=2)
            phase_dir = "phase2"
            current_phase_best_score = 0.0
            if args.local_rank == 0 and output_save:
                output_save_path = os.path.join(output_save_path_base, phase_dir)
                output_best_save_path = os.path.join(output_save_path, 'best_score')
                os.makedirs(output_save_path, exist_ok=True)
                os.makedirs(output_best_save_path, exist_ok=True)
            
        if not ide_debug:
            # 这一行非常重要：保证 DDP 模式下每个 epoch 的数据打乱顺序不同
            train_loader.sampler.set_epoch(epoch)
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)
            
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            if torch.cuda.is_available():
                rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_K_B, cam_R_m2c, cam_t_m2c, obj_ids = [
                x.to('cuda:'+CUDA_DEVICE, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch
            ]
                b_size_xyz = torch.stack([object_meta[int(i)]['size'] for i in obj_ids])
                b_min_xyz = torch.stack([object_meta[int(i)]['min'] for i in obj_ids])
                # obj_info = torch.stack([object_meta[int(i)]['info'] for i in obj_ids])
                # obj_ply = torch.stack([object_meta[int(i)]['ply'] for i in obj_ids])
                
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_mask, pred_code, w2d, scale = net(rgb_c, obj_ids)
                if epoch < epro_start_epoch:
                    current_loss = loss_net.hcce_loss(
                        pred_code[:, :24], pred_code[:, 24:], pred_mask, 
                        GT_Front_hcce, GT_Back_hcce, mask_vis_c
                    )
                    # 构造空的位姿 loss 占位
                    current_loss['mc_loss'] = torch.tensor(0.0, device=rgb_c.device)
                    current_loss['t_loss'] = torch.tensor(0.0, device=rgb_c.device)
                    current_loss['r_loss'] = torch.tensor(0.0, device=rgb_c.device)

                    l_l = [
                        loss_factors['Front_L1Losses'] * torch.sum(current_loss['Front_L1Losses']),
                        loss_factors['Back_L1Losses'] * torch.sum(current_loss['Back_L1Losses']),
                        loss_factors['mask_loss'] * current_loss['mask_loss'],
                        # loss_factors['mc_loss'] * current_loss['mc_loss'],
                        # loss_factors['t_loss'] * current_loss['t_loss'],
                        # loss_factors['r_loss'] * current_loss['r_loss'],
                    ]
                
                else:
                    if not ide_debug:
                        coords_3d = net.module.get_diff_coords(pred_code, b_size_xyz, b_min_xyz)  # 在 DistributedDataParallel 下，需通过 .module 访问自定义方法
                        epropnp = net.module.epropnp
                    else:
                        coords_3d = net.get_diff_coords(pred_code, b_size_xyz, b_min_xyz)
                        epropnp = net.epropnp
                    
                    # 复用 net 里的逻辑生成 2D grid，并根据 Bbox 还原
                    B, _, H, W = pred_code.shape
                    y, x = torch.meshgrid(torch.arange(H, device=rgb_c.device), 
                                        torch.arange(W, device=rgb_c.device), indexing='ij')
                    coords_2d = torch.stack([x, y], dim=-1).float() + 0.5# [H, W, 2]
                    coords_2d = coords_2d[None, ...].repeat(B, 1, 1, 1) # [B, H, W, 2]
                    
                    # 关键：利用 Bbox 将 [0, 127] 还原到原图像素坐标
                    bbox = bbox.to(device=coords_2d.device, dtype=torch.float32)
                    coords_2d[..., 0] = coords_2d[..., 0] * bbox[:, None, None, 2] / float(W) + bbox[:, None, None, 0]
                    coords_2d[..., 1] = coords_2d[..., 1] * bbox[:, None, None, 3] / float(H) + bbox[:, None, None, 1]
                    # 5. 准备位姿目标：将 R 矩阵转为四元数，拼接成 [B, 7]
                    gt_quat = mat2quat_batch(cam_R_m2c)
                    gt_pose = torch.cat([cam_t_m2c, gt_quat], dim=-1) # [B, 7]
                    
                    # 6. 打包 Loss 输入
                    model_out = (pred_mask, pred_code, w2d, scale)
                    targets = (mask_vis_c, GT_Front_hcce, GT_Back_hcce, gt_pose, b_size_xyz, b_min_xyz)
                    
                    if epoch < start_perturb_epoch:
                        pose_perturb_std_ = None
                    else:
                        pose_perturb_std_ = pose_perturb_std
                    
                    current_loss = loss_net(
                        epropnp=epropnp,
                        model_out=model_out,
                        coords_3d=coords_3d,
                        coords_2d=coords_2d,
                        targets=targets,
                        cam_K=cam_K,
                        bbox=bbox,
                        out_res=H,
                        weight_sample=weight_sample,
                        # pose_perturb_std=pose_perturb_std_,
                    )
                    
                    # 构造空的位姿 loss 占位
                    current_loss['Front_L1Losses'] = torch.zeros_like(current_loss['Front_L1Losses'])
                    current_loss['Back_L1Losses'] = torch.zeros_like(current_loss['Back_L1Losses'])
                    current_loss['mask_loss'] = torch.tensor(0.0, device=rgb_c.device)
                
                    l_l = [
                        # loss_factors['Front_L1Losses'] * torch.sum(current_loss['Front_L1Losses']),
                        # loss_factors['Back_L1Losses'] * torch.sum(current_loss['Back_L1Losses']),
                        # loss_factors['mask_loss'] * current_loss['mask_loss'],
                        loss_factors['mc_loss'] * current_loss['mc_loss'],
                        loss_factors['t_loss'] * current_loss['t_loss'],
                        loss_factors['r_loss'] * current_loss['r_loss'],
                    ] 
                loss = torch.stack(l_l).sum()
            
            if not ide_debug:
                torch.distributed.barrier()  
                is_abnormal = torch.isnan(loss).any() or torch.isinf(loss).any() or (loss > 20.0)
                nan_flag = torch.tensor([int(is_abnormal)], device=loss.device)
                dist.all_reduce(nan_flag, op=dist.ReduceOp.SUM)
                if nan_flag.item() > 0:
                    for m in net.model.modules():
                        if isinstance(m, torch.nn.BatchNorm2d):
                            m.reset_running_stats()
                    continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # torch.cuda.empty_cache()
            
            if args.local_rank == 0:
                # 记录各个分项 Loss
                if batch_idx % log_interval == 0 and (logs_done_this_epoch < log_freq_per_epoch - 1) or batch_idx == num_batches - 1:
                    if output_save:
                        record_epoch = int((epoch + logs_done_this_epoch * (1.0 / log_freq_per_epoch)) * log_freq_per_epoch)
                        writer.add_scalar('Train/Loss_Total', loss.item(), record_epoch)
                        writer.add_scalar('Train/Loss_Front', torch.sum(current_loss['Front_L1Losses']).item(), record_epoch)
                        writer.add_scalar('Train/Loss_Back', torch.sum(current_loss['Back_L1Losses']).item(), record_epoch)
                        writer.add_scalar('Train/Loss_Mask', current_loss['mask_loss'].item(), record_epoch)
                        writer.add_scalar('Train/Loss_Monte_Carlo', current_loss['mc_loss'].item(), record_epoch)
                        writer.add_scalar('Train/Loss_Rotation', current_loss['r_loss'].item(), record_epoch)
                        writer.add_scalar('Train/Loss_Trans', current_loss['t_loss'].item(), record_epoch)
                        # 记录学习率
                        writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], record_epoch)
                        # 记录采样标准差
                        # writer.add_scalar('Monitor/std_x', current_loss['monitor']['std_x'].item(), record_epoch)
                        # writer.add_scalar('Monitor/std_y', current_loss['monitor']['std_y'].item(), record_epoch)
                        # writer.add_scalar('Monitor/std_z', current_loss['monitor']['std_z'].item(), record_epoch)
                        # writer.add_scalar('Monitor/std_r', current_loss['monitor']['std_r'].item(), record_epoch)
                        # writer.add_scalar('Monitor/cost_gt_val', current_loss['monitor']['cost_gt_val'].item(), record_epoch)
                        logs_done_this_epoch += 1
                    
                # 更新进度条步数
                pbar.update(1)

                # 实时显示Loss到进度条后缀
                pbar.set_postfix({
                    # "lB": f"{current_factors['Back_L1Losses'] * torch.sum(current_loss['Back_L1Losses']).item():.4f}",
                    # "lF": f"{current_factors['Front_L1Losses'] * torch.sum(current_loss['Front_L1Losses']).item():.4f}",
                    # "lM": f"{current_factors['mask_loss'] * current_loss['mask_loss'].item():.4f}",
                    # "lC": f"{current_factors['mc_loss'] * current_loss['mc_loss'].item():.4f}",
                    # "lT": f"{current_factors['t_loss'] * current_loss['t_loss'].item():.4f}",
                    # "lR": f"{current_factors['r_loss'] * current_loss['r_loss'].item():.4f}",
                    "Tot": f"{loss.item():.4f}"
                })
            
            iteration_step = iteration_step + 1
            
        # 验证
        if isinstance(net, torch.nn.parallel.DataParallel):
            state_dict = net.module.state_dict()
        elif isinstance(net, torch.nn.parallel.DistributedDataParallel):
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()
        
        net_test.load_state_dict(state_dict)
        if args.local_rank == 0: 
            pbar.close()
        if epoch < epro_start_epoch or epro_start_epoch < 0:
            if args.local_rank == 0: 
                print(f"Epoch {epoch} is running RANSAC")
            max_acc_id, max_acc, add_list_l = test_ransac(
                object_meta, net_test, test_loader, 
                local_rank=args.local_rank, 
                world_size=world_size if not ide_debug else 1, 
                device='cuda:'+CUDA_DEVICE,
                writer=writer,
                )
            if args.local_rank == 0 and output_save: 
                writer.add_scalar('Test/ADD-S_Accuracy', max_acc, epoch * log_freq_per_epoch + log_freq_per_epoch - 1)

        else:
            if args.local_rank == 0: 
                print(f"Epoch {epoch} is running EPro")
            max_acc_id, max_acc, add_list_l = test_epro(
                object_meta, net_test, test_loader, 
                local_rank=args.local_rank, 
                world_size=world_size if not ide_debug else 1, 
                device='cuda:'+CUDA_DEVICE,
                writer=writer,
                )
            if args.local_rank == 0 and output_save: 
                writer.add_scalar('Test/ADD-S_Accuracy', max_acc, epoch * log_freq_per_epoch + log_freq_per_epoch - 1)
                # writer.add_scalar('Test/Loss_Monte_Carlo', mc_loss.item(), epoch * log_freq_per_epoch + log_freq_per_epoch - 1)
            
        if args.local_rank == 0:
            if max_acc >= best_score:
                best_score = max_acc
                current_phase_best_score = max_acc
                save_best_checkpoint(best_save_path, net, optimizer, best_score, iteration_step, scheduler, keypoints_=add_list_l, epoch=epoch)
                if output_save:
                    save_best_checkpoint(output_best_save_path, net, optimizer, current_phase_best_score, iteration_step, scheduler, keypoints_=add_list_l, epoch=epoch)
                    
            loss_net.print_error_ratio()
            save_checkpoint(save_path, net, iteration_step, best_score, optimizer, 3, scheduler=scheduler, keypoints_=add_list_l, epoch=epoch)
            if output_save: 
                save_checkpoint(output_save_path, net, iteration_step, current_phase_best_score, optimizer, 3, scheduler=scheduler, keypoints_=add_list_l, epoch=epoch)
        
    if args.local_rank == 0:
        print('end the training in iteration_step:', iteration_step)

