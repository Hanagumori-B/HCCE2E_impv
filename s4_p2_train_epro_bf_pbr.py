import os, torch, argparse
import math
from datetime import datetime, timedelta
import itertools
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from HccePose.bop_loader import BopDataset, TrainBopDatasetBFEPro, TestBopDatasetBFEPro
from HccePose.network_model import HccePose_EPro_Net, HccePose_EPro_Loss, load_checkpoint, save_checkpoint, save_best_checkpoint
# from torch.cuda.amp import autocast as autocast
from torch.amp import GradScaler
from torch import optim
import torch.distributed as dist
from HccePose.visualization import vis_rgb_mask_Coord
from HccePose.PnP_solver import solve_PnP, solve_PnP_comb
from HccePose.metric import add_s
from kasal.bop_toolkit_lib.inout import load_ply
from HccePose.tools.rot_reps import mat2quat_batch, quat2mat_batch, mat2quat
from epropnp.camera import PerspectiveCamera
from epropnp.cost_fun import AdaptiveHuberPnPCost
from epropnp.monte_carlo_pose_loss import MonteCarloPoseLoss
from epropnp.common import evaluate_pnp

def test_epro(obj_ply, 
              obj_info, 
              net: HccePose_EPro_Net, 
              test_loader: torch.utils.data.DataLoader, 
              local_rank, 
              world_size,
            #   num_sample=1024, 
              cuda_device=None,
              ):
    global CUDA_DEVICE
    if cuda_device: CUDA_DEVICE = cuda_device
    net.eval()
    local_add_list = []
    disable_tqdm = (local_rank != 0)
    monte_carlo_pose_loss = MonteCarloPoseLoss().to('cuda:'+CUDA_DEVICE)
    local_mc_losses = []
    for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_K_B, cam_R_m2c, cam_t_m2c) in tqdm(
        enumerate(test_loader), total=len(test_loader), desc='Validation', postfix='<' * 75, disable=disable_tqdm):
        if torch.cuda.is_available():
            rgb_c=rgb_c.to('cuda:'+CUDA_DEVICE, non_blocking=True)
            mask_vis_c=mask_vis_c.to('cuda:'+CUDA_DEVICE, non_blocking=True)
            GT_Front_hcce = GT_Front_hcce.to('cuda:'+CUDA_DEVICE, non_blocking=True)
            GT_Back_hcce = GT_Back_hcce.to('cuda:'+CUDA_DEVICE, non_blocking=True)
            Bbox = Bbox.to('cuda:'+CUDA_DEVICE, non_blocking=True)
            cam_K_cpu = cam_K
            cam_K = cam_K.to('cuda:'+CUDA_DEVICE, non_blocking=True)
        with torch.no_grad(): 
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_results = net.inference_batch(rgb_c, Bbox)
                pred_mask = pred_results['pred_mask']
                bs = pred_mask.shape[0]
                x3d_f = pred_results['pred_front_code_obj']
                x3d_b = pred_results['pred_back_code_obj']
                x2d_roi = pred_results['coord_2d_image']
                w2d_f = pred_results['w2d_front']
                w2d_b = pred_results['w2d_back']
                scale = pred_results['scale']
                
                pred_mask_np = pred_mask.detach().cpu().numpy()
                pred_f_obj_np = x3d_f.detach().cpu().numpy()
                pred_b_obj_np = x3d_b.detach().cpu().numpy()
                x2d_roi_np = x2d_roi.detach().cpu().numpy()
                
                cam_R_m2c = cam_R_m2c.to('cuda:'+CUDA_DEVICE, non_blocking=True)
                cam_t_m2c = cam_t_m2c.to('cuda:'+CUDA_DEVICE, non_blocking=True).squeeze(-1)
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
                
                # # 采样
                
                # b_size = torch.arange(B, device=x3d_f.device).view(-1, 1)
                # pred_mask_flat = pred_mask.reshape(B, -1).float()
                # eps = 1e-6
                # prob = pred_mask_flat + eps
                # prob = prob / (prob.sum(dim=1, keepdim=True) + eps)
                # idx_f = torch.multinomial(prob, num_sample, replacement=True)
                # idx_b = torch.multinomial(prob, num_sample, replacement=True)

                # x3d_f_flat = x3d_f.reshape(B, H*W, 3)
                # x2d_f_flat = x2d_roi.reshape(B, H*W, 2)
                # w2d_f_flat = w2d_f.reshape(B, H*W, 2)

                # sampled_x3d_f = x3d_f_flat[b_size, idx_f, :]
                # sampled_x2d_f = x2d_f_flat[b_size, idx_f, :]
                # sampled_w2d_f = w2d_f_flat[b_size, idx_f, :]
                
                # # 背面采样
                # x3d_b_flat = x3d_b.reshape(B, H*W, 3)
                # x2d_b_flat = x2d_roi.reshape(B, H*W, 2) # x2d 对背面是一样的坐标
                # w2d_b_flat = w2d_b.reshape(B, H*W, 2)
                

                # sampled_x3d_b = x3d_b_flat[b_size, idx_b, :]
                # sampled_x2d_b = x2d_b_flat[b_size, idx_b, :]
                # sampled_w2d_b = w2d_b_flat[b_size, idx_b, :]
                
                # x3d_sampled = torch.cat([sampled_x3d_f, sampled_x3d_b], dim=1).float() # [B, 2*HW, 3]
                # x2d_sampled = torch.cat([sampled_x2d_f, sampled_x2d_b], dim=1).float() # [B, 2*HW, 2]
                # w2d_sampled = torch.cat([sampled_w2d_f, sampled_w2d_b], dim=1).float() # [B, 2*HW, 2]
                
                # # 采样结束
                
                with torch.amp.autocast('cuda', enabled=False):
                    x3d_all, x2d_all, w2d_all = x3d_all.float(), x2d_all.float(), w2d_all.float()
                    # x3d_sampled, x2d_sampled, w2d_sampled = x3d_sampled.float(), x2d_sampled.float(), w2d_sampled.float()
                    cam_K = cam_K.float()
                    cam_K_B = cam_K_B.float()
                    wh_begin = Bbox[:, 0:2]
                    wh_unit = Bbox[:, 2] / float(W)
                    wh_unit = torch.clamp(wh_unit, min=1e-5)
                    
                    mask_flat = pred_mask.reshape(B, -1)
                    mask_all = torch.cat([mask_flat, mask_flat], dim=1).unsqueeze(-1)
                    # n_eff = mask_all.sum(dim=1, keepdim=True) # 形状 (B, 1)
                    # n_eff = torch.clamp(n_eff, min=1.0)
                    # w_mean = (w2d_all * mask_all).sum(dim=1, keepdim=True) / n_eff
                    # w2d_all = (w2d_all - w_mean - torch.log(n_eff)).exp() * scale[:, None, :]
                    w2d_all = (w2d_all - w2d_all.mean(dim=1, keepdim=True) - math.log(w2d_all.size(1))).exp() * scale[:, None, :]
                    # 确保背景点权重为 0
                    w2d_all = w2d_all * mask_all.float()
                    # w2d_all = torch.softmax(w2d_all, dim=1) * scale[:, None, :]
                    # w2d_all = w2d_all * (1.0 / wh_unit[:, None, None])
                    w2d_all = torch.nan_to_num(w2d_all, nan=0.0, posinf=1e5, neginf=0.0)
                    w2d_all = w2d_all.clamp(min=1e-12)

                    allowed_border = 30 * wh_unit[:, None]
                    # w2d_norm = (w2d_sampled - w2d_sampled.mean(dim=1, keepdim=True) - math.log(w2d_sampled.size(1))).exp() * scale[:, None, :]
                    camera = PerspectiveCamera(
                        cam_mats=cam_K, 
                        z_min=0.01, 
                        lb=wh_begin - allowed_border, 
                        ub=wh_begin + Bbox[:, 2:4] + allowed_border
                        )
                    cost_func = AdaptiveHuberPnPCost(relative_delta=0.1)
                    cost_func.set_param(x2d_all, w2d_all)
                    # cost_func.set_param(x2d_sampled, w2d_norm)
                    
                    # print(torch.where(x3d_all.isinf()), torch.where(x2d_all.isinf()), torch.where(w2d_all.isinf()))
                    # print(torch.where(x3d_all.isnan()), torch.where(x2d_all.isnan()), torch.where(w2d_all.isnan()))
                    # print(torch.where(x3d_all<0), torch.where(x2d_all<0), torch.where(w2d_all<0))
                    
                    _, cost_gt, _ = evaluate_pnp(
                        x3d_all, x2d_all, w2d_all, gt_pose,
                        camera, cost_func, out_cost=True)
                    
                    pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt = net.epropnp.monte_carlo_forward(
                        x3d_all, x2d_all, w2d_all, camera, cost_func,
                        # x3d_sampled, x2d_sampled, w2d_norm, camera, cost_func,
                        pose_init=pose_init_tensor, 
                        force_init_solve=False,
                        fast_mode=False # 测试时开启快速模式
                        )
                    local_mc_losses.append(monte_carlo_pose_loss(pose_sample_logweights, cost_gt, scale.detach().mean()).detach())
                    # local_mc_losses.append(monte_carlo_pose_loss(pose_sample_logweights, cost_tgt, scale.detach().mean()).detach())
                pred_rot = quat2mat_batch(pose_opt[:, 3:]).detach().cpu().numpy()
                pred_t = pose_opt[:, :3].detach().cpu().numpy()
                for i in range(B):
                    # 将预测的 rot 和 t 包装成原有格式
                    add_val = add_s(obj_ply, obj_info, 
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
    torch.cuda.empty_cache()
    
    local_add_list = np.concatenate(local_add_list, axis=0) 
    local_tensor = torch.from_numpy(local_add_list).to(f'cuda:{local_rank}')
    local_mc_losses = torch.stack(local_mc_losses).flatten()
    
    if world_size > 1:
        # 准备列表接收所有进程的 Tensor
        # DistributedSampler 会自动补齐数据，确保每个进程拿到的数据量一样多
        gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tensors, local_tensor)
        
        gathered_loss_tensors = [torch.zeros_like(local_mc_losses) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_loss_tensors, local_mc_losses)
        
        # 在所有进程上合并结果（这样每个进程都有完整的结果，方便后续逻辑一致）
        all_add_list = torch.cat(gathered_tensors, dim=0).cpu().numpy()
        mc_losses = torch.cat(gathered_loss_tensors, dim=0).cpu().numpy()
    else:
        all_add_list = local_add_list
        mc_losses = local_mc_losses.cpu().numpy()
        
    if np.isnan(all_add_list).any():
        print("Warning: all_add_list contains nan, fixing...")
        all_add_list = np.nan_to_num(all_add_list, nan=0.0)
        
    add_list_l = np.mean(all_add_list, axis=0)
    max_acc_id = np.argmax(add_list_l)
    max_acc = np.max(add_list_l)
    mc_loss:np.ndarray = mc_losses.mean()
    if local_rank == 0:
        # 使用 \n 确保另起一行，不受进度条残余影响
        print("="*30)
        print(f"EPRO Validation Results")
        print(f"Max Accuracy ID: {max_acc_id}")
        print(f"Max Accuracy:    {max_acc:.4f}")
        print(f'MonteCarloLoss:  {mc_loss:.4f}')
        print("="*30)
    net.train()
    return max_acc_id, float(max_acc), add_list_l, mc_loss

def test_ransac(obj_ply, obj_info, net: HccePose_EPro_Net, test_loader: torch.utils.data.DataLoader, local_rank, world_size):
    net.eval()
    local_add_list = []
    disable_tqdm = (local_rank != 0)
    for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_K_B, cam_R_m2c, cam_t_m2c) in tqdm(
        enumerate(test_loader), total=len(test_loader), desc='Validation', postfix='<' * 65, disable=disable_tqdm):
        if torch.cuda.is_available():
            rgb_c=rgb_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            mask_vis_c=mask_vis_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            GT_Front_hcce = GT_Front_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            GT_Back_hcce = GT_Back_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            Bbox = Bbox.to('cuda:'+CUDA_DEVICE, non_blocking = True)
            cam_K = cam_K.cpu().numpy()
        with torch.no_grad(): 
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_results = net.inference_batch(rgb_c, Bbox)
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
                pred_m_bf_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], pred_back_code_0_np[i], coord_image_np[i], cam_K[i]) for i in range(pred_mask_np.shape[0])]
                for (cam_R_m2c_i, cam_t_m2c_i, pred_m_bf_c_np_i) in zip(cam_R_m2c.detach().cpu().numpy(), cam_t_m2c.detach().cpu().numpy(), pred_m_bf_c_np):
                    info_list = solve_PnP_comb(pred_m_bf_c_np_i, train=True)
                    
                    for info_id_, info_i in enumerate(info_list):
                        info_list[info_id_]['add'] = add_s(obj_ply, obj_info, [[cam_R_m2c_i, cam_t_m2c_i]], [[info_i['rot'], info_i['tvecs']]])[0]
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
    torch.cuda.empty_cache()
    
    local_add_list = np.concatenate(local_add_list, axis=0) 
    local_tensor = torch.from_numpy(local_add_list).to(f'cuda:{local_rank}')
    
    if world_size > 1:
        # 准备列表接收所有进程的 Tensor
        # DistributedSampler 会自动补齐数据，确保每个进程拿到的数据量一样多
        gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tensors, local_tensor)
        
        # 在所有进程上合并结果（这样每个进程都有完整的结果，方便后续逻辑一致）
        all_add_list = torch.cat(gathered_tensors, dim=0).cpu().numpy()
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
    当 `ide_debug` 为 False 时，启用 DDP（分布式数据并行）训练。  

    DDP 训练：  
    screen -S train_ddp
    nohup python -u -m torch.distributed.launch --nproc_per_node 6 /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    python 前可通过`CUDA_VISIBLE_DEVICES=1`指定使用显卡的序号
    
    单卡训练：
    nohup python -u /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    '''
    
    # TODO: 用test的代码去跑train的数据集,检查accuracy和loss的情况; 保证训练和验证中关于EProPnP的代码相同.
    
    ide_debug = False
    
    # Specify the path to the dataset folder.
    # 指定数据集文件夹的路径。
    dataset_name = 'grab'
    dataset_path = '/media/ubuntu/WIN-E/YJP/HCCEPose/datasets/'
    dataset_path = os.path.join(dataset_path, dataset_name)
    
    # Specify the name of the subfolder in the dataset used for loading training data.
    # 指定数据集中用于加载训练数据的子文件夹名称。
    # train_folder_name = 'train_pbr'
    train_folder_name = 'train'
    val_folder_name = 'val'
    
    # The range of object IDs for training.  
    # `start_obj_id` is the starting object ID, and `end_obj_id` is the ending object ID.
    # 训练的物体 ID 范围。  
    # `start_obj_id` 为起始物体 ID，`end_obj_id` 为终止物体 ID。
    start_obj_id = 2
    end_obj_id = 2
    
    # 主干网络类型
    net_name = 'convnext'
    
    # Total number of training batches.
    # 总训练批次。
    total_iteration = 60001
    
    # Learning rate.
    # 学习率。
    lr = 5e-4
    
    # Number of samples per training epoch.
    # 每轮训练的样本数量。
    batch_size = 32
    
    # Number of worker processes used by the DataLoader.
    # DataLoader 的进程数量。
    num_workers = 16
    
    # The number of epochs between saving checkpoints.
    # 保存检查点的间隔轮数。
    log_freq = 500
 
    
    # Scaling ratio for 2D bounding boxes.
    # 2D 包围盒的缩放比例。
    padding_ratio = 1.5
    

    # Whether to enable EfficientNet.
    # 是否启用 EfficientNet。
    # efficientnet_key = None
    
    
    # Whether to enable load_breakpoint.
    # 是否启用 load_breakpoint 加载断点。
    load_breakpoint = False
    
    # 配置学习率衰减
    total_epochs = total_iteration // log_freq
    warmup_epochs = total_epochs * 5 // 100
    
    # 配置开始位姿训练
    # epro_start_epoch = total_epochs // 6
    epro_start_epoch = 0
    
    # 备份存储位置
    output_save = '/media/ubuntu/WIN-E/YJP/HCCEPose/output/'
    
    # 计算mc_loss时均匀采样，真值掩码采样，预测掩码采样的权重
    # weight_sample = [0.8, 0.1, 0.1]
    weight_sample = [0.1, 0.45, 0.45]
    
    # Loss 权重因子
    loss_factors = {
        'Front_L1Losses': 3.0,
        'Back_L1Losses': 3.0,
        'mask_loss': 1.0,
        'mc_loss': 0.02,
        # 't_loss': 3.0,
        # 'r_loss': 3.0,
    }
    
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
        output_save = os.path.join(output_save, dataset_name, 'pose_estimation', now_stamp.strftime('%Y-%m-%d_%H:%M:%S'))
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
    train_bop_dataset_back_front_item = TrainBopDatasetBFEPro(bop_dataset_item, train_folder_name, padding_ratio=padding_ratio)
    
    # ratio = 0.01 means selecting 1% of samples from the dataset for testing.
    # ratio = 0.01 表示从数据集中选择 1% 的样本作为测试数据。
    test_bop_dataset_back_front_item = TestBopDatasetBFEPro(bop_dataset_item, val_folder_name, padding_ratio=padding_ratio, ratio=0.5)
        
    for obj_id in range(start_obj_id, end_obj_id + 1):
        obj_path = bop_dataset_item.obj_model_list[bop_dataset_item.obj_id_list.index(obj_id)]
        print(obj_path)
        obj_ply = load_ply(obj_path)
        obj_info = bop_dataset_item.obj_info_list[bop_dataset_item.obj_id_list.index(obj_id)]
        
        # Create the save path.
        # 创建保存路径。
        writer = None
        save_path = os.path.join(dataset_path, 'HccePose', 'obj_%s'%str(obj_id).rjust(2, '0'))
        best_save_path = os.path.join(save_path, 'best_score')
        
        if args.local_rank == 0:
            os.makedirs(os.path.join(dataset_path, 'HccePose'), exist_ok=True)
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(best_save_path, exist_ok=True)
            
            if output_save:
                output_save_path = os.path.join(output_save, 'obj_%s'%str(obj_id).rjust(2, '0'))
                output_best_save_path = os.path.join(output_save_path, 'best_score')
                os.makedirs(output_save_path, exist_ok=True)
                os.makedirs(output_best_save_path, exist_ok=True)
                
                tb_log_dir = os.path.join(output_save_path, 'runs') 
                writer = SummaryWriter(log_dir=tb_log_dir)
                print(f"TensorBoard log dir: {tb_log_dir}")
                if not load_breakpoint:
                    writer.add_scalar('Test/ADD-S_Accuracy', 0.0, 0)

        # Get the 3D dimensions of the object.
        # 获取物体的 3D 尺寸。
        min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        
        # Define the loss function and neural network.
        # 定义损失函数和神经网络。
        loss_net = HccePose_EPro_Loss()
        scaler = GradScaler()
        net = HccePose_EPro_Net(
                net = net_name,
                input_channels = 3, 
                min_xyz = min_xyz,
                size_xyz = size_xyz,
            )
        net_test = HccePose_EPro_Net(
                net = net_name,
                input_channels = 3, 
                min_xyz = min_xyz,
                size_xyz = size_xyz,
            )
        if torch.cuda.is_available():
            net=net.to('cuda:'+CUDA_DEVICE)
            net_test=net_test.to('cuda:'+CUDA_DEVICE)
            loss_net=loss_net.to('cuda:'+CUDA_DEVICE)
        
        optimizer=optim.Adam(net.parameters(), lr=lr)   
        if net_name == 'convnext':
            optimizer=torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
            
        if epro_start_epoch <= 0:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.01, 
                end_factor=1.0, 
                total_iters=warmup_epochs 
            )
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=total_epochs - warmup_epochs, 
                eta_min=1e-6
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, 
                schedulers=[scheduler_warmup, scheduler_cosine], 
                milestones=[warmup_epochs]
            )
        else:
            epro_cos_epoch = epro_start_epoch * 2
            milestones = [warmup_epochs, epro_start_epoch, epro_cos_epoch]

            # 1. 预热段 (0.01 * base_lr -> base_lr)
            sch1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)

            # 2. 等待段 (保持 base_lr)
            sch2 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epro_start_epoch - warmup_epochs)

            sch3 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=epro_cos_epoch - epro_start_epoch)
            # 3. EPro 段 (从 base_lr * 0.1 开始骤降，然后余弦退火到 1e-6)
            # 注意：这里 factor=0.1 实现了你想要的“手动降速”
            sch4 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=total_epochs - epro_cos_epoch, 
                eta_min=1e-6
            )

            # 组合调度器
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[sch1, sch2, sch3, sch4],
                milestones=milestones
            )

        # Attempt to load weights from an interrupted training session.
        # 尝试加载中断训练时保存的权重。
        
        best_score = 0
        iteration_step = 0
        if load_breakpoint:
            try:
                checkpoint_info = load_checkpoint(save_path, net, optimizer, local_rank=local_rank, CUDA_DEVICE=CUDA_DEVICE)
                best_score = checkpoint_info['best_score']
                iteration_step = checkpoint_info['iteration_step']
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
                # raise e
        
        if not ide_debug:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], find_unused_parameters=True)
        
        # Update the training and testing data loaders respectively.
        # 分别更新训练和测试数据加载器。
        train_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
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
        
        test_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
        if not ide_debug:
            test_sampler = torch.utils.data.DistributedSampler(
                test_bop_dataset_back_front_item, 
                shuffle=False # 测试集通常不打乱
            )
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_bop_dataset_back_front_item, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            drop_last=False,
            sampler=test_sampler,
            shuffle=False
            ) 
        
        if args.local_rank == 0:
            pbar = tqdm(total=log_freq, desc=f"Training Set 1", unit="step", dynamic_ncols=True)
            
        if args.local_rank == 0 and output_save:
            with open(os.path.join(output_save_path, 'config.yaml'), 'w') as f:
                yaml.dump({
                    'ide_debug': ide_debug,
                    'dataset_path': dataset_path,
                    'net_name': net_name,
                    'total_iteration': total_iteration,
                    'batch_size': batch_size,
                    'num_workers': num_workers,
                    'log_freq': log_freq,
                    'lr': lr,
                    'padding_ratio': padding_ratio,
                    'load_breakpoint': load_breakpoint,
                    'total_epochs': total_epochs,
                    'warmup_epochs': warmup_epochs,
                    'output_save': output_save,
                    'loss_factors': loss_factors,
                    'weight_sample': weight_sample,
                    'clip_grad': clip_grad,
                }, f, sort_keys=False)

            total_params = 0
            for name, module in net.named_modules():
                params = sum(p.numel() for p in module.parameters())
                total_params += params
            with open(os.path.join(output_save_path, 'model.txt'), 'w') as f:
                print(f'total_params:{int(total_params)}', net, file=f)
                
        
        # Train
        # 训练
        while True:
            end_training = False
            if not ide_debug:
                train_loader.sampler.set_epoch(iteration_step)
            for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, bbox, cam_K, cam_K_B, cam_R_m2c, cam_t_m2c) in enumerate(train_loader):
                # Test and save checkpoints only in the process where `local_rank = 0`.
                # 仅在 `local_rank = 0` 的进程中执行测试并保存检查点。
                if iteration_step < total_iteration // 5:
                    log_freq_a = 3 * log_freq
                elif iteration_step < total_iteration // 3:
                    log_freq_a = 2 * log_freq
                else:
                    log_freq_a = log_freq
                    
                if (iteration_step % log_freq == 0 and iteration_step > 0):
                    if args.local_rank == 0:
                        pbar.close()
                    if iteration_step % log_freq_a == 0:
                        if isinstance(net, torch.nn.parallel.DataParallel):
                            state_dict = net.module.state_dict()
                        elif isinstance(net, torch.nn.parallel.DistributedDataParallel):
                            state_dict = net.module.state_dict()
                        else:
                            state_dict = net.state_dict()
                        net_test.load_state_dict(state_dict)
                        
                        if iteration_step // log_freq < epro_start_epoch + 1:
                            if args.local_rank == 0: print(f"Epoch {iteration_step // log_freq} is running RANSAC")
                            max_acc_id, max_acc, add_list_l = test_ransac(obj_ply, obj_info, net_test, test_loader, local_rank=args.local_rank, world_size=world_size if not ide_debug else 1)
                        else:
                            if args.local_rank == 0: print(f"Epoch {iteration_step // log_freq} is running EPro")
                            max_acc_id, max_acc, add_list_l, mc_loss = test_epro(
                                obj_ply, obj_info, net_test, test_loader, 
                                local_rank=args.local_rank, 
                                world_size=world_size if not ide_debug else 1, 
                                )
                            if args.local_rank == 0: writer.add_scalar('Test/Loss_Monte_Carlo', mc_loss.item(), iteration_step)
                        
                    if args.local_rank == 0:
                        # if isinstance(net, torch.nn.parallel.DataParallel):
                        #     state_dict = net.module.state_dict()
                        # elif isinstance(net, torch.nn.parallel.DistributedDataParallel):
                        #     state_dict = net.module.state_dict()
                        # else:
                        #     state_dict = net.state_dict()
                        if iteration_step % log_freq_a == 0:
                            # net_test.load_state_dict(state_dict)
                            
                            if output_save: writer.add_scalar('Test/ADD-S_Accuracy', max_acc, iteration_step)
                            
                            if max_acc >= best_score:
                                best_score = max_acc
                                save_best_checkpoint(best_save_path, net, optimizer, best_score, iteration_step, scheduler, keypoints_ = add_list_l)
                                if output_save:
                                    save_best_checkpoint(output_best_save_path, net, optimizer, best_score, iteration_step, scheduler, keypoints_ = add_list_l)
                                    
                            loss_net.print_error_ratio()
                            save_checkpoint(save_path, net, iteration_step, best_score, optimizer, 3, scheduler=scheduler, keypoints_ = add_list_l)
                            if output_save:
                                save_checkpoint(output_save_path, net, iteration_step, best_score, optimizer, 3, scheduler=scheduler, keypoints_ = add_list_l)
                        
                        current_set = iteration_step // log_freq + 1
                        pbar = tqdm(total=log_freq, desc=f"Training Set {current_set}", unit="step", dynamic_ncols=True)
                    if not ide_debug:
                        # print(f"Rank {args.local_rank} waiting for Rank 0 to finish validation...")
                        torch.distributed.barrier() 
                        # print(f"Rank {args.local_rank} resumed.")
                    
                    scheduler.step()
                
                if torch.cuda.is_available():
                    rgb_c=rgb_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    mask_vis_c=mask_vis_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    GT_Front_hcce = GT_Front_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    GT_Back_hcce = GT_Back_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    bbox = bbox.to('cuda:'+CUDA_DEVICE, non_blocking=True)
                    cam_K = cam_K.to('cuda:'+CUDA_DEVICE, non_blocking=True)
                    cam_K_B = cam_K_B.to('cuda:'+CUDA_DEVICE, non_blocking=True)
                    cam_R_m2c = cam_R_m2c.to('cuda:'+CUDA_DEVICE, non_blocking=True)
                    cam_t_m2c = cam_t_m2c.to('cuda:'+CUDA_DEVICE, non_blocking=True).squeeze(-1)
                    
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_mask, pred_code, w2d, scale = net(rgb_c)
                    # pred_front_code = pred_code[:, :24, ...]
                    # pred_back_code = pred_code[:, 24:, ...]
                    # w2d_front = w2d[:, :2, ...]
                    # w2d_back = w2d[:, 2:, ...]
                    if iteration_step // log_freq < epro_start_epoch:
                        current_loss = loss_net.hcce_loss(
                            pred_code[:, :24], pred_code[:, 24:], pred_mask, 
                            GT_Front_hcce, GT_Back_hcce, mask_vis_c, weight_sample
                        )
                        # 构造空的位姿 loss 占位，保持字典结构一致
                        current_loss['mc_loss'] = torch.tensor(0.0, device=rgb_c.device)
                        # current_loss['t_loss'] = torch.tensor(0.0, device=rgb_c.device)
                        # current_loss['r_loss'] = torch.tensor(0.0, device=rgb_c.device)
                        
                        # 设置阶段权重
                        current_factors = {
                            'Front_L1Losses': loss_factors['Front_L1Losses'],
                            'Back_L1Losses': loss_factors['Back_L1Losses'],
                            'mask_loss': loss_factors['mask_loss'],
                            'mc_loss': 0.0, 
                            # 't_loss': 0.0, 
                            # 'r_loss': 0.0  
                        }
                    
                    else:
                        if not ide_debug:
                            coords_3d = net.module.get_diff_coords(pred_code)  # 在 DistributedDataParallel 下，需通过 .module 访问自定义方法
                            epropnp = net.module.epropnp
                        else:
                            coords_3d = net.get_diff_coords(pred_code)
                            epropnp = net.epropnp
                        
                        # 复用 net 里的逻辑生成 2D grid，并根据 Bbox 还原
                        B, _, H, W = pred_code.shape
                        y, x = torch.meshgrid(torch.arange(H, device=rgb_c.device), 
                                            torch.arange(W, device=rgb_c.device), indexing='ij')
                        coords_2d = torch.stack([x, y], dim=-1).float() + 0.5# [H, W, 2]
                        coords_2d = coords_2d[None, ...].repeat(B, 1, 1, 1) # [B, H, W, 2]
                        
                        # 关键：利用 Bbox 将 [0, 127] 还原到原图像素坐标
                        bbox = bbox.to(device=coords_2d.device, dtype=torch.float32)
                        # coords_2d[..., 0] = coords_2d[..., 0] * (bbox[:, 2:3, None] / float(W)) + bbox[:, 0:1, None]
                        # coords_2d[..., 1] = coords_2d[..., 1] * (bbox[:, 3:4, None] / float(H)) + bbox[:, 1:2, None]
                        coords_2d[..., 0] = coords_2d[..., 0] * bbox[:, None, None, 2] / float(W) + bbox[:, None, None, 0]
                        coords_2d[..., 1] = coords_2d[..., 1] * bbox[:, None, None, 3] / float(H) + bbox[:, None, None, 1]
                        # 5. 准备位姿目标：将 R 矩阵转为四元数，拼接成 [B, 7]
                        gt_quat = mat2quat_batch(cam_R_m2c)
                        gt_pose = torch.cat([cam_t_m2c, gt_quat], dim=-1) # [B, 7]
                        
                        # 6. 打包 Loss 输入
                        model_out = (pred_mask, pred_code, w2d, scale)
                        targets = (mask_vis_c, GT_Front_hcce, GT_Back_hcce, gt_pose, size_xyz, min_xyz)
                        
                        current_loss = loss_net(
                            epropnp=epropnp,
                            model_out=model_out,
                            coords_3d=coords_3d,
                            coords_2d=coords_2d,
                            targets=targets,
                            cam_K=cam_K,
                            bbox=bbox,
                            out_res=H
                        )
                        
                        current_factors = loss_factors
                        # current_factors['mc_loss'] *= min((iteration_step // log_freq - epro_start_epoch) * 0.1, 1)
                        # current_factors['t_loss'] *= min((iteration_step // log_freq - epro_start_epoch) * 0.1, 1)
                        # current_factors['r_loss'] *= min((iteration_step // log_freq - epro_start_epoch) * 0.1, 1)
                    
                    l_l = [
                        current_factors['Front_L1Losses'] * torch.sum(current_loss['Front_L1Losses']),
                        current_factors['Back_L1Losses'] * torch.sum(current_loss['Back_L1Losses']),
                        current_factors['mask_loss'] * current_loss['mask_loss'],
                        current_factors['mc_loss'] * current_loss['mc_loss'],
                        # current_factors['t_loss'] * current_loss['t_loss'],
                        # current_factors['r_loss'] * current_loss['r_loss'],
                    ] 
                    loss = torch.stack(l_l).sum()
                
                if not ide_debug:
                    torch.distributed.barrier()  
                    nan_flag = torch.tensor([int(torch.isnan(loss).any())], device=loss.device)
                    dist.all_reduce(nan_flag, op=dist.ReduceOp.SUM)
                    if nan_flag.item() > 0:
                        for m in net.model.modules():
                            if isinstance(m, torch.nn.BatchNorm2d):
                                m.reset_running_stats()
                        continue
                    
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_grad)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # torch.cuda.empty_cache()
                
                if args.local_rank == 0:
                    # 记录各个分项 Loss
                    if output_save:
                        writer.add_scalar('Train/Loss_Total', loss.item(), iteration_step)
                        writer.add_scalar('Train/Loss_Front', torch.sum(current_loss['Front_L1Losses']).item(), iteration_step)
                        writer.add_scalar('Train/Loss_Back', torch.sum(current_loss['Back_L1Losses']).item(), iteration_step)
                        writer.add_scalar('Train/Loss_Mask', current_loss['mask_loss'].item(), iteration_step)
                        writer.add_scalar('Train/Loss_Monte_Carlo', current_loss['mc_loss'].item(), iteration_step)
                        # writer.add_scalar('Train/Loss_Rotation', current_loss['r_loss'].item(), iteration_step)
                        # writer.add_scalar('Train/Loss_Trans', current_loss['t_loss'].item(), iteration_step)
                        # 记录学习率
                        writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], iteration_step)
                        # 记录采样标准差
                        writer.add_scalar('Monitor/std_x', current_loss['monitor']['std_x'].item(), iteration_step)
                        writer.add_scalar('Monitor/std_y', current_loss['monitor']['std_y'].item(), iteration_step)
                        writer.add_scalar('Monitor/std_z', current_loss['monitor']['std_z'].item(), iteration_step)
                        writer.add_scalar('Monitor/std_r', current_loss['monitor']['std_r'].item(), iteration_step)
                        writer.add_scalar('Monitor/cost_gt_val', current_loss['monitor']['cost_gt_val'].item(), iteration_step)
                        
                    # 更新进度条步数
                    pbar.update(1)

                    # 实时显示Loss到进度条后缀
                    pbar.set_postfix({
                        "lB": f"{current_factors['Back_L1Losses'] * torch.sum(current_loss['Back_L1Losses']).item():.4f}",
                        "lF": f"{current_factors['Front_L1Losses'] * torch.sum(current_loss['Front_L1Losses']).item():.4f}",
                        "lM": f"{current_factors['mask_loss'] * current_loss['mask_loss'].item():.4f}",
                        "lC": f"{current_factors['mc_loss'] * current_loss['mc_loss'].item():.4f}",
                        # "lT": f"{current_factors['t_loss'] * current_loss['t_loss'].item():.4f}",
                        # "lR": f"{current_factors['r_loss'] * current_loss['r_loss'].item():.4f}",
                        "Tot": f"{loss.item():.4f}"
                    })
                
                iteration_step = iteration_step + 1
                if iteration_step >=total_iteration:
                    end_training = True
                    break
            if end_training == True:
                if args.local_rank == 0:
                    print('end the training in iteration_step:', iteration_step)
                break
