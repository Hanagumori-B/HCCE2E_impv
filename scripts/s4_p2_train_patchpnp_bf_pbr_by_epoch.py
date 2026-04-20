import os
import torch
import argparse
import math
import time
import shutil
import random
import copy
from datetime import datetime, timedelta
import itertools
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from HccePose.bop_loader import BopDataset, TrainBopDatasetBF_PnPNet, TestBopDatasetBF_PnPNet
from HccePose.network_model import HccePose_PatchPnP_Net, HccePose_PatchPnP_Loss, load_checkpoint, save_checkpoint, save_best_checkpoint
# from torch.cuda.amp import autocast as autocast
from torch.amp import GradScaler
from torch import optim
import torch.distributed as dist
from HccePose.visualization import vis_rgb_mask_Coord
from HccePose.metric import add_s, aad_mm
from kasal.bop_toolkit_lib.inout import load_ply
from HccePose.tools.rot_reps import mat2quat_batch, quat2mat_batch, mat2quat, rot6d_to_mat_batch
from HccePose.tools.t_site_tools import trans_to_site_batch, site_to_trans_batch


def test(obj_ply, 
         obj_info, 
         net: HccePose_PatchPnP_Net,
         test_loader: torch.utils.data.DataLoader,
         local_rank, 
         world_size,
         device=None,
         writer=None,
         epoch=0,
         ):
    net.eval()
    local_add_list = []
    local_aad_list = []
    
    local_err_xy = []  # XY 平移误差 (mm)
    local_err_z = []   # Z 平移误差 (mm)
    local_err_rot = [] # 旋转误差 (deg)
    
    local_batch_max_xy = []
    local_batch_max_z = []
    local_batch_max_rot = []

    disable_tqdm = (local_rank != 0)
    for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, img_size, cam_R_m2c, cam_t_m2c) in tqdm(
        enumerate(test_loader), total=len(test_loader), desc='Validation', postfix='<' * 10, disable=disable_tqdm):
        if torch.cuda.is_available():
            rgb_c=rgb_c.to(device, non_blocking=True)
            mask_vis_c=mask_vis_c.to(device, non_blocking=True)
            GT_Front_hcce = GT_Front_hcce.to(device, non_blocking=True)
            GT_Back_hcce = GT_Back_hcce.to(device, non_blocking=True)
            Bbox = Bbox.to(device, non_blocking=True)
            cam_K = cam_K.to(device, non_blocking=True)
            img_size = img_size.to(device, non_blocking=True)
        with torch.no_grad(): 
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_results = net.inference_batch(rgb_c, cam_K, Bbox, img_size)
                pred_rot = rot6d_to_mat_batch(pred_results['pred_rot_6d'])
                pred_t = pred_results['pred_trans']
                
                pred_rots_np = pred_rot.detach().cpu().numpy()
                pred_trans_np = pred_t.detach().cpu().numpy()
                gt_rots_np = cam_R_m2c.detach().cpu().numpy()
                gt_ts_np = cam_t_m2c.detach().cpu().numpy()
                if gt_ts_np.ndim == 3: # 处理 [B, 3, 1] 变成 [B, 3]
                    gt_ts_np = gt_ts_np.squeeze(-1)
                    
                curr_batch_err_xy = []
                curr_batch_err_z = []
                curr_batch_err_rot = []

                # 2. 遍历 Batch 计算精度
                for i in range(pred_rots_np.shape[0]):
                    # PatchPnP 是端到端的，通常每个物体只输出一个确定位姿
                    # 直接计算该位姿的 ADD-S
                    add_val = add_s(
                        obj_ply, 
                        obj_info, 
                        [[gt_rots_np[i], gt_ts_np[i]]], 
                        [[pred_rots_np[i], pred_trans_np[i]]]
                    )[0]
                    aad_val = aad_mm(
                        obj_ply, 
                        obj_info, 
                        [[gt_rots_np[i], gt_ts_np[i]]], 
                        [[pred_rots_np[i], pred_trans_np[i]]]
                    )[0]
                    
                    local_add_list.append(np.array([add_val]))
                    local_aad_list.append(np.array([aad_val]))
                    
                    #  XY 平移误差 (欧式距离)
                    err_xy = np.linalg.norm(pred_trans_np[i, :2] - gt_ts_np[i, :2])
                    #  Z 平移误差 (绝对差值)
                    err_z = np.abs(pred_trans_np[i, 2] - gt_ts_np[i, 2])
                    #  旋转误差 (角度距离) arccos((Trace(R_pred * R_gt^T) - 1) / 2)
                    rel_rot = pred_rots_np[i] @ gt_rots_np[i].T
                    cos_theta = (np.trace(rel_rot) - 1.0) / 2.0
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    err_rot = np.rad2deg(np.arccos(cos_theta))

                    curr_batch_err_xy.append(err_xy)
                    curr_batch_err_z.append(err_z)
                    curr_batch_err_rot.append(err_rot)
                    
                local_err_xy.extend(curr_batch_err_xy)
                local_err_z.extend(curr_batch_err_z)
                local_err_rot.extend(curr_batch_err_rot)
                
                local_batch_max_xy.append(np.max(curr_batch_err_xy))
                local_batch_max_z.append(np.max(curr_batch_err_z))
                local_batch_max_rot.append(np.max(curr_batch_err_rot))
                
        if batch_idx == 0:
            print(f"GT Translation: {cam_t_m2c[0].cpu().numpy().flatten()}")
            print(f"Pr Translation: {pred_t[0].cpu().numpy().flatten()}")
            print(f"GT Rotation: {cam_R_m2c[0].cpu().numpy().flatten()}")
            print(f"Pr Rotation: {pred_rot[0].cpu().numpy().flatten()}")
            pred_front_code = pred_results['pred_front_code']
            pred_back_code = pred_results['pred_back_code']
            pred_mask = pred_results['pred_mask']
            if local_rank == 0 and writer is not None:
                bs = pred_mask.shape[0]
                random_idx = random.randint(0, bs - 1)
                gt_m = mask_vis_c[random_idx].detach().cpu().unsqueeze(0)
                pred_m = pred_mask[random_idx].detach().cpu().unsqueeze(0)
                img_vis = rgb_c[random_idx].detach().cpu().unsqueeze(0)
                gt_m_3c = gt_m.repeat(3, 1, 1)
                pred_m_3c = pred_m.repeat(3, 1, 1)
                pred_front = pred_front_code[random_idx].detach().cpu()
                pred_back = pred_back_code[random_idx].detach().cpu()
                gt_front_hcce = GT_Front_hcce[random_idx].unsqueeze(0).permute(0, 2, 3, 1).to(device)
                gt_back_hcce = GT_Back_hcce[random_idx].unsqueeze(0).permute(0, 2, 3, 1).to(device)
                with torch.no_grad():
                    gt_front_decoded = net.hcce_decode(gt_front_hcce) / 255.0
                    gt_back_decoded = net.hcce_decode(gt_back_hcce) / 255.0
                gt_front_vis = gt_front_decoded.squeeze(0).permute(2, 0, 1).cpu()
                gt_back_vis = gt_back_decoded.squeeze(0).permute(2, 0, 1).cpu()
                img_vis = torch.nn.functional.interpolate(img_vis, size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
                mask_comparison = torch.cat([img_vis, gt_m_3c, pred_m_3c], dim=2)
                front_comparison = torch.cat([img_vis, gt_front_vis, pred_front], dim=2)
                back_comparison = torch.cat([img_vis, gt_back_vis, pred_back], dim=2)
                writer.add_image('Visual/Mask_Comparison', mask_comparison, global_step=epoch)
                writer.add_image('Visual/Front_Comparison', front_comparison, global_step=epoch)
                writer.add_image('Visual/Back_Comparison', back_comparison, global_step=epoch)
                # print("GT F Min:", gt_front_vis.min().item())
                # print("GT F Max:", gt_front_vis.max().item())
                # print("Pred F Min:", pred_front.min().item())
                # print("Pred F Max:", pred_front.max().item())
    local_add_list = np.concatenate(local_add_list, axis=0) 
    local_tensor = torch.from_numpy(local_add_list).to(f'cuda:{local_rank}')
    local_aad_list = np.concatenate(local_aad_list, axis=0) 
    local_aad_tensor = torch.from_numpy(local_aad_list).to(f'cuda:{local_rank}')
    
    def collect_results(local_data):
        data_tensor = torch.tensor(local_data, device=device)
        if world_size > 1:
            gathered = [torch.zeros_like(data_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, data_tensor)
            return torch.cat(gathered, dim=0).cpu().numpy()
        return data_tensor.cpu().numpy()
    
    # if world_size > 1:
    #     # 准备列表接收所有进程的 Tensor
    #     # DistributedSampler 会自动补齐数据，确保每个进程拿到的数据量一样多
    #     gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    #     torch.distributed.all_gather(gathered_tensors, local_tensor)
    #     gathered_aad_tensors = [torch.zeros_like(local_aad_tensor) for _ in range(world_size)]
    #     torch.distributed.all_gather(gathered_aad_tensors, local_aad_tensor)
        
    #     # 在所有进程上合并结果（这样每个进程都有完整的结果，方便后续逻辑一致）
    #     all_add_list = torch.cat(gathered_tensors, dim=0).cpu().numpy()
    #     all_aad_list = torch.cat(gathered_aad_tensors, dim=0).cpu().numpy()
    # else:
    all_add_list = collect_results(local_add_list)
    all_aad_list = collect_results(local_aad_list)
    all_xy = collect_results(local_err_xy)
    all_z = collect_results(local_err_z)
    all_rot = collect_results(local_err_rot)
    all_max_xy = collect_results(local_batch_max_xy)
    all_max_z = collect_results(local_batch_max_z)
    all_max_rot = collect_results(local_batch_max_rot)
    
    add_list_l = np.mean(all_add_list, axis=0)
    max_acc_id = np.argmax(add_list_l)
    max_acc = np.max(add_list_l)
    
    aad_list_l = np.mean(all_aad_list, axis=0)
    max_acc_aad = np.max(aad_list_l)

    if local_rank == 0:
        writer.add_scalar('Test/Avg XY Error', np.mean(all_xy), epoch)
        writer.add_scalar('Test/Avg Z Error', np.mean(all_z), epoch)
        writer.add_scalar('Test/Avg Rot Error', np.mean(all_rot), epoch)
        print("=" * 30)
        print(f"RANSAC Validation Results")
        print(f"Max Accuracy (ADD):    {max_acc:.4f}")
        print(f"Max Accuracy (AAD):    {max_acc_aad:.4f}")
        print("-" * 30)
        print(f"Avg XY Error:          {np.mean(all_xy):.2f} mm")
        print(f"Avg Z  Error:          {np.mean(all_z):.2f} mm")
        print(f"Avg Rot Error:         {np.mean(all_rot):.2f} deg")
        print("-" * 30)
        print(f"Avg Batch Max XY Err:  {np.mean(all_max_xy):.2f} mm")
        print(f"Avg Batch Max Z  Err:  {np.mean(all_max_z):.2f} mm")
        print(f"Avg Batch Max Rot Err: {np.mean(all_max_rot):.2f} deg")
        print("=" * 30)
    net.train()
    return max_acc_id, float(max_acc), float(max_acc_aad), add_list_l

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
    当 `ide_debug` 为 False 时，启用 (DDP分布式数据并行)训练。  

    DDP 训练：  
    screen -S train_ddp
    PYTHONPATH=. python -u -m torch.distributed.launch --nproc_per_node 2 /root/xxxxxx/s4_p2_train_bf_pbr.py
    python 前可通过`CUDA_VISIBLE_DEVICES=1`指定使用显卡的序号
    
    单卡训练：
    PYTHONPATH=. python -u /root/xxxxxx/s4_p2_train_bf_pbr.py
    '''
    
    # 调试nan
    detect_nan = False
    torch.autograd.set_detect_anomaly(detect_nan) 
    
    ide_debug = False
    
    # Specify the path to the dataset folder.
    # 指定数据集文件夹的路径。
    dataset_name = 'grabv1'
    dataset_path = '/media/ubuntu/DISK-C/YJP/HCCEPose/datasets/'
    dataset_path = os.path.join(dataset_path, dataset_name)
    
    # Specify the name of the subfolder in the dataset used for loading training data.
    # 指定数据集中用于加载训练数据的子文件夹名称。
    train_folder_name = 'train_pbr'
    # train_folder_name = 'train'
    val_folder_name = 'val'
    
    # The range of object IDs for training.  
    # `start_obj_id` is the starting object ID, and `end_obj_id` is the ending object ID.
    # 训练的物体 ID 范围。  
    # `start_obj_id` 为起始物体 ID，`end_obj_id` 为终止物体 ID。
    obj_id_list = [4, 5]
    # obj_id_list = [1, 2, 3, 4, 5]
    
    # 主干网络类型
    net_name = 'convnext'
    
    # Total number of training batches.
    # 总训练批次。
    total_epochs = 80
    
    # Learning rate.
    # 学习率。
    hcce_lr = 5e-4
    patch_pnp_lr = 2e-4
    
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
    
    # Whether to enable load_breakpoint.
    # 是否启用 load_breakpoint 加载断点。
    load_breakpoint = False
    manual_load_path = '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grabv1/pose_estimation2/2026-04-13_21:56:21'
    
    # 配置学习率衰减
    warmup_epochs = 3
    
    # 备份存储位置
    output_save = '/media/ubuntu/DISK-C/YJP/HCCEPose/output/'
    
    
    # Loss 权重因子
    loss_factors = {
        'Front_L1Losses': 2.0,
        'Back_L1Losses': 2.0,
        'mask_loss': 1.0,
        'coord_front_loss': 1.0,
        'coord_back_loss': 1.0,
        'center_loss': 1.5,
        'z_loss': 2.0,
        # 'r_loss': 0.0,
        'pm_r_loss': 1.5,
        # 'pm_xy_loss': 1.0,
        # 'pm_z_loss': 2.0,
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
        output_save = os.path.join(output_save, dataset_name, 'pose_estimation2', now_stamp.strftime('%Y-%m-%d_%H:%M:%S'))
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
    np.random.seed(local_rank)
    
    bop_dataset_item = BopDataset(dataset_path, local_rank=local_rank)
    train_bop_dataset_back_front_item = TrainBopDatasetBF_PnPNet(bop_dataset_item, train_folder_name, padding_ratio=padding_ratio)
    
    # ratio = 0.1 means selecting 10% of samples from the dataset for testing.
    # ratio = 0.1 表示从数据集中选择 10% 的样本作为测试数据。
    test_bop_dataset_back_front_item = TestBopDatasetBF_PnPNet(bop_dataset_item, val_folder_name, padding_ratio=padding_ratio, ratio=1.0)
        
    for obj_id in obj_id_list:
        obj_path = bop_dataset_item.obj_model_list[bop_dataset_item.obj_id_list.index(obj_id)]
        print(obj_path)
        obj_ply = load_ply(obj_path)
        train_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
        test_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
        # obj_info = bop_dataset_item.obj_info_list[bop_dataset_item.obj_id_list.index(obj_id)]
        obj_info = train_bop_dataset_back_front_item.model_info_obj
        
        # 对模型中的点进行采样
        model_v = np.asarray(obj_ply['pts'])
        if model_v.shape[0] > 1000:
            selection = np.random.choice(model_v.shape[0], 1000, replace=False)
            model_points = model_v[selection]
        else:
            model_points = model_v
        model_points = torch.from_numpy(model_points).float().to('cuda:'+CUDA_DEVICE)
        
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
                os.makedirs(tb_log_dir, exist_ok=True)
                if load_breakpoint:
                    potential_old_runs = None
                    base_dir = os.path.join(manual_load_path, 'obj_%s'%str(obj_id).rjust(2, '0'))
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
                    writer.add_scalar('Test/AAD-S_Accuracy(5mm)', 0.0, 0)


        # Get the 3D dimensions of the object.
        # 获取物体的 3D 尺寸。
        min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        is_symmetric = False
        sym_infos = None
        if 'symmetries_discrete' in obj_info and len(obj_info['symmetries_discrete']) > 0:
            is_symmetric = True
            sym_infos = obj_info['symmetries_discrete']
        if 'symmetries_continuous' in obj_info and len(obj_info['symmetries_continuous']) > 0:
            is_symmetric = True
            sym_infos = obj_info['symmetries_continuous']
        sym_infos = torch.from_numpy(sym_infos).to('cuda:'+CUDA_DEVICE)
        
        # Define the loss function and neural network.
        # 定义损失函数和神经网络。
        loss_net = HccePose_PatchPnP_Loss(size_xyz, is_symmetric)
        scaler = GradScaler()
        net = HccePose_PatchPnP_Net(
                net=net_name,
                input_channels=3, 
                min_xyz=min_xyz,
                size_xyz=size_xyz,
            )
        net_test = HccePose_PatchPnP_Net(
                net=net_name,
                input_channels=3, 
                min_xyz=min_xyz,
                size_xyz=size_xyz,
            )
        if torch.cuda.is_available():
            net=net.to('cuda:'+CUDA_DEVICE)
            net_test=net_test.to('cuda:'+CUDA_DEVICE)
            loss_net=loss_net.to('cuda:'+CUDA_DEVICE)
        
        optimizer=optim.Adam(net.parameters(), lr=hcce_lr)   
        if net_name == 'convnext':
            base_params = itertools.chain(net.net.parameters(), net.decode_net.parameters())
            optimizer=torch.optim.AdamW([
                    {'params': base_params, 'lr': hcce_lr},                    # param_groups[0]
                    {'params': net.pnp_net.parameters(), 'lr': patch_pnp_lr},  # param_groups[1]
                ], 
                weight_decay=1e-4
            )
        
        # Update the training and testing data loaders respectively.
        # 分别更新训练和测试数据加载器。
        # train_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path) # 提到前面去更新
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
        
        # test_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
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
        
        # scheduler
        iter_per_epoch = len(train_loader)
        
        # scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, 
        #     start_factor=0.01, 
        #     end_factor=1.0, 
        #     total_iters=warmup_epochs * iter_per_epoch
        # )
        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, 
        #     T_max=(total_epochs - warmup_epochs) * iter_per_epoch, 
        #     eta_min=1e-6
        # )
        # scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimizer, 
        #     schedulers=[
        #         scheduler_warmup, 
        #         scheduler_cosine,
        #         ], 
        #     milestones=[warmup_epochs * iter_per_epoch,]
        # )
        
        def get_asynchronous_lr_lambdas():
            min_factor = 0.01 # 最终都降到初始值的 1% (即 1e-6 和 3e-6)

            # 规则 A：Backbone 的快退火
            warmup_iters = warmup_epochs * iter_per_epoch
            backbone_cool_iters = total_epochs * iter_per_epoch // 3 * 2
            total_iters = total_epochs * iter_per_epoch
            hold_iters = total_epochs * iter_per_epoch // 3
            # backbone_cool_iters = total_iters
            def backbone_lambda(iter):
                if iter < warmup_iters:
                    # 线性 Warmup: 0.01 -> 1.0
                    return min_factor + (1.0 - min_factor) * (iter / warmup_iters)
                elif iter < backbone_cool_iters:
                    # 余弦退火: 1.0 -> 0.01
                    progress = (iter - warmup_iters) / (backbone_cool_iters - warmup_iters)
                    cos_out = (1 + math.cos(math.pi * progress)) / 2
                    return min_factor + (1.0 - min_factor) * cos_out
                else:
                    # 彻底冷却，保持在 1% (1e-6) 锁定坐标图
                    return min_factor

            # 规则 B：PatchPnP 的慢退火
            def pnp_lambda(iter):
                if iter < warmup_iters:
                    # 线性 Warmup
                    return min_factor + (1.0 - min_factor) * (iter / warmup_iters)
                elif iter < hold_iters:
                    return 1.0
                else:
                    # 余弦退火
                    progress = (iter - hold_iters) / (total_iters - hold_iters)
                    cos_out = (1 + math.cos(math.pi * progress)) / 2
                    return min_factor + (1.0 - min_factor) * cos_out
            return[backbone_lambda, pnp_lambda]

        # 挂载调度器
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=get_asynchronous_lr_lambdas()
        )

    

        # Attempt to load weights from an interrupted training session.
        # 尝试加载中断训练时保存的权重。
        
        best_score = 0
        iteration_step = 0
        start_epoch = 0
        
        if load_breakpoint:
            try:
                load_path = os.path.join(manual_load_path, 'obj_%s'%str(obj_id).rjust(2, '0')) if manual_load_path else save_path
                checkpoint_info = load_checkpoint(load_path, net, optimizer, local_rank=local_rank, CUDA_DEVICE=CUDA_DEVICE)
                # print(checkpoint_info)
                best_score = checkpoint_info['best_score']
                iteration_step = checkpoint_info['iteration_step']
                start_epoch = checkpoint_info.get('epoch')
                if start_epoch is None:
                    start_epoch = checkpoint_info.get('epoch_step')
                if start_epoch is None:
                    start_epoch = iteration_step // max(len(train_loader), 1)
                start_epoch = int(start_epoch)
                if 'scheduler_state_dict' in checkpoint_info and checkpoint_info['scheduler_state_dict']:
                    scheduler.load_state_dict(checkpoint_info['scheduler_state_dict'])
                    print(f"Successfully resumed scheduler from step {iteration_step}")
                else:
                    # 备选方案：如果之前的模型没存 scheduler 状态，手动快进
                    print("Scheduler state not found, fast-forwarding...")
                    for _ in range(iteration_step - 1):
                        scheduler.step()
                    print(f"Successfully resumed scheduler from step {iteration_step}")
                start_epoch += 1
            except Exception as e:
                print('no checkpoint', e)
                # raise e
        
        if not ide_debug:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], find_unused_parameters=True)
        
        
        # if args.local_rank == 0:
        #     pbar = tqdm(total=1, desc=f"Training", unit="step", dynamic_ncols=True)
            
        if args.local_rank == 0 and output_save:
            with open(os.path.join(output_save_path, 'config.yaml'), 'w') as f:
                yaml.dump({
                    'ide_debug': ide_debug,
                    'dataset_path': dataset_path,
                    'net_name': net_name,
                    'obj_id_list': obj_id_list,
                    'batch_size': batch_size,
                    'num_workers': num_workers,
                    'log_freq_per_epoch': log_freq_per_epoch,
                    'lr': [hcce_lr, patch_pnp_lr],
                    'padding_ratio': padding_ratio,
                    'load_breakpoint': load_breakpoint,
                    'total_epochs': total_epochs,
                    'warmup_epochs': warmup_epochs,
                    'output_save': output_save,
                    'loss_factors': loss_factors,
                    'clip_grad': clip_grad,
                }, f, sort_keys=False)

            total_params = 0
            for name, module in net.named_modules():
                params = sum(p.numel() for p in module.parameters())
                total_params += params
            with open(os.path.join(output_save_path, 'model.txt'), 'w') as f:
                print(f'total_params:{int(total_params)}', net, file=f)
            
        # net_test.load_state_dict(net.module.state_dict())
        # test(obj_ply, obj_info, 
        #     net_test, test_loader, 
        #     local_rank=args.local_rank, 
        #     world_size=world_size if not ide_debug else 1,
        #     device='cuda:'+CUDA_DEVICE,
        #     writer=writer
        #     )
        
        # Train
        # 训练
        for epoch in range(start_epoch, total_epochs):
            if not ide_debug:
                # 保证 DDP 模式下每个 epoch 的数据打乱顺序不同
                train_loader.sampler.set_epoch(epoch)
                pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)
            
            num_batches = len(train_loader)
            log_interval = max(num_batches // log_freq_per_epoch, 1) # 每隔 10% 的进度记录一次
            logs_done_this_epoch = 0
                
            for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, bbox, cam_K, img_size, cam_R_m2c, cam_t_m2c) in enumerate(train_loader):
                B = rgb_c.shape[0]
                if torch.cuda.is_available():
                    rgb_c=rgb_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    mask_vis_c=mask_vis_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    GT_Front_hcce = GT_Front_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    GT_Back_hcce = GT_Back_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                    bbox = bbox.to('cuda:'+CUDA_DEVICE, non_blocking=True)
                    cam_K = cam_K.to('cuda:'+CUDA_DEVICE, non_blocking=True)
                    img_size = img_size.to('cuda:'+CUDA_DEVICE, non_blocking=True)
                    cam_R_m2c = cam_R_m2c.to('cuda:'+CUDA_DEVICE, non_blocking=True)
                    cam_t_m2c = cam_t_m2c.to('cuda:'+CUDA_DEVICE, non_blocking=True).squeeze(-1)
                    
                    gt_t_site = trans_to_site_batch(cam_t_m2c, cam_K, bbox)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_mask, pred_code, pred_rot_6d, pred_t_site, pred_coords_f, pred_coords_b = net(rgb_c, bbox, img_size)
                with torch.autocast(device_type='cuda', enabled=False):
                    current_factors = copy.copy(loss_factors)
                    
                    pred_code = pred_code.float()
                    pred_mask = pred_mask.float()
                    pred_rot_6d = pred_rot_6d.float()
                    pred_t_site = pred_t_site.float()
    
                    batch_model_points = model_points.unsqueeze(0).expand(B, -1, -1)
                    current_loss = loss_net(
                        pred_code[:, :24], pred_code[:, 24:], pred_mask, 
                        pred_rot_6d, pred_t_site, pred_coords_f, pred_coords_b,
                        GT_Front_hcce, GT_Back_hcce, mask_vis_c, 
                        cam_R_m2c, gt_t_site, 
                        batch_model_points, net if ide_debug else net.module,
                        sym_infos=sym_infos.repeat(B)
                    )
                    
                    l_l = [
                        current_factors['Front_L1Losses'] * torch.sum(current_loss['Front_L1Losses']),
                        current_factors['Back_L1Losses'] * torch.sum(current_loss['Back_L1Losses']),
                        current_factors['mask_loss'] * current_loss['mask_loss'],
                        current_factors['coord_front_loss'] * current_loss['coord_front_loss'],
                        current_factors['coord_back_loss'] * current_loss['coord_back_loss'],
                        current_factors['center_loss'] * current_loss['center_loss'],
                        current_factors['z_loss'] * current_loss['z_loss'],
                        # current_factors['r_loss'] * current_loss['r_loss'],
                        current_factors['pm_r_loss'] * current_loss['pm_r_loss'],
                        # current_factors['pm_xy_loss'] * current_loss['pm_xy_loss'],
                        # current_factors['pm_z_loss'] * current_loss['pm_z_loss'],
                    ] 
                    loss = torch.stack(l_l).sum()
                
                if not detect_nan:
                    nan_found = torch.isnan(loss) or torch.isinf(loss)
                    nan_flag = torch.tensor([int(nan_found)], device=loss.device)

                    if not ide_debug:
                        dist.all_reduce(nan_flag, op=dist.ReduceOp.SUM)

                    if nan_flag.item() > 0:
                        optimizer.zero_grad(set_to_none=True) # 清空梯度
                        for m in net.module.modules():
                            if isinstance(m, torch.nn.BatchNorm2d):
                                m.reset_running_stats()
                        optimizer.zero_grad(set_to_none=True)
                        for m in (net.module.modules() if not ide_debug else net.modules()):
                            if isinstance(m, torch.nn.BatchNorm2d):
                                m.reset_running_stats()
                        del loss, current_loss, pred_mask, pred_code, pred_rot_6d 
                        torch.cuda.empty_cache() # 强制清理显存碎片
                        scaler.update()
                        scheduler.step()
                        continue
                    
                if args.local_rank == 0:
                    # 记录各个分项 Loss
                    if batch_idx % log_interval == 0 and (logs_done_this_epoch < log_freq_per_epoch - 1) or batch_idx == num_batches - 1:
                        if output_save:
                            record_epoch = int((epoch + logs_done_this_epoch * (1.0 / log_freq_per_epoch)) * log_freq_per_epoch)
                            writer.add_scalar('Train/Loss_Total', loss.item(), record_epoch)
                            writer.add_scalar('Train/Loss_Front', torch.sum(current_loss['Front_L1Losses']).item(), record_epoch)
                            writer.add_scalar('Train/Loss_Back', torch.sum(current_loss['Back_L1Losses']).item(), record_epoch)
                            writer.add_scalar('Train/Loss_Mask', current_loss['mask_loss'].item(), record_epoch)
                            writer.add_scalar('Train/Loss_Coord_Front', current_loss['coord_front_loss'].item(), record_epoch)
                            writer.add_scalar('Train/Loss_Coord_Back', current_loss['coord_back_loss'].item(), record_epoch)
                            writer.add_scalar('Train/Loss_Center', current_loss['center_loss'].item(), record_epoch)
                            # writer.add_scalar('Train/Loss_Rotation', current_loss['r_loss'].item(), record_epoch)
                            writer.add_scalar('Train/Loss_Point_Match_Rotation', current_loss['pm_r_loss'].item(), record_epoch)
                            writer.add_scalar('Train/Loss_Z', current_loss['z_loss'].item(), record_epoch)
                            # writer.add_scalar('Train/Loss_Point_Match_XY', current_loss['pm_xy_loss'].item(), record_epoch)
                            # writer.add_scalar('Train/Loss_Point_Match_Z', current_loss['pm_z_loss'].item(), record_epoch)
                            # 记录学习率
                            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], record_epoch)
                            writer.add_scalar('Train/Learning_Rate_PatchPnP', optimizer.param_groups[1]['lr'], record_epoch)
                            logs_done_this_epoch += 1
                    # 更新进度条步数
                    pbar.update(1)

                    # 实时显示Loss到进度条后缀
                    pbar.set_postfix({
                        "Total Loss": f"{loss.item():.4f}"
                    })
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step() 
                
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

            max_acc_id, max_acc, max_acc_aad, add_list_l = test(
                obj_ply, obj_info, 
                net_test, test_loader, 
                local_rank=args.local_rank, 
                world_size=world_size if not ide_debug else 1,
                device='cuda:'+CUDA_DEVICE,
                writer=writer,
                epoch=epoch * log_freq_per_epoch + log_freq_per_epoch - 1
                )
            if args.local_rank == 0 and output_save: 
                writer.add_scalar('Test/ADD-S_Accuracy', max_acc, epoch * log_freq_per_epoch + log_freq_per_epoch - 1)
                writer.add_scalar('Test/AAD-S_Accuracy(5mm)', max_acc_aad, epoch * log_freq_per_epoch + log_freq_per_epoch - 1)
                
            if args.local_rank == 0:
                if max_acc >= best_score:
                    best_score = max_acc
                    save_best_checkpoint(best_save_path, net, optimizer, best_score, iteration_step, scheduler, keypoints_ = add_list_l, epoch=epoch)
                    if output_save:
                        save_best_checkpoint(output_best_save_path, net, optimizer, best_score, iteration_step, scheduler, keypoints_ = add_list_l, epoch=epoch)
                        
                loss_net.print_error_ratio()
                save_checkpoint(save_path, net, iteration_step, best_score, optimizer, 3, scheduler=scheduler, keypoints_ = add_list_l, epoch=epoch)
                if output_save:
                    save_checkpoint(output_save_path, net, iteration_step, best_score, optimizer, 3, scheduler=scheduler, keypoints_ = add_list_l, epoch=epoch)
            
        if args.local_rank == 0:
            print('end the training in iteration_step:', iteration_step)

        torch.cuda.empty_cache()
        time.sleep(5)
