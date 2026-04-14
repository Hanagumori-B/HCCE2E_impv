import os
import torch
import argparse
import random
import math
import shutil
from datetime import datetime, timedelta
import itertools
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from HccePose.bop_loader import BopDataset, TrainBopDatasetBFEPro, TestBopDatasetBFEPro
from HccePose.network_model import HccePose_BF_Net, HccePose_Loss, load_checkpoint, save_checkpoint, save_best_checkpoint
# from torch.cuda.amp import autocast as autocast
from torch.amp import GradScaler
from torch import optim
import torch.distributed as dist
from HccePose.visualization import vis_rgb_mask_Coord
from HccePose.PnP_solver import solve_PnP, solve_PnP_comb
from HccePose.metric import add_s, aad_mm
from kasal.bop_toolkit_lib.inout import load_ply
from HccePose.tools.rot_reps import mat2quat_batch, quat2mat_batch, mat2quat
from epropnp.camera import PerspectiveCamera
from epropnp.cost_fun import AdaptiveHuberPnPCost
from epropnp.monte_carlo_pose_loss import MonteCarloPoseLoss
from epropnp.common import evaluate_pnp

def test_ransac(obj_ply,
                obj_info, 
                net: HccePose_BF_Net, 
                test_loader: torch.utils.data.DataLoader,
                local_rank,
                world_size,
                device=None,
                writer=None,
                ):
    net.eval()
    local_add_list = []
    local_aad_list = []
    disable_tqdm = (local_rank != 0)
    for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_K_B, cam_R_m2c, cam_t_m2c) in tqdm(
        enumerate(test_loader), total=len(test_loader), desc='Validation', postfix='<' * 25, disable=disable_tqdm):
        if torch.cuda.is_available():
            rgb_c=rgb_c.to(device, non_blocking = True)
            mask_vis_c=mask_vis_c.to(device, non_blocking = True)
            GT_Front_hcce = GT_Front_hcce.to(device, non_blocking = True)
            GT_Back_hcce = GT_Back_hcce.to(device, non_blocking = True)
            Bbox = Bbox.to(device, non_blocking = True)
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
                bs = pred_mask.shape[0]
                for i, (cam_R_m2c_i, cam_t_m2c_i, pred_m_bf_c_np_i) in enumerate(zip(
                    cam_R_m2c.detach().cpu().numpy(), cam_t_m2c.detach().cpu().numpy(), pred_m_bf_c_np)):
                    info_list = solve_PnP_comb(pred_m_bf_c_np_i, train=True)
                    if i == 0:
                        sample_0_info = max(info_list, key=lambda x: x['num'])
                    for info_id_, info_i in enumerate(info_list):
                        info_list[info_id_]['add'] = add_s(
                            obj_ply, obj_info, 
                            [[cam_R_m2c_i, cam_t_m2c_i]], 
                            [[info_i['rot'], info_i['tvecs']]])[0]
                        info_list[info_id_]['aad'] = aad_mm(
                            obj_ply, obj_info, 
                            [[cam_R_m2c_i, cam_t_m2c_i]], 
                            [[info_i['rot'], info_i['tvecs']]])[0]
                    add_list = []
                    aad_list = []
                    for i_ in range(len(info_list)):
                        info_list_i = itertools.combinations(info_list, len(info_list) - i_)
                        for info_list_i_j in info_list_i:
                            best_add = 0
                            best_s = 0
                            for info_list_i_j_k in info_list_i_j:
                                if info_list_i_j_k['num'] > best_s:
                                    best_s = info_list_i_j_k['num']
                                    best_add = info_list_i_j_k['add']
                                    best_aad = info_list_i_j_k['aad']
                            add_list.append(best_add)
                            aad_list.append(best_aad)
                    add_list = np.array(add_list)
                    aad_list = np.array(aad_list)
                    local_add_list.append(add_list)
                    local_aad_list.append(aad_list)
                    
            if batch_idx == 0:
                print(f"GT Translation: {cam_t_m2c[0].cpu().numpy().flatten()}")
                print(f"Pi Translation: {sample_0_info['tvecs'].flatten()}")
                print(f"GT Rotation: {cam_R_m2c[0].cpu().numpy().flatten()}")
                print(f"Pi Rotation: {sample_0_info['rot'].flatten()}")
                # if local_rank == 0 and writer is not None:
                #     random_idx = random.randint(0, bs - 1)
                #     gt_m = mask_vis_c[random_idx].detach().cpu().unsqueeze(0)
                #     pred_m = pred_mask[random_idx].detach().cpu().unsqueeze(0)
                #     img_vis = rgb_c[random_idx].detach().cpu().unsqueeze(0)
                #     gt_m_3c = gt_m.repeat(3, 1, 1)
                #     pred_m_3c = pred_m.repeat(3, 1, 1)
                #     pred_front = pred_front_code[random_idx].detach().cpu().permute(2, 0, 1)
                #     pred_back = pred_back_code[random_idx].detach().cpu().permute(2, 0, 1)
                #     gt_front_hcce = GT_Front_hcce[random_idx].unsqueeze(0).permute(0, 2, 3, 1).to(device)
                #     gt_back_hcce = GT_Back_hcce[random_idx].unsqueeze(0).permute(0, 2, 3, 1).to(device)
                #     with torch.no_grad():
                #         gt_front_decoded = net.hcce_decode(gt_front_hcce) / 255.0
                #         gt_back_decoded = net.hcce_decode(gt_back_hcce) / 255.0
                #     gt_front_vis = gt_front_decoded.squeeze(0).permute(2, 0, 1).cpu()
                #     gt_back_vis = gt_back_decoded.squeeze(0).permute(2, 0, 1).cpu()
                #     img_vis = torch.nn.functional.interpolate(img_vis, size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
                #     mask_comparison = torch.cat([img_vis, gt_m_3c, pred_m_3c], dim=2)
                #     front_comparison = torch.cat([img_vis, gt_front_vis, pred_front], dim=2)
                #     back_comparison = torch.cat([img_vis, gt_back_vis, pred_back], dim=2)
                #     writer.add_image('Visual/Mask_Comparison', mask_comparison, global_step=0)
                #     writer.add_image('Visual/Front_Comparison', front_comparison, global_step=0)
                #     writer.add_image('Visual/Back_Comparison', back_comparison, global_step=0)
    torch.cuda.empty_cache()
    
    local_add_list = np.concatenate(local_add_list, axis=0) 
    local_tensor = torch.from_numpy(local_add_list).to(f'cuda:{local_rank}')
    local_aad_list = np.concatenate(local_aad_list, axis=0) 
    local_tensor_aad = torch.from_numpy(local_aad_list).to(f'cuda:{local_rank}')
    
    if world_size > 1:
        # 准备列表接收所有进程的 Tensor
        # DistributedSampler 会自动补齐数据，确保每个进程拿到的数据量一样多
        gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tensors, local_tensor)
        gathered_tensors_aad = [torch.zeros_like(local_tensor_aad) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tensors_aad, local_tensor_aad)
        
        # 在所有进程上合并结果（这样每个进程都有完整的结果，方便后续逻辑一致）
        all_add_list = torch.cat(gathered_tensors, dim=0).cpu().numpy()
        all_aad_list = torch.cat(gathered_tensors_aad, dim=0).cpu().numpy()
    else:
        all_add_list = local_add_list
        all_aad_list = local_aad_list
        
    add_list_l = np.mean(all_add_list, axis=0)
    max_acc_id = np.argmax(add_list_l)
    max_acc = np.max(add_list_l)
    
    aad_list_l = np.mean(all_aad_list, axis=0)
    max_acc_aad = np.max(aad_list_l)
    if local_rank == 0:
        print("="*30)
        print(f"RANSAC Validation Results")
        print(f"Max Accuracy ID: {max_acc_id}")
        print(f"Max Accuracy:    {max_acc:.4f}")
        print(f"Max Accuracy (AAD):    {max_acc_aad:.4f}")
        print("="*30)
    net.train()
    return max_acc_id, float(max_acc), float(max_acc_aad), add_list_l

if __name__ == '__main__':
    '''
    When `ide_debug` is set to True, single-GPU mode is used, allowing IDE debugging.  
    When `ide_debug` is set to False, DDP (Distributed Data Parallel) training is enabled.  

    DDP Training:  
    screen -S train_ddp
    nohup python -u -m torch.distributed.launch --nproc_per_node 2 /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    
    Single-GPU Training:  
    nohup python -u /root/xxxxxx/s4_p2_train_bf_pbr.py > log4.file 2>&1 &
    
    ------------------------------------------------------    
    
    当 `ide_debug` 为 True 时，仅使用单卡，可在 IDE 中进行调试。  
    当 `ide_debug` 为 False 时，启用 DDP(分布式数据并行)训练。  

    DDP 训练：  
    screen -S train_ddp
    PYTHONPATH=. python -u -m torch.distributed.launch --nproc_per_node 2 /root/xxxxxx/s4_p2_train_bf_pbr.py
    python 前可通过`CUDA_VISIBLE_DEVICES=1`指定使用显卡的序号
    
    单卡训练：
    PYTHONPATH=. python -u /root/xxxxxx/s4_p2_train_bf_pbr.py
    '''
    
    # TODO: 用test的代码去跑train的数据集,检查accuracy和loss的情况; 保证训练和验证中关于EProPnP的代码相同.
    
    ide_debug = False
    
    # Specify the path to the dataset folder.
    # 指定数据集文件夹的路径。
    dataset_name = 'grabv1'
    dataset_path = '/media/ubuntu/WIN-E/YJP/HCCEPose/datasets/'
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
    # start_obj_id = 2
    # obj_id_list = [1, 2, 3, 4, 5]
    obj_id_list = [4, 5]
    
    # 主干网络类型
    # net_name = 'convnext'
    net_name = 'resnet'
    
    # Total number of training batches.
    # 总训练批次。
    total_epochs = 65
    
    # Learning rate.
    # 学习率。
    lr = 3e-4
    
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
    
    # 配置开始位姿训练 0从开始就训练， -1不使用epro

    # Whether to enable EfficientNet.
    # 是否启用 EfficientNet。
    # efficientnet_key = None
    
    
    # Whether to enable load_breakpoint.
    # 是否启用 load_breakpoint 加载断点。
    load_breakpoint = False
    manual_load_path = '/media/ubuntu/WIN-E/YJP/HCCEPose/output/grabv1/pose_estimation/2026-03-28_21:13:28'
    validate_after_load = False
    
    # 配置学习率衰减
    warmup_epochs = 3
    
    
    # 备份存储位置
    output_save = '/media/ubuntu/WIN-E/YJP/HCCEPose/output/'
    
    # Loss 权重因子
    loss_factors = {
        'Front_L1Losses': 3.0,
        'Back_L1Losses': 3.0,
        'mask_loss': 1.0,
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
    
    # ratio = 0.1 means selecting 10% of samples from the dataset for testing.
    # ratio = 0.1 表示从数据集中选择 10% 的样本作为测试数据。
    test_bop_dataset_back_front_item = TestBopDatasetBFEPro(bop_dataset_item, val_folder_name, padding_ratio=padding_ratio, ratio=0.67)
    
    for obj_id in obj_id_list:
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
                            
                # writer = SummaryWriter(log_dir=tb_log_dir)
                # print(f"TensorBoard log dir: {tb_log_dir}")
            if not load_breakpoint:
                writer.add_scalar('Test/ADD-S_Accuracy', 0.0, 0)
                writer.add_scalar('Test/AAD-S_Accuracy(5mm)', 0.0, 0)

        # Get the 3D dimensions of the object.
        # 获取物体的 3D 尺寸。
        min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        
        # Define the loss function and neural network.
        # 定义损失函数和神经网络。
        loss_net = HccePose_Loss()
        scaler = GradScaler()
        net = HccePose_BF_Net(
                net = net_name,
                input_channels = 3, 
                min_xyz = min_xyz,
                size_xyz = size_xyz,
            )
        net_test = HccePose_BF_Net(
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
            shuffle=False
            ) 
        
        
        iter_per_epoch = len(train_loader)
    
            
        # scheduler
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
    
        # Attempt to load weights from an interrupted training session.
        # 尝试加载中断训练时保存的权重。
        
        best_score = 0
        iteration_step = 0
        start_epoch = 0
        
        if load_breakpoint:
            try:
                load_path = os.path.join(manual_load_path, 'obj_%s'%str(obj_id).rjust(2, '0')) if manual_load_path else save_path
                if os.path.exists(os.path.join(load_path, 'phase2')):
                    load_path = os.path.join(load_path, 'phase2')
                elif os.path.exists(os.path.join(load_path, 'phase1')):
                    load_path = os.path.join(load_path, 'phase1')
                checkpoint_info = load_checkpoint(load_path, net, optimizer, local_rank=local_rank, CUDA_DEVICE=CUDA_DEVICE)
                best_score = checkpoint_info['best_score']
                iteration_step = checkpoint_info['iteration_step']
                saved_epoch = checkpoint_info.get('epoch_step', iteration_step // len(train_loader))
                start_epoch = saved_epoch + 1
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
                    'lr': lr,
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
                
        if load_breakpoint and validate_after_load:
            if args.local_rank == 0: 
                print(f"Epoch {start_epoch} is running RANSAC")
            max_acc_id, max_acc, max_acc_aad, add_list_l = test_ransac(
                obj_ply, obj_info, net_test, test_loader, 
                local_rank=args.local_rank, 
                world_size=world_size if not ide_debug else 1, 
                device='cuda:'+CUDA_DEVICE,
                writer=writer,
                )
        
        # Train
        # 训练
        for epoch in range(start_epoch, total_epochs):
            if not ide_debug:
                # 这一行非常重要：保证 DDP 模式下每个 epoch 的数据打乱顺序不同
                train_loader.sampler.set_epoch(epoch)
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)
            
            num_batches = len(train_loader)
            log_interval = max(num_batches // log_freq_per_epoch, 1) # 每隔 10% 的进度记录一次
            logs_done_this_epoch = 0
        
            for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, bbox, cam_K, cam_K_B, cam_R_m2c, cam_t_m2c) in enumerate(train_loader):
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
                    pred_mask, pred_code = net(rgb_c)

                    current_loss = loss_net(
                        pred_code[:, :24], pred_code[:, 24:], pred_mask, 
                        GT_Front_hcce, GT_Back_hcce, mask_vis_c
                    )
                    # 构造空的位姿 loss 占位，保持字典结构一致
                    
                    l_l = [
                        loss_factors['Front_L1Losses'] * torch.sum(current_loss['Front_L1Losses']),
                        loss_factors['Back_L1Losses'] * torch.sum(current_loss['Back_L1Losses']),
                        loss_factors['mask_loss'] * current_loss['mask_loss'],
                    ] 
                    
                    loss = torch.stack(l_l).sum()
                    
                
                
                if args.local_rank == 0:
                    if batch_idx % log_interval == 0 and (logs_done_this_epoch < log_freq_per_epoch - 1) or batch_idx == num_batches - 1:
                        if output_save:
                            record_epoch = int((epoch + logs_done_this_epoch * (1.0 / log_freq_per_epoch)) * log_freq_per_epoch)
                            if epoch < total_epochs:
                                writer.add_scalar('Train/Loss_Total', loss.item(), record_epoch)
                            writer.add_scalar('Train/Loss_Front', torch.sum(current_loss['Front_L1Losses']).item(), record_epoch)
                            writer.add_scalar('Train/Loss_Back', torch.sum(current_loss['Back_L1Losses']).item(), record_epoch)
                            writer.add_scalar('Train/Loss_Mask', current_loss['mask_loss'].item(), record_epoch)
                            # 记录学习率
                            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], record_epoch)
                            logs_done_this_epoch += 1
                    # 更新进度条步数
                    pbar.update(1)

                    # 实时显示Loss到进度条后缀
                    pbar.set_postfix({"Tot": f"{loss.item():.4f}"})
                
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                # torch.cuda.empty_cache()
                
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
            max_acc_id, max_acc, max_acc_aad, add_list_l = test_ransac(
                obj_ply, obj_info, net_test, test_loader, 
                local_rank=args.local_rank,
                world_size=world_size if not ide_debug else 1,
                device='cuda:'+CUDA_DEVICE,
                writer=writer,
                )

            if args.local_rank == 0 and output_save: 
                writer.add_scalar('Test/ADD-S_Accuracy', max_acc, epoch * log_freq_per_epoch + log_freq_per_epoch - 1)
                writer.add_scalar('Test/AAD-S_Accuracy(5mm)', max_acc_aad, epoch * log_freq_per_epoch + log_freq_per_epoch - 1)
                
            if args.local_rank == 0:
                if max_acc >= best_score:
                    best_score = max_acc
                    current_phase_best_score = max_acc
                    save_best_checkpoint(best_save_path, net, optimizer, best_score, iteration_step, scheduler, keypoints_ = add_list_l, epoch=epoch)
                    if output_save:
                        save_best_checkpoint(output_best_save_path, net, optimizer, current_phase_best_score, iteration_step, scheduler, keypoints_ = add_list_l, epoch=epoch)
                        
                loss_net.print_error_ratio()
                save_checkpoint(save_path, net, iteration_step, best_score, optimizer, 3, scheduler=scheduler, keypoints_ = add_list_l, epoch=epoch)
                if output_save:
                    save_checkpoint(output_save_path, net, iteration_step, current_phase_best_score, optimizer, 3, scheduler=scheduler, keypoints_ = add_list_l, epoch=epoch)
            
        if args.local_rank == 0:
            print('end the training in iteration_step:', iteration_step)
            
        torch.cuda.empty_cache()

