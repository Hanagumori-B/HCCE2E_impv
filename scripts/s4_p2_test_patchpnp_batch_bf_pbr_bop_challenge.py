import os
import sys
import torch
import time
import cv2
import bisect
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Sampler, ConcatDataset
from datetime import datetime
from HccePose.bop_loader import BopDataset, TestBopDatasetBF_PnPNet, pycoco_utils
from HccePose.network_model import HccePose_PatchPnP_Net, load_checkpoint
from torch.cuda.amp import autocast as autocast
from kasal.bop_toolkit_lib.inout import load_ply
from kasal.utils.io_json import write_dict2json
from HccePose.visualization import vis_rgb_mask_Coord
from HccePose.metric import add_s
from HccePose.tools.rot_reps import rot6d_to_mat_batch


class MultiObjectPoseNet(torch.nn.Module):
    def __init__(self, hcce_list_dict):
        super().__init__()
        # 使用 ModuleDict 将多个模型包装在一起
        # 注意: key 必须是字符串
        self.models = torch.nn.ModuleDict({str(k): v for k, v in hcce_list_dict.items()})
        self.streams = {str(k): torch.cuda.Stream() for k in hcce_list_dict.keys()}
        
    def forward(self, img_batch, cam_K, bbox_batch, img_size_tensor, class_ids):
        """
        img_batch:[N, 3, 256, 256] 包含了各种类别的混合 batch
        class_ids: [N] 比如 tensor([1, 1, 2, 3, 1, ...]) 对应每个样本的类别
        """
        device = img_batch.device
        N = img_batch.shape[0]
        
        # 准备好存放整体结果的 tensor
        pred_rot_6d_all = torch.zeros((N, 6), device=device, dtype=torch.float32)
        pred_trans_all = torch.zeros((N, 3), device=device, dtype=torch.float32)
        
        # 寻找当前 Batch 中包含的 unique 类别
        unique_classes = torch.unique(class_ids)
        
        # 内部根据类别路由数据 (这里依然是按类别循环，但在大批量数据下极其高效)
        for cls in unique_classes:
            cls_item = cls.item()
            str_cls = str(cls_item)
            if str_cls not in self.models:
                continue
                
            # 获取属于当前类别的 mask (索引)
            mask = (class_ids == cls)
            
            # 从大 batch 中抽出当前类别的小 batch
            sub_img_batch = img_batch[mask]
            sub_bbox_batch = bbox_batch[mask]
            sub_cam_K = cam_K[mask]
            sub_img_size = img_size_tensor[mask]
            
            with torch.cuda.stream(self.streams[str_cls]):
                sub_results = self.models[str_cls].inference_batch(
                    sub_img_batch, sub_cam_K, sub_bbox_batch, sub_img_size
                )
                pred_rot_6d_all[mask] = sub_results['pred_rot_6d'].to(torch.float32)
                pred_trans_all[mask] = sub_results['pred_trans'].to(torch.float32)
                
        # 同步所有流，等待五个网络在 GPU 上并列计算完毕
        torch.cuda.synchronize()
            
        return {'pred_rot_6d': pred_rot_6d_all, 'pred_trans': pred_trans_all}


class ObjIDWrapperDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, obj_id, use_gt_bbox=False):
            self.dataset = dataset
            self.obj_id = obj_id
            self.use_gt_bbox = use_gt_bbox 

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            # 获取原数据 (返回的是一个长长的 tuple)
            data = self.dataset[idx]
            # 在 tuple 末尾追加类别 id
            if self.use_gt_bbox:
                # 补齐scene, image, score
                scene_id, image_id = self.get_meta(idx)
                score = 1.0  # Ground Truth 可见区域数据的默认置信度设为 1.0
                return (*data, scene_id, image_id, score, self.obj_id)
            else:
                return (*data, self.obj_id)
        
        def get_meta(self, idx):
            inner_ds = self.dataset 
            obj_key = 'obj_%s' % str(inner_ds.current_obj_id).rjust(6, '0')
            info_ = inner_ds.dataset_info['obj_info'][obj_key][idx]
            scene_id = int(info_['scene'])
            image_id = int(info_['image'])
            
            return scene_id, image_id
 

class ImageBatchSampler(Sampler):
    def __init__(self, dataset):  
        self.dataset = dataset
        self.batches =[]
        
        print("正在极速预扫描数据集 (直接读取内存元数据，不加载图像)...")
        image_to_indices = defaultdict(list)
        
        # 遍历超级数据集的总长度（比如 3000）
        for idx in tqdm(range(len(dataset)), desc="打包进度", colour='green'):
            
            # PyTorch ConcatDataset 底层索引映射：把绝对索引映射到具体的子 Dataset
            if isinstance(dataset, ConcatDataset):
                dataset_idx = bisect.bisect_right(dataset.cumulative_sizes, idx)
                if dataset_idx == 0:
                    sample_idx = idx
                else:
                    sample_idx = idx - dataset.cumulative_sizes[dataset_idx - 1]
                # 获取到包裹这一条数据的真正的 ObjIDWrapperDataset
                real_dataset = dataset.datasets[dataset_idx]
            else:
                sample_idx = idx
                real_dataset = dataset
            
            # 调用我们刚才写好的极速查表方法
            sid, iid = real_dataset.get_meta(sample_idx)
            
            key = (sid, iid)
            image_to_indices[key].append(idx)
                
        # 提取出所有的索引列表，每一个子列表代表一张图片上的所有目标
        self.batches = list(image_to_indices.values())
        print(f"扫描完成！共发现 {len(self.batches)} 张包含目标的独立图片。")

    def __iter__(self):
        # 每次 yield 出去的列表，就是一个 Batch 的 indices
        for batch in self.batches:
            yield batch

    def __len__(self):
        # 总 Batch 数量等于图片数量
        return len(self.batches)


def write_csv(filepath, obj_id_l, scene_id_l, img_id_l, r_l, t_l, score_l, time_l):
    data = []
    for obj_id, scene_id, img_id, r, t, score, elapsed in zip(obj_id_l, scene_id_l, img_id_l, r_l, t_l, score_l, time_l):
        R_flat = [float(r[i][j]) for i in range(3) for j in range(3)]
        t_flat = [float(t[i]) for i in range(3)]
        data.append({
            'scene_id': int(scene_id),
            'im_id': int(img_id),
            'obj_id': int(obj_id),
            'score': float(score),
            'R': ' '.join(map(str, R_flat)),
            't': ' '.join(map(str, t_flat)),
            'time': float(elapsed),
        })
    df = pd.DataFrame(data, columns=['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
    # df['time'] = df.groupby(['scene_id', 'im_id'])['time'].transform('sum')
    df.to_csv(filepath, index=False)

if __name__ == '__main__':
    np.random.seed(0)
    
    net_name = 'convnext'
    dataset_name = 'grabv1'
    
    sys.path.insert(0, os.getcwd())
    current_dir = os.path.dirname(sys.argv[0])
    dataset_path = os.path.join(current_dir, '..', 'datasets', dataset_name)
    
    use_gt_bbox = True
    
    if use_gt_bbox:
        bbox_2D = None
    else:
        bbox_2D = '/media/ubuntu/DISK-C/YJP/HCCEPose/datasets/grabv1/test/gt_bbox2d.json'
    # bbox_2D = os.path.join(dataset_path, 'yolo11', 'yolo_detections.json')
    
    csv_save_path = f'/media/ubuntu/DISK-C/YJP/HCCEPose/output/{dataset_name}/test'
    now_stamp = datetime.now()
    csv_save_path = os.path.join(csv_save_path, net_name, now_stamp.strftime('%Y-%m-%d_%H:%M:%S'))
    os.makedirs(csv_save_path, exist_ok=True)
    
    dataset_folder_name = 'test'
    
    obj_id_list = [1, 2, 3, 4, 5]
    
    checkpoint_map = {
        1: '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grabv1/pose_estimation2/2026-04-18_09:52:09/obj_01/best_score/',
        2: '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grabv1/pose_estimation2/2026-04-18_09:52:09/obj_02/best_score/',
        3: '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grabv1/pose_estimation2/2026-04-18_09:52:09/obj_03/best_score/',
        4: '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grabv1/pose_estimation2/2026-04-18_09:52:09/obj_04/best_score/',
        5: '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grabv1/pose_estimation2/2026-04-18_09:52:09/obj_05/best_score/',
    }
    
    CUDA_DEVICE = '0'
    
    vis_op = False
    # vis_op = True

    batch_size = 24
    num_workers = 4
    
    padding_ratio = 1.5
    
    bop_dataset_item = BopDataset(dataset_path)
    
    print(dataset_path)
    # if bbox_2D is not None:
    #     test_bop_dataset_back_front_item = TestBopDatasetBF_PnPNet(bop_dataset_item, dataset_folder_name, padding_ratio=padding_ratio, bbox_2D=bbox_2D)
    # else:
    #     test_bop_dataset_back_front_item = TestBopDatasetBF_PnPNet(bop_dataset_item, dataset_folder_name, padding_ratio=padding_ratio)


    hcce_list_dict = {}
    wrapped_datasets =[]
    for obj_id in obj_id_list:
        obj_path = bop_dataset_item.obj_model_list[bop_dataset_item.obj_id_list.index(obj_id)]
        print(obj_path)
        
        if obj_id in checkpoint_map:
            best_save_path = checkpoint_map[obj_id]
        else:
            save_path = os.path.join(dataset_path, 'HccePose', 'obj_%s'%str(obj_id).rjust(2, '0'))
            # save_path = '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grab/pose_estimation/2026-01-30_13:39:14'
            save_path = os.path.join(save_path, 'obj_%s'%str(obj_id).rjust(2, '0'))
            best_save_path = os.path.join(save_path, 'best_score')
        
        obj_ply = load_ply(obj_path)
        obj_info = bop_dataset_item.obj_info_list[bop_dataset_item.obj_id_list.index(obj_id)]
        
        min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        
        net = HccePose_PatchPnP_Net(
                net=net_name,
                input_channels = 3, 
                min_xyz = min_xyz,
                size_xyz = size_xyz,
            )
        checkpoint_info = load_checkpoint(best_save_path, net, CUDA_DEVICE=CUDA_DEVICE)
        best_score, iteration_step, keypoints_ = \
            checkpoint_info['best_score'], checkpoint_info['iteration_step'], checkpoint_info['keypoints_']
        if torch.cuda.is_available():
            net=net.to('cuda:'+CUDA_DEVICE)
        net.eval()
        hcce_list_dict[str(obj_id)] = net
        
        kwargs = {'padding_ratio': padding_ratio}
        if bbox_2D is not None:
            kwargs['bbox_2D'] = bbox_2D
        ds_item = TestBopDatasetBF_PnPNet(bop_dataset_item, dataset_folder_name, **kwargs)
        ds_item.update_obj_id(obj_id, obj_path)
        
        # 包装并放入列表
        wrapped_datasets.append(ObjIDWrapperDataset(ds_item, obj_id, use_gt_bbox))
        
        # test_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
        # test_loader = torch.utils.data.DataLoader(test_bop_dataset_back_front_item, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False) 
        
        # rgb_np = cv2.imread(test_bop_dataset_back_front_item.dataset_info['obj_info']['obj_' + str(obj_id).rjust(6, '0')][0]['rgb'])
        
        # pred_list = []
        
    mixed_dataset = ConcatDataset(wrapped_datasets)
    image_batch_sampler = ImageBatchSampler(mixed_dataset)
    mixed_loader = torch.utils.data.DataLoader(
        mixed_dataset, 
        batch_sampler=image_batch_sampler, 
        num_workers=num_workers,
        pin_memory=True  # 推荐开启，能加快张量转移到 GPU 的速度
    ) # 注意：使用了 batch_sampler 后，不能再指定 batch_size, shuffle 和 drop_last 参数
    
    
    multi_model = MultiObjectPoseNet(hcce_list_dict).to('cuda:'+CUDA_DEVICE)
    multi_model.eval()
    
    obj_id_l, scene_id_l, img_id_l, r_l, t_l, score_l, time_l = [], [], [], [], [], [],[]

    print(f"Total batches to process: {len(mixed_loader)}")
    
    # 遍历超级 DataLoader
    for batch_idx, batch_data in enumerate(mixed_loader):
        # 解包 13 个返回值（最后 1 个是我们在 Wrapper 中加入的 class_id）
        (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, image_size, 
         cam_R_m2c, cam_t_m2c, scene_id, image_id, score, class_ids) = batch_data
        
        if torch.cuda.is_available():
            rgb_c = rgb_c.to('cuda:'+CUDA_DEVICE, non_blocking=True)
            Bbox = Bbox.to('cuda:'+CUDA_DEVICE, non_blocking=True)
            cam_K = cam_K.to('cuda:'+CUDA_DEVICE, non_blocking=True)
            image_size = image_size.to('cuda:'+CUDA_DEVICE, non_blocking=True)
            class_ids = class_ids.to('cuda:'+CUDA_DEVICE, non_blocking=True)
            
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1_ = time.time()
        
        # 一次性将整个混合了各种框的 Batch 送入你的大模型
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                pred_results = multi_model(rgb_c, cam_K, Bbox, image_size, class_ids)
            
        pred_rot_6d = pred_results['pred_rot_6d']
        pred_trans = pred_results['pred_trans']
        
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t2_ = time.time()
        
        # 此时 current_bs 的含义是当前这张图片里包含了多少个目标
        current_bs = rgb_c.shape[0]
        elapsed_time_per_sample = (t2_ - t1_)
        
        pred_rot_mats = rot6d_to_mat_batch(pred_rot_6d).detach().cpu().numpy()
        pred_trans_np = pred_trans.detach().cpu().numpy()
        
        # 遍历解析当前批次的结果
        for id_ in range(current_bs):
            obj_id_l.append(int(class_ids[id_].item()))
            scene_id_l.append(int(scene_id[id_].item()))
            img_id_l.append(int(image_id[id_].item()))
            r_l.append(pred_rot_mats[id_].reshape((3,3)))
            t_l.append(pred_trans_np[id_].reshape((3)))
            score_l.append(float(score[id_].item()))
            time_l.append(float(elapsed_time_per_sample))
            
        cur_scene = scene_id[0].item()
        cur_image = image_id[0].item()
        print(f'Batch {batch_idx}:[Scene: {cur_scene} | Image: {cur_image}] Processed {current_bs} targets \t Time: {t2_ - t1_:.06f}s')
    
    write_csv(os.path.join(csv_save_path, f'det6d_{dataset_name}-{dataset_folder_name}.csv'), obj_id_l, scene_id_l, img_id_l, r_l, t_l, score_l, time_l)
    print("Done!")

"""
    >>> >>> python /media/ubuntu/DISK-C/YJP/HCCEPose/tools/generate_targets.py <your csv file> <output path (without filename)>

    >>> export BOP_PATH="/media/ubuntu/DISK-C/YJP/HCCEPose/datasets" (export the path of your datasets)
    >>> CUDA_VISIBLE_DEVICES=1 (to choose gpu)
    >>> xvfb-run -a python /media/ubuntu/DISK-C/YJP/HCCEPose/bop_toolkit/scripts/eval_bop19_pose.py\
        --renderer_type=vispy\
        --result_filenames=<path of your csv file, its file name should be like: {algorithm}_{dataset}-{split}.csv>\
        --targets_filename=<path of your target file (test_targets_bop19.json)>\
        --eval_path=<save dir>
"""
