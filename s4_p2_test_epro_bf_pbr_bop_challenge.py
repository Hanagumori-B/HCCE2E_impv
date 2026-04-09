import os, sys, torch, time, cv2, math
from datetime import datetime
import numpy as np
import pandas as pd
from HccePose.bop_loader import BopDataset, TestBopDatasetBackFront, pycoco_utils
from HccePose.network_model import HccePose_EPro_Net, load_checkpoint
from torch.cuda.amp import autocast as autocast
from kasal.bop_toolkit_lib.inout import load_ply
from kasal.utils.io_json import write_dict2json
from HccePose.PnP_solver import solve_PnP, solve_PnP_comb
from HccePose.visualization import vis_rgb_mask_Coord
from HccePose.metric import add_s
from epropnp.camera import PerspectiveCamera
from epropnp.cost_fun import AdaptiveHuberPnPCost
from HccePose.tools.rot_reps import mat2quat_batch, quat2mat_batch

def gen_mask(img, mask, Bbox, crop_size=128, interpolation=None):
    Bbox = Bbox.copy()
    center_x = Bbox[0] + 0.5 * Bbox[2]
    center_y = Bbox[1] + 0.5 * Bbox[3]
    w_2 = Bbox[2] / 2
    pts1 = np.float32([[center_x - w_2, center_y - w_2], [center_x - w_2, center_y + w_2], [center_x + w_2, center_y - w_2]])
    pts2 = np.float32([[0, 0], [0, crop_size], [crop_size, 0]])
    M = cv2.getAffineTransform(pts2, pts1)
    mask[mask > 0] = 255
    mask = mask.astype(np.uint8)
    mask_origin = cv2.warpAffine(mask, M, (img.shape[1], img.shape[0]), flags=interpolation)
    return mask_origin

def write_csv(filepath, obj_id_l, scene_id_l, img_id_l, r_l, t_l, score_l):
    data = []
    for obj_id, scene_id, img_id, r, t, score in zip(obj_id_l, scene_id_l, img_id_l, r_l, t_l, score_l):
        R_flat = [float(r[i][j]) for i in range(3) for j in range(3)]
        t_flat = [float(t[i]) for i in range(3)]
        data.append({
            'scene_id': int(scene_id),
            'im_id': int(img_id),
            'obj_id': int(obj_id),
            'score': float(score),
            'R': ' '.join(map(str, R_flat)),
            't': ' '.join(map(str, t_flat)),
            'time': -1,
        })
    df = pd.DataFrame(data, columns=['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
    df.to_csv(filepath, index=False)

if __name__ == '__main__':
    np.random.seed(0)
    
    net_name = 'convnext'

    dataset_name = 'grab'
    
    sys.path.insert(0, os.getcwd())
    current_dir = os.path.dirname(sys.argv[0])
    dataset_path = os.path.join(current_dir, 'datasets', dataset_name)
    
    bbox_2D = '/media/ubuntu/WIN-E/YJP/HCCEPose/datasets/grab/gt_bbox2d.json'
    # bbox_2D = os.path.join(dataset_path, 'yolo11', 'yolo_detections.json')
    
    
    train_folder_name = 'test'
    
    # obj_id_list = [1, 2, 3]
    obj_id_list = [2]
    
    CUDA_DEVICE = '1'
    
    vis_op = False
    # vis_op = True
    
    pnp_op = 'ransac+vvs+comb' # ['epnp', 'ransac', 'ransac+vvs', 'ransac+comb', 'ransac+vvs+comb']
    pnp_op_l = [['epnp', 'ransac', 'ransac+vvs', 'ransac+comb', 'ransac+vvs+comb'],[0,2,1]]
    
    batch_size = 1
    num_workers = 8
    reprojectionError = 4
    
    padding_ratio = 1.5
    efficientnet_key = None
    
    bop_dataset_item = BopDataset(dataset_path)
    
    csv_save_path = '/media/ubuntu/WIN-E/YJP/HCCEPose/output/grab/test'
    now_stamp = datetime.now()
    csv_save_path = os.path.join(csv_save_path, net_name, now_stamp.strftime('%Y-%m-%d_%H:%M:%S'))
    os.makedirs(csv_save_path, exist_ok=True)
    
    if bbox_2D is not None:
        test_bop_dataset_back_front_item = TestBopDatasetBackFront(bop_dataset_item, train_folder_name, padding_ratio=padding_ratio, bbox_2D=bbox_2D)
    else:
        test_bop_dataset_back_front_item = TestBopDatasetBackFront(bop_dataset_item, train_folder_name, padding_ratio=padding_ratio)

    
    pred_list_all = {}
    
    for obj_id in obj_id_list:
        obj_path = bop_dataset_item.obj_model_list[bop_dataset_item.obj_id_list.index(obj_id)]
        print(obj_path)
        
        save_path = os.path.join(dataset_path, 'HccePose', 'obj_%s'%str(obj_id).rjust(2, '0'))
        best_save_path = os.path.join(save_path, 'best_score')
        
        obj_ply = load_ply(obj_path)
        obj_info = bop_dataset_item.obj_info_list[bop_dataset_item.obj_id_list.index(obj_id)]
        
        min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']],dtype=np.float32)).to('cuda:'+CUDA_DEVICE)
        
        net = HccePose_EPro_Net(
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
            
        test_bop_dataset_back_front_item.update_obj_id(obj_id, obj_path)
        test_loader = torch.utils.data.DataLoader(test_bop_dataset_back_front_item, batch_size=batch_size, 
                                                shuffle=False, num_workers=num_workers, drop_last=False) 
        
        rgb_np = cv2.imread(test_bop_dataset_back_front_item.dataset_info['obj_info']['obj_' + str(obj_id).rjust(6, '0')][0]['rgb'])
        
        pred_list = []
        
        for batch_idx, (rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_R_m2c, cam_t_m2c, scene_id, image_id, score) in enumerate(test_loader):
            if torch.cuda.is_available():
                rgb_c=rgb_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                mask_vis_c=mask_vis_c.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                GT_Front_hcce = GT_Front_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                GT_Back_hcce = GT_Back_hcce.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                Bbox = Bbox.to('cuda:'+CUDA_DEVICE, non_blocking = True)
                cam_K = cam_K.to('cuda:'+CUDA_DEVICE)
                cam_K_cpu = cam_K.cpu().numpy()
            # with autocast():
            t1_ = time.time()
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
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
            
            pred_mask = pred_results['pred_mask']
            x3d_f = pred_results['pred_front_code_obj']
            x3d_b = pred_results['pred_back_code_obj']
            x2d_roi = pred_results['coord_2d_image']
            w2d_f = pred_results['w2d_front']
            w2d_b = pred_results['w2d_back']
            scale = pred_results['scale']
            bs, H, W, _ = x3d_f.shape
            pred_mask_np = pred_mask.detach().cpu().numpy()
            pred_f_obj_np = x3d_f.detach().cpu().numpy()
            pred_b_obj_np = x3d_b.detach().cpu().numpy()
            x2d_roi_np = x2d_roi.detach().cpu().numpy()
            
            if vis_op:
                vis_rgb_mask_Coord(rgb_c, pred_mask, pred_front_code, pred_back_code, img_path='show_vis.jpg')
            
            pred_mask_np = pred_mask.detach().cpu().numpy()
            pred_front_code_0_np = pred_front_code_0.detach().cpu().numpy()
            pred_back_code_0_np = pred_back_code_0.detach().cpu().numpy()
            results = []
            coord_image_np = coord_image.detach().cpu().numpy()
            
            # if pnp_op in ['epnp', 'ransac', 'ransac+vvs']:
            #     pred_m_f_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], coord_image_np[i], cam_K[i]) for i in range(pred_mask_np.shape[0])]
            #     for id_, pred_m_f_c_np_i in enumerate(pred_m_f_c_np):
            #         result_i = solve_PnP(pred_m_f_c_np_i, pnp_op=pnp_op_l[1][pnp_op_l[0].index(pnp_op)], reprojectionError=reprojectionError)
            #         results.append(result_i)
            #         mask_rle = pycoco_utils.binary_mask_to_rle(gen_mask(rgb_np, pred_m_bf_c_np_i[0], Bbox[id_].detach().clone().cpu().numpy(), interpolation=cv2.INTER_NEAREST))
            #         pred_list.append([result_i['rot'], result_i['tvecs'], mask_rle, 
            #                             int(scene_id[id_].cpu().numpy()), 
            #                             int(image_id[id_].numpy()), 
            #                             float(score[id_].numpy())])
                    
            # else:
            #     pred_m_bf_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], pred_back_code_0_np[i], coord_image_np[i], cam_K[i]) for i in range(pred_mask_np.shape[0])]
            #     for id_, pred_m_bf_c_np_i in enumerate(pred_m_bf_c_np):
            #         if pnp_op == 'ransac+comb':
            #             pnp_op_0 = 2
            #         else:
            #             pnp_op_0 = 1
            #         result_i = solve_PnP_comb(pred_m_bf_c_np_i, keypoints_, pnp_op=pnp_op_0, reprojectionError=reprojectionError / 128 * Bbox[id_].detach().clone().cpu().numpy()[2])
            #         results.append(result_i)
            #         mask_rle = pycoco_utils.binary_mask_to_rle(gen_mask(rgb_np, pred_m_bf_c_np_i[0], Bbox[id_].detach().clone().cpu().numpy(), interpolation=cv2.INTER_NEAREST))
            #         pred_list.append([result_i['rot'], result_i['tvecs'], mask_rle, 
            #                             int(scene_id[id_].cpu().numpy()), 
            #                             int(image_id[id_].numpy()), 
            #                             float(score[id_].numpy())])
            
            pose_inits = []
            for i in range(bs):
                pred_m_bf_c_np_i = (pred_mask_np[i], pred_f_obj_np[i], pred_b_obj_np[i], x2d_roi_np[i], cam_K_cpu[i])
                result_i = solve_PnP_comb(pred_m_bf_c_np_i, keypoints_, pnp_op=1, reprojectionError=reprojectionError / 128 * Bbox[i].detach().clone().cpu().numpy()[2])
                rot = result_i['rot']
                tvecs = result_i['tvecs']
                # 转换为四元数格式 [qw, qx, qy, qz]
                r_mat = torch.from_numpy(rot).float()
                t_vec = torch.from_numpy(tvecs).float().reshape(3)
                quat = mat2quat_batch(r_mat.unsqueeze(0)) # 注意 Batch 维度
                pose_init = torch.cat([t_vec, quat.squeeze(0)], dim=0)
                pose_inits.append(pose_init)
            
            pose_init_tensor = torch.stack(pose_inits).to(rgb_c.device)
            
            with torch.no_grad():
                # 数据准备：拼接前后表面
                x3d_all = torch.cat([x3d_f.reshape(bs, -1, 3), x3d_b.reshape(bs, -1, 3)], dim=1).float()
                x2d_all = torch.cat([x2d_roi.reshape(bs, -1, 2), x2d_roi.reshape(bs, -1, 2)], dim=1).float()
                w2d_all_raw = torch.cat([w2d_f.reshape(bs, -1, 2), w2d_b.reshape(bs, -1, 2)], dim=1).float()
                
                # 计算权重 (复刻 test_epro 中的权重计算逻辑)
                mask_flat = pred_mask.reshape(bs, -1)
                mask_all = torch.cat([mask_flat, mask_flat], dim=1).unsqueeze(-1).float()
                
                # 归一化权重
                w2d_all = (w2d_all_raw - w2d_all_raw.mean(dim=1, keepdim=True) - math.log(w2d_all_raw.size(1))).exp() * scale[:, None, :]
                w2d_all = torch.nan_to_num(w2d_all, nan=0.0)
                w2d_all = w2d_all * mask_all # 背景点权重置零
                w2d_all = w2d_all.clamp(min=1e-12)
                # 设置相机和代价函数
                wh_begin = Bbox[:, 0:2]
                wh_unit = Bbox[:, 2] / float(W)
                allowed_border = 30 * wh_unit[:, None]
                
                camera = PerspectiveCamera(
                    cam_mats=cam_K.float(), 
                    lb=wh_begin - allowed_border, 
                    ub=wh_begin + Bbox[:, 2:4] + allowed_border
                )
                cost_func = AdaptiveHuberPnPCost(relative_delta=0.1)
                cost_func.set_param(x2d_all, w2d_all)

                # 调用 Monte Carlo Forward 进行位姿精修
                # 在推理阶段，我们通常取 pose_opt (优化后的位姿)
                pose_opt, _, _, _, _, _ = net.epropnp.monte_carlo_forward(
                    x3d_all, x2d_all, w2d_all, camera, cost_func,
                    pose_init=pose_init_tensor, 
                    force_init_solve=False,
                    fast_mode=False
                )

                # 4. 解析结果并存入 pred_list
                pred_rot_final = quat2mat_batch(pose_opt[:, 3:]).cpu().numpy()
                pred_t_final = pose_opt[:, :3].cpu().numpy()

                for i in range(bs):
                    # 将精修后的结果存入原本的推理列表
                    # 注意：mask_rle 部分可以沿用之前的 RANSAC 逻辑
                    mask_rle = pycoco_utils.binary_mask_to_rle(gen_mask(rgb_np, pred_mask_np[i], Bbox[i].cpu().numpy(), interpolation=cv2.INTER_NEAREST))
                    
                    pred_list.append([
                        pred_rot_final[i], 
                        pred_t_final[i].reshape(3, 1), 
                        mask_rle, 
                        int(scene_id[i]), 
                        int(image_id[i]), 
                        float(score[i])
                    ])
    
            t2_ = time.time()
            print(f'{obj_id}:{batch_idx}:\t\t{t2_ - t1_:.06f}s')

            torch.cuda.empty_cache()
        pred_list_all[obj_id] = pred_list
    
    
    seg2d_list, obj_id_l, scene_id_l, img_id_l, r_l, t_l, score_l = [], [], [], [], [], [], []
    for obj_id in pred_list_all:
        pred_list = pred_list_all[obj_id]

        for pred_i in pred_list:
            rot, tvecs, mask_rle, scene_id, image_id, score = pred_i
            seg2d_list.append(
                {
                    "scene_id"     : int(scene_id),
                    "image_id"     : int(image_id),
                    "category_id"  : int(obj_id),
                    "score"        : float(score),
                    "bbox"         : [-1, -1, -1, -1],
                    "segmentation" : mask_rle,
                    "time"         : -1,
                }
            )
            obj_id_l.append(int(obj_id))
            scene_id_l.append(int(scene_id))
            img_id_l.append(int(image_id))
            r_l.append(rot.reshape((3,3)))
            t_l.append(tvecs.reshape((3)))
            score_l.append(float(score))
            
    write_dict2json(os.path.join(csv_save_path, f'seg2d_{dataset_name}-{train_folder_name}.json'), seg2d_list)
    
    write_csv(os.path.join(csv_save_path, f'det6d_{dataset_name}-{train_folder_name}.csv'), obj_id_l, scene_id_l, img_id_l, r_l, t_l, score_l)

    pass

"""
    >>> >>> python /media/ubuntu/WIN-E/YJP/HCCEPose/tools/generate_targets.py <your csv file> <output path (without filename)>

    >>> export BOP_PATH="/media/ubuntu/WIN-E/YJP/HCCEPose/datasets" (export the path of your datasets)
    >>> CUDA_VISIBLE_DEVICES=1 (to choose gpu)
    >>> xvfb-run -a python /media/ubuntu/WIN-E/YJP/HCCEPose/bop_toolkit/scripts/eval_bop19_pose.py\
        --renderer_type=vispy\
        --result_filenames=<path of your csv file, its file name should be like: {algorithm}_{dataset}-{split}.csv>\
        --targets_filename=<path of your target file (test_targets_bop19.json)>\
        --eval_path=<save dir>
"""
