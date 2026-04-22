import os, sys, torch, time, cv2
import numpy as np
import pandas as pd
from datetime import datetime
from HccePose.bop_loader import BopDataset, TestBopDatasetBackFront, pycoco_utils
from HccePose.network_model import HccePose_BF_Net, load_checkpoint
from torch.cuda.amp import autocast as autocast
from kasal.bop_toolkit_lib.inout import load_ply
from kasal.utils.io_json import write_dict2json
from HccePose.PnP_solver import solve_PnP, solve_PnP_comb
from HccePose.visualization import vis_rgb_mask_Coord
from HccePose.metric import add_s

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
    df['time'] = df.groupby(['scene_id', 'im_id'])['time'].transform('sum')
    df.to_csv(filepath, index=False)

if __name__ == '__main__':
    np.random.seed(0)
    
    net_name = 'convnext'

    dataset_name = 'grabv1'
    
    sys.path.insert(0, os.getcwd())
    current_dir = os.path.dirname(sys.argv[0])
    dataset_path = os.path.join(current_dir, '..', 'datasets', dataset_name)
    
    bbox_2D = '/media/ubuntu/DISK-C/YJP/HCCEPose/datasets/grabv1/test/gt_bbox2d.json'
    # bbox_2D = os.path.join(dataset_path, 'yolo11', 'yolo_detections.json')
    
    csv_save_path = f'/media/ubuntu/DISK-C/YJP/HCCEPose/output/{dataset_name}/test'
    now_stamp = datetime.now()
    csv_save_path = os.path.join(csv_save_path, net_name, now_stamp.strftime('%Y-%m-%d_%H:%M:%S'))
    os.makedirs(csv_save_path, exist_ok=True)
    
    train_folder_name = 'test'
    
    obj_id_list = [1, 2, 3, 4, 5]
    
    checkpoint_map = {
        1: '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grabv1/pose_estimation/2026-04-04_09:28:20/obj_01/best_score/',
        2: '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grabv1/pose_estimation/2026-04-04_09:28:20/obj_02/best_score/',
        3: '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grabv1/pose_estimation/2026-04-04_09:28:20/obj_03/best_score/',
        4: '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grabv1/pose_estimation/2026-04-07_16:58:22/obj_04/best_score/',
        5: '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grabv1/pose_estimation/2026-04-07_16:58:22/obj_05/best_score/',
    }
    
    CUDA_DEVICE = '0'
    
    vis_op = False
    # vis_op = True
    
    pnp_op = 'ransac+vvs+comb' # ['epnp', 'ransac', 'ransac+vvs', 'ransac+comb', 'ransac+vvs+comb']
    pnp_op_l = [['epnp', 'ransac', 'ransac+vvs', 'ransac+comb', 'ransac+vvs+comb'],[0,2,1]]
    
    batch_size = 1 # 为计算时间准确，batch size一定要等于1
    num_workers = 8
    reprojectionError = 4
    
    padding_ratio = 1.5
    
    bop_dataset_item = BopDataset(dataset_path)
    
    if bbox_2D is not None:
        test_bop_dataset_back_front_item = TestBopDatasetBackFront(bop_dataset_item, train_folder_name, padding_ratio=padding_ratio, bbox_2D=bbox_2D)
    else:
        test_bop_dataset_back_front_item = TestBopDatasetBackFront(bop_dataset_item, train_folder_name, padding_ratio=padding_ratio)

    
    pred_list_all = {}
    
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
        
        net = HccePose_BF_Net(
                net=net_name,
                input_channels = 3, 
                min_xyz = min_xyz,
                size_xyz = size_xyz,
            )
        checkpoint_info = load_checkpoint(best_save_path, net, CUDA_DEVICE=CUDA_DEVICE, strict=False)
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
                cam_K = cam_K.cpu().numpy()
            # with autocast():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1_ = time.time()
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
            
            if vis_op is not None:
                vis_rgb_mask_Coord(rgb_c, pred_mask, pred_front_code, pred_back_code, img_path='show_vis.jpg')
            
            pred_mask_np = pred_mask.detach().cpu().numpy()
            pred_front_code_0_np = pred_front_code_0.detach().cpu().numpy()
            pred_back_code_0_np = pred_back_code_0.detach().cpu().numpy()
            results = []
            coord_image_np = coord_image.detach().cpu().numpy()
            
            if pnp_op in ['epnp', 'ransac', 'ransac+vvs']:
                pred_m_f_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], coord_image_np[i], cam_K[i]) for i in range(pred_mask_np.shape[0])]
                for id_, pred_m_f_c_np_i in enumerate(pred_m_f_c_np):
                    result_i = solve_PnP(pred_m_f_c_np_i, pnp_op=pnp_op_l[1][pnp_op_l[0].index(pnp_op)], reprojectionError=reprojectionError)
                    results.append(result_i)
                    mask_rle = pycoco_utils.binary_mask_to_rle(gen_mask(rgb_np, pred_m_bf_c_np_i[0], Bbox[id_].detach().clone().cpu().numpy(), interpolation=cv2.INTER_NEAREST))
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t2_ = time.time()
                    elapsed_time = t2_ - t1_
                    pred_list.append([result_i['rot'], result_i['tvecs'], mask_rle, 
                                        int(scene_id[id_].cpu().numpy()), 
                                        int(image_id[id_].numpy()), 
                                        float(score[id_].numpy()),
                                        elapsed_time
                                        ])
            else:
                pred_m_bf_c_np = [(pred_mask_np[i], pred_front_code_0_np[i], pred_back_code_0_np[i], coord_image_np[i], cam_K[i]) for i in range(pred_mask_np.shape[0])]
                for id_, pred_m_bf_c_np_i in enumerate(pred_m_bf_c_np):
                    if pnp_op == 'ransac+comb':
                        pnp_op_0 = 2
                    else:
                        pnp_op_0 = 1
                    result_i = solve_PnP_comb(pred_m_bf_c_np_i, keypoints_, pnp_op=pnp_op_0, reprojectionError=reprojectionError / 128 * Bbox[id_].detach().clone().cpu().numpy()[2])
                    results.append(result_i)
                    mask_rle = pycoco_utils.binary_mask_to_rle(gen_mask(rgb_np, pred_m_bf_c_np_i[0], Bbox[id_].detach().clone().cpu().numpy(), interpolation=cv2.INTER_NEAREST))
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t2_ = time.time()
                    elapsed_time = t2_ - t1_
                    pred_list.append([result_i['rot'], result_i['tvecs'], mask_rle, 
                                        int(scene_id[id_].cpu().numpy()), 
                                        int(image_id[id_].numpy()), 
                                        float(score[id_].numpy()),
                                        elapsed_time
                                        ])
            print(f'{obj_id}:{batch_idx}:\t\t{t2_ - t1_:.06f}s')

            torch.cuda.empty_cache()
        pred_list_all[obj_id] = pred_list
    
    
    seg2d_list, obj_id_l, scene_id_l, img_id_l, r_l, t_l, score_l = [], [], [], [], [], [], []
    time_l = []
    for obj_id in pred_list_all:
        pred_list = pred_list_all[obj_id]

        for pred_i in pred_list:
            rot, tvecs, mask_rle, scene_id, image_id, score, elapsed = pred_i
            seg2d_list.append(
                {
                    "scene_id"     : int(scene_id),
                    "image_id"     : int(image_id),
                    "category_id"  : int(obj_id),
                    "score"        : float(score),
                    "bbox"         : [-1, -1, -1, -1],
                    "segmentation" : mask_rle,
                    "time"         : elapsed,
                }
            )
            obj_id_l.append(int(obj_id))
            scene_id_l.append(int(scene_id))
            img_id_l.append(int(image_id))
            r_l.append(rot.reshape((3,3)))
            t_l.append(tvecs.reshape((3)))
            score_l.append(float(score))
            time_l.append(float(elapsed))
            
    write_dict2json(os.path.join(csv_save_path, f'seg2d_{dataset_name}-{train_folder_name}.json'), seg2d_list)
    
    write_csv(os.path.join(csv_save_path, f'det6d_{dataset_name}-{train_folder_name}.csv'), obj_id_l, scene_id_l, img_id_l, r_l, t_l, score_l, time_l)

    pass

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
