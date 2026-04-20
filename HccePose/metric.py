from HccePose.bop_loader import pose_error
import numpy as np


def add_s(obj_ply, obj_info, gt_list, pred_list, thresh=0.1, is_symmetric=False, sym_infos=None):
    pts = obj_ply['pts']
    e_list = []
    for (gt_Rt, pred_Rt) in zip(gt_list, pred_list):
        if is_symmetric:
            e = sym_add(pred_Rt[0], pred_Rt[1], gt_Rt[0], gt_Rt[1], pts, sym_infos)
        else:
            e = pose_error.add(pred_Rt[0], pred_Rt[1], gt_Rt[0], gt_Rt[1], pts)
        e_list.append(e)
    e_list = np.array(e_list)
    
    pass_list = e_list.copy()
    
    pass_list[pass_list < thresh * obj_info['diameter']] = 0
    pass_list[pass_list > 0] = -1
    pass_list += 1
    
    return np.mean(pass_list), pass_list, e_list

def aad_mm(obj_ply, obj_info, gt_list, pred_list, thresh=5, is_symmetric=False, sym_infos=None):
    """
    Average Absolute Distance (mm)
    """
    pts = obj_ply['pts']
    e_list = []
    for (gt_Rt, pred_Rt) in zip(gt_list, pred_list):
        if is_symmetric:
            e = sym_add(pred_Rt[0], pred_Rt[1], gt_Rt[0], gt_Rt[1], pts, sym_infos)
        else:
            e = pose_error.add(pred_Rt[0], pred_Rt[1], gt_Rt[0], gt_Rt[1], pts)
        e_list.append(e)
    e_list = np.array(e_list)
    
    pass_list = e_list.copy()
    
    pass_list[pass_list < thresh] = 0
    pass_list[pass_list > 0] = -1
    pass_list += 1
    
    return np.mean(pass_list), pass_list, e_list

def sym_add(R_est, t_est, R_gt, t_gt, pts, sym_infos=None):
    pts_est = pose_error.misc.transform_pts_Rt(pts, R_est, t_est)
    pts_gt = pose_error.misc.transform_pts_Rt(pts, R_gt, t_gt)
    diff = pts_est - pts_gt
    min_e = np.sqrt(np.sum(np.square(diff), axis=1)).mean()
    if sym_infos is None or len(sym_infos) == 0:
        return min_e
    for sym_R in sym_infos:
        R_gt_sym = R_gt.dot(sym_R)
        pts_gt_sym = pose_error.misc.transform_pts_Rt(pts, R_gt_sym, t_gt)
        diff_sym = pts_est - pts_gt_sym
        e = np.sqrt(np.sum(np.square(diff_sym), axis=1)).mean()
        if e < min_e: min_e = e
    return min_e

def sym_add_angular_filter(R_est, t_est, R_gt, t_gt, pts, sym_infos=None, top_m=1):
    # 先按角度筛选最接近的位姿，再计算点云距离
    if sym_infos is None or len(sym_infos) == 0:
        R_candidates = R_gt[np.newaxis, ...] # (1, 3, 3)
    else:
        R_gt_syms = R_gt @ sym_infos
        R_candidates = np.concatenate([R_gt[np.newaxis, ...], R_gt_syms], axis=0)
    similarities = np.sum(R_est * R_candidates, axis=(1, 2)) # 计算迹 (Trace)：trace(R_est @ R_cand.T), trace(A @ B.T) 等于 np.sum(A * B)
    m = min(top_m, len(R_candidates)) # np.argsort 是升序排序，所以取最后 top_m 个
    top_indices = np.argsort(similarities)[-m:][::-1] # 迹越大，说明旋转角度越接近, theta = arccos((tr(R1@R2.T)-1)/2), arccos是单调减函数, 故迹越大，theta越小
    
    pts_est = pose_error.misc.transform_pts_Rt(pts, R_est, t_est)
    pts_gt = pose_error.misc.transform_pts_Rt(pts, R_gt, t_gt)
    diff = pts_est - pts_gt
    min_e = np.sqrt(np.sum(np.square(diff), axis=1)).mean()
    for idx in top_indices:
            best_R_gt_cand = R_candidates[idx]
            pts_gt_cand = pose_error.misc.transform_pts_Rt(pts, best_R_gt_cand, t_gt)
            diff = pts_est - pts_gt_cand
            current_error = np.sqrt(np.sum(diff**2, axis=1)).mean()
            if current_error < min_e:
                min_e = current_error
                
    return min_e
