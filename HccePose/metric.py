from HccePose.bop_loader import pose_error
import numpy as np


def add_s(obj_ply, obj_info, gt_list, pred_list, thresh=0.1, is_symmetric=False, sym_infos=None):
    pts = obj_ply['pts']
    e_list = []
    for (gt_Rt, pred_Rt) in zip(gt_list, pred_list):
        # if 'symmetries_discrete' in obj_info or 'symmetries_continuous' in obj_info:
        #     if len(pts) > 5000:
        #         selected_indices = np.linspace(0, len(pts) - 1, 2000).astype(int)
        #         pts = pts[selected_indices]
        #     e = pose_error.adi(pred_Rt[0], pred_Rt[1], gt_Rt[0], gt_Rt[1], pts)
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
    min_e = pose_error.add(R_est, t_est, R_gt, t_gt, pts)
    if sym_infos is None or len(sym_infos) == 0:
        return min_e
    for sym_R in sym_infos:
        R_gt_sym = R_gt.dot(sym_R)
        e = pose_error.add(R_est, t_est, R_gt_sym, t_gt, pts)
        if e < min_e: min_e = e
    return min_e
