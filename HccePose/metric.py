from HccePose.bop_loader import pose_error
import numpy as np


def add_s(obj_ply, obj_info, gt_list, pred_list, thresh=0.1):
    pts = obj_ply['pts']
    e_list = []
    for (gt_Rt, pred_Rt) in zip(gt_list, pred_list):
        if 'symmetries_discrete' in obj_info or 'symmetries_continuous' in obj_info:
            if len(pts) > 5000:
                selected_indices = np.linspace(0, len(pts) - 1, 2000).astype(int)
                pts = pts[selected_indices]
            e = pose_error.adi(pred_Rt[0], pred_Rt[1], gt_Rt[0], gt_Rt[1], pts)
        else:
            e = pose_error.add(pred_Rt[0], pred_Rt[1], gt_Rt[0], gt_Rt[1], pts)
        e_list.append(e)
    e_list = np.array(e_list)
    
    pass_list = e_list.copy()
    
    pass_list[pass_list < thresh * obj_info['diameter']] = 0
    pass_list[pass_list > 0] = -1
    pass_list += 1
    
    return np.mean(pass_list), pass_list, e_list

def aad_mm(obj_ply, obj_info, gt_list, pred_list, thresh=5):
    """
    Average Absolute Distance (mm)
    """
    pts = obj_ply['pts']
    e_list = []
    for (gt_Rt, pred_Rt) in zip(gt_list, pred_list):
        if 'symmetries_discrete' in obj_info or 'symmetries_continuous' in obj_info:
            if len(pts) > 5000:
                selected_indices = np.linspace(0, len(pts) - 1, 2000).astype(int)
                pts = pts[selected_indices]
            e = pose_error.adi(pred_Rt[0], pred_Rt[1], gt_Rt[0], gt_Rt[1], pts)
        else:
            e = pose_error.add(pred_Rt[0], pred_Rt[1], gt_Rt[0], gt_Rt[1], pts)
        e_list.append(e)
    e_list = np.array(e_list)
    
    pass_list = e_list.copy()
    
    pass_list[pass_list < thresh] = 0
    pass_list[pass_list > 0] = -1
    pass_list += 1
    
    return np.mean(pass_list), pass_list, e_list
    