import os
import json
import numpy as np
import pandas as pd
import trimesh
import argparse
from collections import defaultdict
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from bop_toolkit_o.bop_toolkit_lib import misc, pose_error

# ================= 配置区域 =================
DATASET_ROOT = "/media/ubuntu/DISK-C/YJP/HCCEPose/datasets/grabv1" 
NUM_POINTS = 5000 # 采样点数，建议与训练一致
USE_HUNGARIAN = True

# 如果 CSV 是米(m)，设为 1000.0；如果是毫米(mm)，设为 1.0
UNIT_MULTIPLIER = 1.0 
# ===========================================

def compute_rotation_error(R_est, R_gt, is_symmetric=False, sym_info=None):
    """计算考虑对称性的旋转误差（度）"""
    def geodesic_dist(R1, R2):
        trace = np.trace(np.dot(R1, R2.T))
        arg = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(arg))

    min_deg = geodesic_dist(R_est, R_gt)
    if is_symmetric and sym_info is not None:
        for sym_R in sym_info:
            R_gt_sym = np.dot(R_gt, sym_R)
            deg = geodesic_dist(R_est, R_gt_sym)
            if deg < min_deg:
                min_deg = deg
    return min_deg

class BOPEvaluator:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.models_dir = os.path.join(root_dir, "models")
        self.test_dir = os.path.join(root_dir, "test")
        self.models_info = self._load_models_info()
        self.model_points = {} 
        self.gt_cache = {}

    def _load_models_info(self):
        info_path = os.path.join(self.models_dir, "models_info.json")
        with open(info_path, 'r') as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}

    def get_model_data(self, obj_id):
        """按照您提供的逻辑加载对称性和点云"""
        obj_id = int(obj_id)
        info = self.models_info[obj_id]
        diameter = info['diameter']
        is_symmetric = False
        
        # 复制一份 info 避免原地修改影响后续
        info_copy = info.copy()

        if 'symmetries_continuous' in info_copy:
            if len(info_copy['symmetries_continuous']):
                if "axis" in info_copy['symmetries_continuous'][0]:
                    sym_dicts = misc.get_symmetry_transformations(info_copy, np.pi / 180)
                    sym_matrices = [sym_d['R'].astype(np.float32) for sym_d in sym_dicts]
                    info_copy['symmetries_discrete'] = np.array(sym_matrices)
                    is_symmetric = True
        
        if 'symmetries_discrete' in info_copy:
            if len(info_copy['symmetries_discrete']) > 0:
                syms = np.array(info_copy['symmetries_discrete']).astype(np.float32)
                if syms.shape[-1] == 16 or (syms.ndim == 3 and syms.shape[1:] == (4, 4)):
                    syms = syms.reshape(-1, 4, 4)[:, :3, :3]
                elif syms.shape[-1] == 9:
                    syms = syms.reshape(-1, 3, 3)
                info_copy['symmetries_discrete'] = syms
                is_symmetric = True

        if obj_id not in self.model_points:
            ply_path = os.path.join(self.models_dir, f"obj_{obj_id:06d}.ply")
            mesh = trimesh.load(ply_path)
            self.model_points[obj_id] = trimesh.sample.sample_surface(mesh, NUM_POINTS)[0].astype(np.float32)
            
        sym_info = info_copy.get('symmetries_discrete', None) if is_symmetric else None
        return self.model_points[obj_id], diameter, is_symmetric, sym_info

    def load_scene_gt(self, scene_id):
        if scene_id not in self.gt_cache:
            path = os.path.join(self.test_dir, f"{scene_id:06d}", "scene_gt.json")
            if not os.path.exists(path):
                path = os.path.join(self.test_dir, str(scene_id), "scene_gt.json")
            with open(path, 'r') as f:
                self.gt_cache[scene_id] = {int(k): v for k, v in json.load(f).items()}
        return self.gt_cache[scene_id]

def sym_add_single(R_est, t_est, R_gt, t_gt, pts, sym_infos=None):
    """您提供的 ADD-S 计算核心"""
    t_est = t_est.reshape(3, 1)
    t_gt = t_gt.reshape(3, 1)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_result')
    args = parser.parse_args()

    evaluator = BOPEvaluator(DATASET_ROOT)
    df = pd.read_csv(args.csv_result)
    
    # 统计容器
    stats = defaultdict(lambda: {'rot': [], 'err_x': [], 'err_y': [], 'err_z': [], 'adds': []})

    # 按场景图组处理
    grouped = df.groupby(['scene_id', 'im_id'])
    print(f"分析中... (匹配模式: 全局最优分配, 单位系数: {UNIT_MULTIPLIER})")

    for (scene_id, im_id), img_df in tqdm(grouped):
        scene_gt = evaluator.load_scene_gt(scene_id)
        img_gt_all = scene_gt.get(int(im_id), [])
        
        for obj_id in img_df['obj_id'].unique():
            pts, diameter, is_sym, sym_info = evaluator.get_model_data(obj_id)
            
            # 1. 提取所有预测
            pred_rows = img_df[img_df['obj_id'] == obj_id]
            preds = []
            for _, row in pred_rows.iterrows():
                preds.append({
                    'R': np.fromstring(row['R'], sep=' ').reshape(3, 3),
                    't': np.fromstring(row['t'], sep=' ').reshape(3, 1) * UNIT_MULTIPLIER
                })
            
            # 2. 提取所有真值
            gts = []
            for gt_item in img_gt_all:
                if gt_item['obj_id'] == obj_id:
                    gts.append({
                        'R': np.array(gt_item['cam_R_m2c']).reshape(3, 3),
                        't': np.array(gt_item['cam_t_m2c']).reshape(3, 1)
                    })
            if not gts: continue

            # 3. 计算误差矩阵 (用于匈牙利匹配)
            # 使用 ADD/ADD-S 作为匹配代价是最稳妥的
            cost_matrix = np.zeros((len(preds), len(gts)))
            for i, p in enumerate(preds):
                for j, g in enumerate(gts):
                    if is_sym:
                        cost_matrix[i, j] = sym_add_single(p['R'], p['t'], g['R'], g['t'], pts, sym_info)
                    else:
                        cost_matrix[i, j] = pose_error.add(p['R'], p['t'], g['R'], g['t'], pts)

            # 4. 匈牙利算法最优分配
            if USE_HUNGARIAN:
                p_idx_list, g_idx_list = linear_sum_assignment(cost_matrix)
            else:
                # 贪婪匹配：每个预测值找矩阵中对应的最小真值索引
                p_idx_list = np.arange(len(preds))
                g_idx_list = np.argmin(cost_matrix, axis=1)

            # 5. 记录匹配对的解耦误差
            for p_idx, g_idx in zip(p_idx_list, g_idx_list):
                p, g = preds[p_idx], gts[g_idx]
                
                rot_e = compute_rotation_error(p['R'], g['R'], is_sym, sym_info)
                dx = np.abs(p['t'][0, 0] - g['t'][0, 0])
                dy = np.abs(p['t'][1, 0] - g['t'][1, 0])
                dz = np.abs(p['t'][2, 0] - g['t'][2, 0]) # 深度误差
                
                stats[obj_id]['rot'].append(rot_e)
                stats[obj_id]['err_x'].append(dx)
                stats[obj_id]['err_y'].append(dy)
                stats[obj_id]['err_z'].append(dz)
                stats[obj_id]['adds'].append(cost_matrix[p_idx, g_idx])

    # === 输出汇总 ===
    print()
    print('MEAN')
    print("="*95)
    print(f"{'OBJ ID':<8} | {'Count':<6} | {'Rot(deg)':<10} | {'X(mm)':<8} | {'Y(mm)':<8} | {'Z(mm)':<8} | {'ADD(S)(mm)':<8}")
    print("-" * 95)

    all_data = defaultdict(list)
    for obj_id in sorted(stats.keys()):
        s = stats[obj_id]
        n = len(s['rot'])
        m = {k: np.mean(v) for k, v in s.items()}
        print(f"{obj_id:<8} | {n:<6} | {m['rot']:<10.3f} | {m['err_x']:<8.2f} | {m['err_y']:<8.2f} | {m['err_z']:<8.2f} | {m['adds']:<8.2f}")
        for k, v in s.items(): all_data[k].extend(v)

    print("-" * 95)
    if all_data['rot']:
        tm = {k: np.mean(v) for k, v in all_data.items()}
        print(f"{'OVERALL':<8} | {len(all_data['rot']):<6} | {tm['rot']:<10.3f} | {tm['err_x']:<8.2f} | {tm['err_y']:<8.2f} | {tm['err_z']:<8.2f} | {tm['adds']:<8.2f}")
    print("="*95)
    
    print()
    print('MEDIAN')
    print("="*95)
    print(f"{'OBJ ID':<8} | {'Count':<6} | {'Rot(deg)':<10} | {'X(mm)':<8} | {'Y(mm)':<8} | {'Z(mm)':<8} | {'ADD(S)(mm)':<8}")
    print("-" * 95)

    all_data = defaultdict(list)
    for obj_id in sorted(stats.keys()):
        s = stats[obj_id]
        n = len(s['rot'])
        m = {k: np.median(v) for k, v in s.items()}
        print(f"{obj_id:<8} | {n:<6} | {m['rot']:<10.3f} | {m['err_x']:<8.2f} | {m['err_y']:<8.2f} | {m['err_z']:<8.2f} | {m['adds']:<8.2f}")
        for k, v in s.items(): all_data[k].extend(v)

    print("-" * 95)
    if all_data['rot']:
        tm = {k: np.median(v) for k, v in all_data.items()}
        print(f"{'OVERALL':<8} | {len(all_data['rot']):<6} | {tm['rot']:<10.3f} | {tm['err_x']:<8.2f} | {tm['err_y']:<8.2f} | {tm['err_z']:<8.2f} | {tm['adds']:<8.2f}")
    print("="*95)

if __name__ == "__main__":
    main()