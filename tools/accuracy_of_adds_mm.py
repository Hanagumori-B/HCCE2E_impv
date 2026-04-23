import os
import json
import numpy as np
import pandas as pd
import trimesh
import argparse
from scipy.spatial.distance import cdist
from collections import defaultdict
from tqdm import tqdm
from bop_toolkit_o.bop_toolkit_lib import inout, misc, pose_error

# ================= 配置区域 =================
# 数据集根目录 (包含 models 和 test 文件夹的目录)
DATASET_ROOT = "/media/ubuntu/DISK-C/YJP/HCCEPose/datasets/grabv1" 

# 预测结果 CSV 文件路径
parser = argparse.ArgumentParser()
parser.add_argument('csv_result')
args = parser.parse_args()
CSV_PATH = args.csv_result

# 模型点云采样数量
NUM_POINTS = 5000
# ===========================================

class BOPEvaluator:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.models_dir = os.path.join(root_dir, "models")
        self.test_dir = os.path.join(root_dir, "test") # 如果文件夹名叫 val 请改为 val
        
        # 缓存
        self.models_info = self._load_models_info()
        self.model_points = {} # {obj_id: points_array}
        self.gt_cache = {}     # {scene_id: gt_dict}

    def _load_models_info(self):
        """读取 models_info.json 获取直径和对称信息"""
        info_path = os.path.join(self.models_dir, "models_info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"找不到 {info_path}, BOP格式必须包含此文件以获取直径。")
        
        with open(info_path, 'r') as f:
            data = json.load(f)
        # 转换 key 为 int
        return {int(k): v for k, v in data.items()}

    def get_model_data(self, obj_id):
        """加载模型点云和直径、对称性"""
        obj_id = int(obj_id)
        
        # 1. 获取直径和对称标识
        if obj_id not in self.models_info:
            raise ValueError(f"obj_id {obj_id} 不在 models_info.json 中")
        
        info = self.models_info[obj_id]
        diameter = info['diameter']
        
        # 判断是否对称: 如果有 symmetries_discrete 或 symmetries_continuous 字段且不为空
        is_symmetric = False
        if 'symmetries_continuous' in info:
            if len(info['symmetries_continuous']):
                if "axis" in info['symmetries_continuous'][0]:
                    sym_dicts = misc.get_symmetry_transformations(info, np.pi / 180)
                    
                    # 将其转换为 3x3 矩阵列表
                    sym_matrices = []
                    for sym_d in sym_dicts:
                        sym_matrices.append(sym_d['R'].astype(np.float32))
                    
                    info['symmetries_discrete'] = np.array(sym_matrices)
                    is_symmetric = True

            info.pop("symmetries_continuous")
        
        if 'symmetries_discrete' in info:
            if len(info['symmetries_discrete']) == 0:
                info.pop("symmetries_discrete")
                is_symmetric = False
            else:
                syms = info['symmetries_discrete']
                syms = np.array(syms).astype(np.float32)
                if syms.shape[-1] == 16 or (syms.ndim == 3 and syms.shape[1:] == (4, 4)):
                    syms = syms.reshape(-1, 4, 4)[:, :3, :3] # 只要左上角 3x3
                elif syms.shape[-1] == 9:
                    syms = syms.reshape(-1, 3, 3)
                elif syms.ndim == 3 and syms.shape[1:] == (3, 3):
                    pass
                info['symmetries_discrete'] = syms
                is_symmetric = True
        # 2. 加载点云 (如果没缓存)
        if obj_id not in self.model_points:
            ply_path = os.path.join(self.models_dir, f"obj_{obj_id:06d}.ply")
            mesh = trimesh.load(ply_path)
            # 采样表面点
            points = trimesh.sample.sample_surface(mesh, NUM_POINTS)[0]
            # 确保单位是 mm (BOP 标准是 mm)
            # 如果你的模型本身是米，这里可能需要 points * 1000
            self.model_points[obj_id] = points
        if is_symmetric:
            sym_info = info['symmetries_discrete']
        else:
            sym_info = None
        return self.model_points[obj_id], diameter, is_symmetric, sym_info

    def get_gt_pose(self, scene_id, im_id, obj_id):
        """从 scene_gt.json 获取真值"""
        scene_id = int(scene_id)
        im_id = int(im_id)
        obj_id = int(obj_id)
        
        # 懒加载：如果没读过这个scene的gt，就读一次
        if scene_id not in self.gt_cache:
            # 格式化 scene_id，例如 0 -> "000000"
            scene_dir_name = f"{scene_id:06d}"
            gt_path = os.path.join(self.test_dir, scene_dir_name, "scene_gt.json")
            
            if not os.path.exists(gt_path):
                # 尝试不带补零的文件夹名 (防备非标准命名)
                gt_path_alt = os.path.join(self.test_dir, str(scene_id), "scene_gt.json")
                if os.path.exists(gt_path_alt):
                    gt_path = gt_path_alt
                else:
                    raise FileNotFoundError(f"找不到 GT 文件: {gt_path}")
            
            with open(gt_path, 'r') as f:
                # BOP json key 是 string 类型的 im_id
                original_gt = json.load(f)
                # 转换为 int key 方便查找
                self.gt_cache[scene_id] = {int(k): v for k, v in original_gt.items()}
        
        # 查找特定帧
        scene_gt = self.gt_cache[scene_id]
        if im_id not in scene_gt:
            return None # 该帧没有标注
            
        # 查找特定物体 (一张图可能有多个同类物体，这里简化逻辑，匹配 obj_id)
        # 注意：BOP GT 是一张图里一个 list 的物体。我们需要遍历找到对应 obj_id 的那一个
        # 在 CSV 结果中，如果一张图有多个相同 obj_id，通常很难匹配。
        # 这里假设 CSV 是针对某个具体的 instance 匹配好的，或者取第一个匹配到的 GT。
        
        candidates = [item for item in scene_gt[im_id] if item['obj_id'] == obj_id]
        
        if not candidates:
            return None
        
        # ！！！注意！！！：
        # 如果一张图里有2个杯子(obj_id=1)，而你的 CSV 里预测了一个杯子。
        # 标准评测代码（如 bop_toolkit）会做匈牙利匹配 (Hungarian matching)。
        # 这里为了简化，我们取距离预测值最近的一个 GT，或者默认取第一个 GT。
        # 为简单起见，这里返回所有同类物体的 GT 列表，让计算函数去选最近的那个（贪心策略）。
        
        gt_poses = []
        for item in candidates:
            R = np.array(item['cam_R_m2c']).reshape(3, 3)
            t = np.array(item['cam_t_m2c'])
            gt_poses.append((R, t))
            
        return gt_poses

def sym_add(R_est, t_est, R_gt, t_gt, pts, sym_infos=None):
    # 确保 t 是 (3,1)
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
        if e < min_e: 
            min_e = e
    return min_e

def main():
    evaluator = BOPEvaluator(DATASET_ROOT)
    df = pd.read_csv(CSV_PATH)
    
    threshold_list = [5, 10]
    
    # 总体计数器
    total_correct = np.zeros(len(threshold_list), dtype=np.int32)
    total_count = 0
    
    # 分类计数器：{obj_id: {'correct': 0, 'total': 0}}
    per_obj_stats = defaultdict(lambda: {'correct': np.zeros(len(threshold_list), dtype=np.int32), 'total': 0})
    
    print(f"正在评估 {len(df)} 条记录...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            scene_id = int(row['scene_id'])
            im_id = int(row['im_id'])
            obj_id = int(row['obj_id'])
            
            R_pred = np.fromstring(row['R'], sep=' ').reshape(3, 3)
            t_pred = np.fromstring(row['t'], sep=' ')
            
            points, _, is_symmetric, sym_infos = evaluator.get_model_data(obj_id)
            gt_poses_list = evaluator.get_gt_pose(scene_id, im_id, obj_id)
            
            if not gt_poses_list:
                continue # 无GT跳过
            
            # 计算误差 (取与所有同类GT实例的最小误差)
            min_error = float('inf')
            for (R_gt, t_gt) in gt_poses_list:
                if is_symmetric:
                    err = sym_add(R_pred, t_pred, R_gt, t_gt, points, sym_infos)
                else:
                    err = pose_error.add(R_pred, t_pred, R_gt, t_gt, points)
                if err < min_error:
                    min_error = err
            
            # 判定阈值 0.1d
            for i, th in enumerate(threshold_list):
                threshold = th
                is_correct = min_error <= threshold
                if is_correct:
                    per_obj_stats[obj_id]['correct'][i] += 1
                    total_correct[i] += 1

            per_obj_stats[obj_id]['total'] += 1
            total_count += 1
                
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            continue

        # if total_count % 100 == 0:
        #     print(f"Processed {total_count} ...")

    # === 输出最终结果 ===
    for i, th in enumerate(threshold_list):
        print(f">>>*** ERROR <= {th}mm ***<<<\n" + "="*50)
        print(f"{'OBJECT ID':<10} | {'TOTAL':<8} | {'CORRECT':<8} | {'ACCURACY (%)':<10}")
        print("-" * 50)
        
        # 对 obj_id 排序后输出
        sorted_ids = sorted(per_obj_stats.keys())
        
        for obj_id in sorted_ids:
            stats = per_obj_stats[obj_id]
            acc = (stats['correct'][i] / stats['total']) * 100 if stats['total'] > 0 else 0.0
            print(f"{obj_id:<10} | {stats['total']:<8} | {stats['correct'][i]:<8} | {acc:.2f}")
        
        print("-" * 50)
        
        # 输出总体准确率
        if total_count > 0:
            overall_acc = (total_correct[i] / total_count) * 100
            print(f"{'OVERALL':<10} | {total_count:<8} | {total_correct[i]:<8} | {overall_acc:.2f}")
        else:
            print("未处理任何有效数据")
        print("="*50)

if __name__ == "__main__":
    main()