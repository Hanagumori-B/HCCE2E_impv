import json
import os
import sys
from tqdm import tqdm  

# ================= 配置区域 =================

# 1. 你的自建数据集路径 (指向包含 scene_id 文件夹的目录)
# 例如: /home/data/my_dataset/test
DATASET_ROOT = "/media/ubuntu/DISK-C/YJP/HCCEPose/datasets/grab/test" 

# 2. 输出文件名
OUTPUT_PATH = "/media/ubuntu/DISK-C/YJP/HCCEPose/datasets/grab/seg2d2_grab-test.json"

# 3. 边界框类型选择
# 'bbox_visib': 仅包含物体可见部分的边界框 (推荐，通常用于遮挡场景)
# 'bbox_obj': 物体完整投影的边界框 (即使被遮挡也包含在内)
BBOX_TYPE = 'bbox_visib' 

# ===========================================

def generate_gt_detection_file():
    if not os.path.exists(DATASET_ROOT):
        print(f"错误: 数据集路径不存在 -> {DATASET_ROOT}")
        return

    detection_results = []
    
    # 获取所有场景文件夹 (通常是数字命名，如 000001, 000002)
    scene_ids = sorted([d for d in os.listdir(DATASET_ROOT) 
                        if os.path.isdir(os.path.join(DATASET_ROOT, d))])
    
    print(f"找到 {len(scene_ids)} 个场景，开始处理...")

    for scene_id_str in tqdm(scene_ids):
        scene_dir = os.path.join(DATASET_ROOT, scene_id_str)
        
        # 定义 GT 和 Info 文件的路径
        scene_gt_path = os.path.join(scene_dir, 'scene_gt.json')
        scene_gt_info_path = os.path.join(scene_dir, 'scene_gt_info.json')

        # 检查文件是否存在
        if not os.path.exists(scene_gt_path) or not os.path.exists(scene_gt_info_path):
            print(f"跳过场景 {scene_id_str}: 缺少 scene_gt.json 或 scene_gt_info.json")
            continue

        # 加载 JSON 数据
        with open(scene_gt_path, 'r') as f:
            scene_gt = json.load(f)
        with open(scene_gt_info_path, 'r') as f:
            scene_gt_info = json.load(f)

        # 遍历该场景下的每一张图片
        # json 的 key 是 image_id (字符串格式)
        for img_id_str in scene_gt.keys():
            image_id = int(img_id_str)
            scene_id = int(scene_id_str)

            # 获取该图片中所有物体的列表
            gt_objs = scene_gt[img_id_str]          # 包含 obj_id, cam_R_m2c, cam_t_m2c
            info_objs = scene_gt_info[img_id_str]   # 包含 bbox_visib, bbox_obj, px_count_visib

            # 确保两个列表长度一致
            assert len(gt_objs) == len(info_objs), \
                f"Scene {scene_id} Image {image_id}: GT 和 Info 条目数量不匹配"

            # 遍历图片中的每个物体
            for i in range(len(gt_objs)):
                obj_gt = gt_objs[i]
                obj_info = info_objs[i]

                # 获取 obj_id
                obj_id = int(obj_gt['obj_id'])

                # 获取 bbox
                # 注意：BOP 格式的 bbox 是 [x, y, w, h]
                bbox = obj_info[BBOX_TYPE]

                # 过滤掉无效的 bbox (例如完全不可见的情况，BOP中通常是 [-1, -1, -1, -1])
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue

                # 构建检测结果条目
                entry = {
                    "scene_id": scene_id,
                    "image_id": image_id,
                    "category_id": obj_id,
                    "score": 1.0,           # GT 也就是 100% 置信度
                    "bbox": bbox,           # [x, y, w, h]
                    "time": 0.0             # 耗时占位符
                }
                detection_results.append(entry)

    # 保存结果
    print(f"正在保存 {len(detection_results)} 条检测结果到 {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, 'w') as f:
        # 使用 indent=2 方便人类阅读，如果在意文件大小可以去掉 indent
        json.dump(detection_results, f, indent=4) 
    
    print("done!")

if __name__ == "__main__":
    generate_gt_detection_file()