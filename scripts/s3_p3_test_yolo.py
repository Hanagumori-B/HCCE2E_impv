import os
import json
import cv2
import glob
import torch
from ultralytics import YOLO
from tqdm import tqdm

# ================= 用户配置区域 =================

# 1. 你的 YOLOv11 模型路径 (.pt 文件)
MODEL_PATH = "/media/ubuntu/DISK-C/YJP/HCCEPose/datasets/grab/yolo11/train_obj_s/detection/obj_s/yolo11-detection-obj_s.pt"  

# 2. 测试集数据的根目录
# 假设结构是: DATASET_ROOT/scene_id/rgb/image_id.png
# 例如: /home/data/bop/ycbv/test
DATASET_ROOT = "/media/ubuntu/DISK-C/YJP/HCCEPose/datasets/grab/test" 

# 3. 结果保存路径 (这就是你要传给 hcce 的 bbox_2D 参数)
OUTPUT_JSON_PATH = "/media/ubuntu/DISK-C/YJP/HCCEPose/datasets/grab/yolo11/yolo_detections.json"

# 4. 类别映射 (非常重要！！！)
# YOLO 输出的 class_id 是从 0 开始的 (0, 1, 2...)
# BOP 数据集的 obj_id 通常是从 1 开始的 (1, 2, 3...) 或者特定的 ID
# 比如: YOLO预测的 0 对应 obj_1, 1 对应 obj_2。
# 如果是一一对应且只差 1，可以用 lambda x: x + 1
# 如果不是连续的，请用字典映射，例如: {0: 1, 1: 5, 2: 10}
def map_yolo_id_to_bop_id(yolo_cls_id):
    return int(yolo_cls_id) + 1  # 默认假设：yolo 0 -> bop 1

# 5. 置信度阈值 (过滤掉低质量检测，防止噪声)
CONF_THRESHOLD = 0.7

# ==============================================

def run_inference():
    # 加载模型
    print(f"Loading YOLOv11 model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    detection_results = []
    
    # 获取所有场景文件夹 (假设文件夹名是 scene_id，如 000001, 000002)
    # 根据你的数据集结构调整，如果是扁平结构，请修改此处遍历逻辑
    scene_folders = sorted([f for f in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, f))])
    
    print(f"Found {len(scene_folders)} scenes.")

    for scene_id_str in tqdm(scene_folders, desc="Processing Scenes"):
        scene_path = os.path.join(DATASET_ROOT, scene_id_str)
        rgb_path = os.path.join(scene_path, 'rgb') # BOP 标准通常在 rgb 文件夹下
        
        if not os.path.exists(rgb_path):
            # 兼容有些数据集直接把图放在 scene 目录下
            rgb_path = scene_path 
            
        # 获取该场景下所有图片 (.png, .jpg)
        img_files = sorted(glob.glob(os.path.join(rgb_path, "*.png")) + glob.glob(os.path.join(rgb_path, "*.jpg")))

        for img_file in img_files:
            # 解析 image_id
            # 文件名通常是 000001.png -> image_id = 1
            filename = os.path.basename(img_file)
            image_id = int(filename.split('.')[0])
            scene_id = int(scene_id_str)

            # --- YOLO 推理 ---
            # stream=False 确保加载到内存处理
            # verbose=False 减少打印
            results = model.predict(img_file, conf=CONF_THRESHOLD, verbose=False)[0]

            # 获取推理时间 (ms -> s)
            inference_time = sum(results.speed.values()) / 1000.0

            # 处理每个检测到的框
            for box in results.boxes:
                # 1. 获取边界框 (XYXY 格式: x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 2. 转换为 BOP 格式 (XYWH: x_top_left, y_top_left, width, height)
                x = float(x1)
                y = float(y1)
                w = float(x2 - x1)
                h = float(y2 - y1)
                
                # 3. 获取类别和置信度
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                
                # 4. 映射类别 ID
                obj_id = map_yolo_id_to_bop_id(cls_id)

                # 5. 构建结果字典
                entry = {
                    "scene_id": scene_id,
                    "image_id": image_id,
                    "category_id": obj_id,
                    "score": score,
                    "bbox": [x, y, w, h],  # 必须是 [x, y, w, h]
                    "time": inference_time
                }
                
                detection_results.append(entry)

    # 保存为 JSON
    print(f"Saving {len(detection_results)} detections to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(detection_results, f, indent=4) 
    
    print("Done!")

if __name__ == "__main__":
    run_inference()