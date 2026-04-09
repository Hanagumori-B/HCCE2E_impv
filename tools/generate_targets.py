import os
import json
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csv_result')
parser.add_argument('out_dir')
args = parser.parse_args()
# 你的预测 csv 路径
RESULT_PATH = args.csv_result
# 输出的 targets 文件路径
TARGETS_OUT = os.path.join(args.out_dir, "test_targets_bop19.json")

def main():
    # 简单的 CSV 解析，不依赖外部库
    if not os.path.exists(RESULT_PATH):
        print("错误:找不到CSV文件")
        return

    targets = []
    seen = set()

    print("Reading CSV...")
    with open(RESULT_PATH, 'r') as f:
        lines = f.readlines()
        
    # 假设第一行是 header: scene_id,im_id,obj_id,...
    # 找到列索引
    header = lines[0].strip().split(',')
    try:
        idx_scene = header.index('scene_id')
        idx_im = header.index('im_id')
        idx_obj = header.index('obj_id')
    except ValueError:
        print("CSV header 格式不对，必须包含 scene_id, im_id, obj_id")
        return

    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < 3: continue
        
        scene_id = int(parts[idx_scene])
        im_id = int(parts[idx_im])
        obj_id = int(parts[idx_obj])
        
        key = (scene_id, im_id, obj_id)
        if key not in seen:
            targets.append({
                "im_id": im_id,
                "inst_count": 1,
                "obj_id": obj_id,
                "scene_id": scene_id
            })
            seen.add(key)

    with open(TARGETS_OUT, 'w') as f:
        json.dump(targets, f, indent=2)
    
    print(f"生成的 targets 文件已保存到: {os.path.abspath(TARGETS_OUT)}")

if __name__ == "__main__":
    main()