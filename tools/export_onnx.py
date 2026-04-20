import os
import sys
import numpy as np
import torch
import torch.onnx
import HccePose.network_model as models
from HccePose.bop_loader import BopDataset


def export_onnx(model: torch.nn.Module, 
                checkpoint: str, 
                input_like: list[int],
                save_dir: str = "model.onnx",
                ):
    state_dict = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    dummy_input = torch.randn(*input_like)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_dir,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['mask', 'binary_code'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} 
    )
    print(f'Save {os.path.abspath(checkpoint)} as {os.path.abspath(save_dir)}')


if __name__ == '__main__':
    dataset_name = 'grabv1'
    dataset_path = '/media/ubuntu/DISK-C/YJP/HCCEPose/datasets/'
    dataset_path = os.path.join(dataset_path, dataset_name)
    obj_id = 1
    
    bop_dataset_item = BopDataset(dataset_path)
    obj_info = bop_dataset_item.obj_info_list[bop_dataset_item.obj_id_list.index(obj_id)]
    min_xyz = torch.from_numpy(np.array([obj_info['min_x'], obj_info['min_y'], obj_info['min_z']],dtype=np.float32))
    size_xyz = torch.from_numpy(np.array([obj_info['size_x'], obj_info['size_y'], obj_info['size_z']],dtype=np.float32))
    
    model = models.HccePose_BF_Net(
        net='convnext',
        input_channels=3, 
        min_xyz=min_xyz,
        size_xyz=size_xyz,
    )
    input_like = [1, 3, 256, 256]
    
    export_onnx(
        model,
        '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grabv1/pose_estimation/2026-04-04_09:28:20/obj_01/best_score/0_8030step54900_60.pt',
        input_like,
        '/media/ubuntu/DISK-C/YJP/HCCEPose/output/grabv1/pose_estimation/2026-04-04_09:28:20/obj_01/best.onnx'
    )
