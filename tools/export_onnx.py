import torch
import torch.onnx
import os
import HccePose.network_model as models


def export_onnx(model: torch.nn.Module, 
                checkpoint: str, 
                input_like: list[int],
                save_dir: str = "model.onnx",
                ):
    state_dict = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)
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
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} 
    )
    print(f'Save {os.path.abspath(checkpoint)} as {os.path.abspath(save_dir)}')


if __name__ == '__main__':
    model = models.HccePose_BF_Net()
    input_like = [1, 3, 256, 256]
    export_onnx(
        model,
        '/media/ubuntu/WIN-E/YJP/HCCEPose/output/grabv1/pose_estimation/2026-04-04_09:28:20/obj_01/best_score/0_8030step54900_60.pt',
        input_like,
        '/media/ubuntu/WIN-E/YJP/HCCEPose/output/grabv1/pose_estimation/2026-04-04_09:28:20/obj_01/best.onnx'
    )
