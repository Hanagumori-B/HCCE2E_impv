from HccePose.bop_loader import *
from kasal.bop_toolkit_lib.inout import load_ply


def get_obj_meta(bop_item:BopDataset, target_obj_ids: list[int], device):
    obj_meta = {}
    for o_id in target_obj_ids:
        idx = bop_item.obj_id_list.index(o_id)
        info = bop_item.obj_info_list[idx]
        obj_meta[o_id] = {
            'size': torch.tensor([info['size_x'], info['size_y'], info['size_z']], dtype=torch.float32).to(device),
            'min': torch.tensor([info['min_x'], info['min_y'], info['min_z']], dtype=torch.float32).to(device),
            'ply': load_ply(bop_item.obj_model_list[idx]),
            'info': info
        }
    return obj_meta
    

class Multi_TrainBopDatasetBFEPro(TrainBopDatasetBFEPro):
    def __init__(self, bop_dataset_item, folder_name, obj_ids, **kwargs):
        super().__init__(bop_dataset_item, folder_name, **kwargs)
        self.obj_ids = obj_ids
        self.all_samples = []
        self.samples_count = {}
        
        for o_id in self.obj_ids:
            obj_key = 'obj_%s' % str(o_id).rjust(6, '0')
            if obj_key in self.dataset_info['obj_info']:
                samples = self.dataset_info['obj_info'][obj_key]
                for s in samples:
                    # 为每个样本标记它所属的 obj_id
                    sample_with_id = s.copy()
                    sample_with_id['target_obj_id'] = o_id
                    self.all_samples.append(sample_with_id)
                self.samples_count[obj_key] = len(samples)
        self.nSamples = len(self.all_samples)
        print(f"Multi-Object Dataset Initialized. Total samples: {self.nSamples} for objects: {self.obj_ids}")
        for obj_key in self.samples_count.keys():
            print(f"{obj_key} : {self.samples_count[obj_key]} samples")
    
    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        info_ = self.all_samples[index]
        cur_obj_id = info_['target_obj_id']
        
        # --- 以下逻辑基本复用 TrainBopDatasetBFEPro，但去掉了对 self.current_obj_id 的依赖 ---
        rgb = cv2.imread(info_['rgb'])
        mask_vis = cv2.imread(info_['mask_visib_path'], 0)
        label_image_name = os.path.basename(info_['mask_path']).split('.')[0]
        
        # 这里的路径是通用的，scene 已经区分了不同数据源
        front_label_image_path = os.path.join(self.target_dir_front, info_['scene'], label_image_name + '.png')
        back_label_image_path = os.path.join(self.target_dir_back, info_['scene'], label_image_name + '.png')
        
        GT_Front = cv2.imread(front_label_image_path)
        GT_Back = cv2.imread(back_label_image_path)
        if GT_Front is None: GT_Front = np.zeros((self.crop_size_gt, self.crop_size_gt, 3))
        if GT_Back is None: GT_Back = np.zeros((self.crop_size_gt, self.crop_size_gt, 3))
        
        if self.aug_op == 'imgaug': rgb = self.apply_augmentation(rgb)
        
        Bbox = aug_square_fp32(np.array(info_['bbox_visib']), rgb.shape, padding_ratio=self.padding_ratio)
        rgb_c = crop_square_resize(rgb, Bbox, self.crop_size_img, interpolation=cv2.INTER_LINEAR)
        mask_vis_c = crop_square_resize(mask_vis, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        
        GT_Front_c = crop_square_resize(GT_Front, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Front_hcce = self.hcce_encode(GT_Front_c)
        GT_Back_c = crop_square_resize(GT_Back, Bbox, self.crop_size_gt, interpolation=cv2.INTER_NEAREST)
        GT_Back_hcce = self.hcce_encode(GT_Back_c)
        
        rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce = self.preprocess(rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce)
        
        # 准备相机和位姿
        cam_K = np.array(info_['cam_K']).reshape((3, 3))
        bX, bY, bW, bH = Bbox
        bscale = self.crop_size_img / bW
        cam_K_B = cam_K.copy()
        cam_K_B[0, 0] *= bscale
        cam_K_B[1, 1] *= bscale
        cam_K_B[0, 2] = (cam_K[0, 2] - bX) * bscale
        cam_K_B[1, 2] = (cam_K[1, 2] - bY) * bscale
        
        cam_R_m2c = np.array(info_['cam_R_m2c']).reshape((3, 3))
        cam_t_m2c = np.array(info_['cam_t_m2c']).flatten()
        
        Bbox_tensor = torch.from_numpy(Bbox.astype(np.float32))
        cam_K_tensor = torch.from_numpy(cam_K.astype(np.float32))
        cam_K_B_tensor = torch.from_numpy(cam_K_B.astype(np.float32))
        cam_R_tensor = torch.from_numpy(cam_R_m2c.astype(np.float32))
        cam_t_tensor = torch.from_numpy(cam_t_m2c.astype(np.float32))
        
        return rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox_tensor, cam_K_tensor, cam_K_B_tensor, cam_R_tensor, cam_t_tensor, cur_obj_id
    
    
class Multi_TestBopDatasetBFEPro(TestBopDatasetBFEPro):
    def __init__(self, bop_dataset_item, folder_name, obj_ids, **kwargs):
        super().__init__(bop_dataset_item, folder_name, **kwargs)
        self.obj_ids = obj_ids
        self.all_samples = []
        self.samples_count = {}
        
        # 将所有指定物体的样本汇集
        for o_id in self.obj_ids:
            obj_key = 'obj_%s' % str(o_id).rjust(6, '0')
            if obj_key in self.dataset_info['obj_info']:
                # 处理 ratio 缩放（如果 TestDataset 设置了 ratio）
                samples = self.dataset_info['obj_info'][obj_key]
                if self.ratio != 1.0:
                    samples = samples[:int(len(samples) * self.ratio)]
                for s in samples:
                    sample_with_id = s.copy()
                    sample_with_id['target_obj_id'] = o_id
                    self.all_samples.append(sample_with_id)
                self.samples_count[obj_key] = len(samples)
                
        self.nSamples = len(self.all_samples)           
        print(f"Multi-Object Dataset Initialized. Total samples: {self.nSamples} for objects: {self.obj_ids}")
        for obj_key in self.samples_count.keys():
            print(f"{obj_key} : {self.samples_count[obj_key]} samples")
    
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        info_ = self.all_samples[index]
        cur_obj_id = info_['target_obj_id']
        
        # 复用父类的单样本获取逻辑，但需绕过 update_obj_id 的内部限制
        # 直接调用 super().__getitem__ 会因为它内部查找 self.current_obj_id 而报错
        # 因此这里我们手动解构 super().__getitem__ 的逻辑
        
        cam_K = np.array(info_['cam_K']).reshape((3,3))
        cam_R_m2c = np.array(info_['cam_R_m2c']).reshape((3,3)) if 'cam_R_m2c' in info_ else np.eye(3)
        cam_t_m2c = np.array(info_['cam_t_m2c']).reshape((3,1)) if 'cam_t_m2c' in info_ else np.zeros((3,1))
        
        rgb = cv2.imread(info_['rgb'])
        mask_vis = cv2.imread(info_['mask_visib_path'], 0) if 'mask_visib_path' in info_ else np.zeros((self.crop_size_gt, self.crop_size_gt))
        
        if 'mask_path' in info_:
            label_name = os.path.basename(info_['mask_path']).split('.')[0]
            GT_Front = cv2.imread(os.path.join(self.target_dir_front, info_['scene'], label_name + '.png'))
            GT_Back = cv2.imread(os.path.join(self.target_dir_back, info_['scene'], label_name + '.png'))
        else:
            GT_Front, GT_Back = None, None
            
        if GT_Front is None: GT_Front = np.zeros((self.crop_size_gt, self.crop_size_gt, 3))
        if GT_Back is None: GT_Back = np.zeros((self.crop_size_gt, self.crop_size_gt, 3))

        Bbox = pad_square_fp32(np.array(info_['bbox_visib']), padding_ratio=self.padding_ratio)
        
        rgb_c = crop_square_resize(rgb, Bbox, self.crop_size_img, cv2.INTER_LINEAR)
        mask_vis_c = crop_square_resize(mask_vis, Bbox, self.crop_size_gt, cv2.INTER_NEAREST)
        GT_Front_c = crop_square_resize(GT_Front, Bbox, self.crop_size_gt, cv2.INTER_NEAREST)
        GT_Front_hcce = self.hcce_encode(GT_Front_c)
        GT_Back_c = crop_square_resize(GT_Back, Bbox, self.crop_size_gt, cv2.INTER_NEAREST)
        GT_Back_hcce = self.hcce_encode(GT_Back_c)
        
        rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce = self.preprocess(rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce)
        
        # 计算 cam_K_B
        bX, bY, bW, _ = Bbox
        bscale = self.crop_size_img / bW
        cam_K_B = cam_K.copy()
        cam_K_B[0, 0] *= bscale
        cam_K_B[1, 1] *= bscale
        cam_K_B[0, 2] = (cam_K[0, 2] - bX) * bscale
        cam_K_B[1, 2] = (cam_K[1, 2] - bY) * bscale
        
        # 返回数据 + cur_obj_id
        if self.bbox_2D is None:
            return rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_K_B, cam_R_m2c, cam_t_m2c, cur_obj_id
        else:
            return rgb_c, mask_vis_c, GT_Front_hcce, GT_Back_hcce, Bbox, cam_K, cam_K_B, cam_R_m2c, cam_t_m2c, int(info_['scene']), int(info_['image']), info_['score'], cur_obj_id
        
        
if __name__ == '__main__':
    dataset_name = 'grab'
    dataset_path = '/media/ubuntu/WIN-E/YJP/HCCEPose/datasets/'
    dataset_path = os.path.join(dataset_path, dataset_name)
    bop_dataset_item = BopDataset(dataset_path)
    print(get_obj_meta(bop_dataset_item, [1, 2, 3], '1'))
    train_bop_dataset = Multi_TrainBopDatasetBFEPro(bop_dataset_item, 'train', [1,2,3])
    test_bop_dataset = Multi_TestBopDatasetBFEPro(bop_dataset_item, 'val', [1,2,3])
