import torch

def site_to_trans_batch(t_site, cam_K, bbox, s_zoom=128.0):
    """
    从GDR-Net 的 t_site 参数还原为绝对平移量 trans  
    
    Args:
        t_site: [B, 3] 还原后的 [delta_x, delta_y, delta_z]
        cam_K: [B, 3, 3] 相机内参
        bbox:  [B, 4] 检测框 [x1, y1, w, h]
        s_zoom: 训练时的 ROI 缩放尺寸 (默认 256)
    
    Return:
        trans: [B, 3] 绝对平移 [tx, ty, tz]
    """
    B = t_site.shape[0]
    
    x1, y1, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    cx = x1 + w * 0.5  # Bbox 中心 x
    cy = y1 + h * 0.5  # Bbox 中心 y
    s_o = torch.max(w, h) # s_o = max(w, h)
    
    # 2D 投影中心 (o_x, o_y)
    delta_x = t_site[:, 0]
    delta_y = t_site[:, 1]
    ox = delta_x * w + cx
    oy = delta_y * h + cy
    
    delta_z = t_site[:, 2]
    r = s_zoom / s_o 
    tz = r * torch.exp(delta_z)  # t_z = delta_z * r
    v = torch.stack([ox, oy, torch.ones_like(ox)], dim=-1).unsqueeze(-1) # [B, 3, 1]
    inv_K = torch.linalg.inv(cam_K)
    
    trans = tz.view(B, 1, 1) * torch.bmm(inv_K, v)
    trans = trans.view(B, 3) # [B, 3]
    
    return trans


def trans_to_site_batch(trans, cam_K, bbox, s_zoom=128.0):
    """
    从绝对平移量 trans 还原为 GDR-Net 的 t_site 参数
    
    Args:
        trans: [B, 3] 绝对平移 [tx, ty, tz]
        cam_K: [B, 3, 3] 相机内参
        bbox:  [B, 4] 检测框 [x1, y1, w, h]
        s_zoom: 训练时的 ROI 缩放尺寸 (默认 256)
    
    Return:
        t_site: [B, 3] 还原后的 [delta_x, delta_y, delta_z]
    """
    B = trans.shape[0]
    
    # 1. 获取 Bbox 相关的参数
    x1, y1, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    s_o = torch.max(w, h)
    
    # 计算投影矢量 P = K @ trans
    p = torch.bmm(cam_K, trans.unsqueeze(-1)).view(B, 3)
    
    px = p[:, 0]
    py = p[:, 1]
    tz = p[:, 2] # 这里的 tz 即为公式中的绝对深度
    
    eps = 1e-8  # 避免除以 0 (通常深度 tz > 0)
    ox = px / (tz + eps)
    oy = py / (tz + eps)
    
    # 还原 SITE 参数 根据公式
    # delta_x = (ox - cx) / w
    # delta_y = (oy - cy) / h
    # delta_z = tz / r , 其中 r = s_zoom / s_o
    
    delta_x = (ox - cx) / w
    delta_y = (oy - cy) / h
    
    r = s_zoom / s_o
    delta_z = torch.log(tz / (r + eps) + eps)
    
    # 4. 合并结果
    t_site = torch.stack([delta_x, delta_y, delta_z], dim=-1)
    
    return t_site


if __name__ == '__main__':
    t = torch.tensor([-9.0421e+01, -1.7380e+02,  7.9580e+02]).view(1, 3)
    cam_K = torch.tensor([867.83, 0.0, 382.97, 
                          0.0, 868.02, 259.37, 
                          0.0, 0.0, 1.0]).view(1, 3, 3)
    bbox = torch.tensor([236.7570,  36.5019,  88.2151,  88.2151]).view(1, 4)
    print(t)
    tt = trans_to_site_batch(t, cam_K, bbox)
    print(tt)
    print(site_to_trans_batch(tt, cam_K, bbox))