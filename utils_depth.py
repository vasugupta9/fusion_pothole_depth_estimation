import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

class CustomDepthAnything : 
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # 'mps' device on mac not presently working due to some unsupported operations
        print('[pytorch] using device:', self.device)
        
        self.model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format("vitl")).to(self.device).eval()
        total_params = sum(param.numel() for param in self.model.parameters())
        print('Total parameters: {:.2f}M'.format(total_params / 1e6))        
        
        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])


    def get_depth_pred(self, image) : 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        h, w = image.shape[:2]
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)        
        with torch.no_grad():
            depth = self.model(image)
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            depth = depth.cpu().numpy()
        return depth     

    def get_normalised_depth(self, depth):
        norm_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        norm_depth = norm_depth.astype(np.uint8)
        return norm_depth 
            
    def draw_depth_map(self, norm_depth):
        depth_frame = cv2.applyColorMap(norm_depth, cv2.COLORMAP_INFERNO)
        return depth_frame
    