#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os.path

import cv2
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
from .retinex import retinex
from PIL import Image
'''---------------------------测试用的代码---------------------------------'''
def temp_PILtoTorch(pil_image):
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def img_to_bayer_mask(img, path):
    """Computes binary RGB Bayer mask values from integer pixel coordinates."""
    mask = torch.zeros_like(img)
    for x in range(img.shape[2]):
        for y in range(img.shape[1]):
            mask[0, y, x] = (x % 2 == 0) * (y % 2 == 0)
            # Green is top right (0, 1) and bottom left (1, 0).
            mask[1, y, x] = (x % 2 == 1) * (y % 2 == 0) + (x % 2 == 0) * (y % 2 == 1)
            # Blue is bottom right (1, 1).
            mask[2, y, x] = (x % 2 == 1) * (y % 2 == 1)
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    cv2.imwrite(path, np.uint8(mask.numpy().transpose(1, 2, 0)*255))
    return mask
'''---------------------------测试用的代码上---------------------------------'''




class temp_Camera(nn.Module):
    # 之前的相机模型
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 source_path=None
                 ):
        super(temp_Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.retinex = retinex() #增加的retinex模型

        # try:
        #     self.bayer_mask = Image.open(os.path.join(source_path, "mask", image_name + ".png"))
        #     self.bayer_mask = PILtoTorch(self.bayer_mask)
        # except:
        #     self.bayer_mask = img_to_bayer_mask(image, os.path.join(source_path, "mask", image_name + ".png"))
        #
        # try:
        #     self.pre = Image.open(os.path.join(source_path, "preprocess", image_name + ".png"))
        #     # self.pre = cv2.imread(os.path.join(source_path, "preprocess", image_name + ".png"))
        #     self.pre = PILtoTorch(self.pre)
        # except:
        #     self.pre = self.retinex.forward(image, save_path=os.path.join(source_path, "preprocess", image_name + ".png"))


        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")


        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)# 这里的gt_image是resize以后的
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        # 距离相机平面znear和zfar之间且在视锥内的物体才会被渲染
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans# 相机中心的平移
        self.scale = scale# 相机中心坐标的缩放

        # 世界到相机坐标系的变换矩阵的转置,4×4, W^T
        # getWorld2View2(RC2W, tW2C, trans, scale): W2C
        # getWorld2View2(R, T, trans, scale)).transpose(0, 1): W2C的转置W^T，注意不是C2W
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda() # 投影矩阵的转置 J^T
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda() # 从世界坐标系到图像的变换矩阵  W^T J^T
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # 相机在世界坐标系下的坐标，twc
        # world_view_transform：W2C的转置 Tcw^T
        # world_view_transform.inverse()：Tcw^T^{-1} = Tcw^{-1}^T = Twc^T = [Rwc  0]
        #                                                                   [twc  1]
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class Camera(nn.Module): #新的相机模型
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 train_test_exp=False, is_test_dataset=False, is_test_view=False,
                 source_path = None
                 ):
        super(Camera, self).__init__()

        self.uid = uid# cam_info的id，以images名字的顺序排列1111123334231；遍历cam_infos时enumerate生成的index, 从0开始
        self.colmap_id = colmap_id# 对应的cam_intrinsics的相机的id
        self.R = R# COLMAP生成的是W2C，之前取了转置，C2W   RC2W, tW2C
        self.T = T# tcw
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.retinex = retinex()  # 增加的retinex模型

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution) # 这里的image是resize以后的
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else:
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None and depth_params is not None and depth_params["scale"] > 0:
            invdepthmapScaled = invdepthmap * depth_params["scale"] + depth_params["offset"]
            invdepthmapScaled = cv2.resize(invdepthmapScaled, resolution)
            invdepthmapScaled[invdepthmapScaled < 0] = 0
            if invdepthmapScaled.ndim != 2:
                invdepthmapScaled = invdepthmapScaled[..., 0]
            self.invdepthmap = torch.from_numpy(invdepthmapScaled[None]).to(self.data_device)

            if self.alpha_mask is not None:
                self.depth_mask = self.alpha_mask.clone()
            else:
                self.depth_mask = torch.ones_like(self.invdepthmap > 0)

            if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params[
                "med_scale"]:
                self.depth_mask *= 0
            else:
                self.depth_reliable = True

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

