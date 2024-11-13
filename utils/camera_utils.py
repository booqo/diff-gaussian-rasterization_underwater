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

from scene.cameras import Camera,temp_Camera
import numpy as np
from utils.general_utils import PILtoTorch #后续原版可能弃用了
from utils.graphics_utils import fov2focal
from tqdm import tqdm
import os
import cv2
from PIL import Image

WARNED = False

def loadCam(args, id, cam_info, resolution_scale,is_test_dataset):

    #用于处理深度文件，原版高斯24年九月后面加的，为了处理nerf的数据、
    image = Image.open(cam_info.image_path)
    if cam_info.depth_path != "":
        try:
            invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)
        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None

    # 原始gs
    orig_w, orig_h = image.size

    if args.resolution in [1, 2, 4, 8]: # 原始尺寸/ (缩放倍数)，缩放倍数 = 缩放因子（默认1.0） * 传入缩放参数（默认-1）
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:# 默认执行这里
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600# 这里的1600指的是宽度，1920会缩放至1600，1920/1600=1.2
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)# 1.2*1.0=1.2
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # '''--------------------------------------老代码----------------------------------'''
    # resized_image_rgb = PILtoTorch(cam_info.image, resolution)# 将原始图像resize，然后添加颜色通道，成为CHW的图像
    #
    # gt_image = resized_image_rgb[:3, ...] # 提取张量的前3个通道，表示RGB图像，这里的GTimage已经是根据分辨率resize以后的
    # loaded_mask = None
    #
    # if resized_image_rgb.shape[1] == 4:# 如果 C 通道上的维度为4（有alpha） # Q：这里为什么是第二个？HWC到底是怎样排列的？
    #     loaded_mask = resized_image_rgb[3:4, ...]# 提取C中的第4个通道（Alpha通道）作为 loaded_mask，加mask掩膜在这加
    #
    # return temp_Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
    #               FoVx=cam_info.FovX, FoVy=cam_info.FovY,
    #               image=gt_image, gt_alpha_mask=loaded_mask,
    #               image_name=cam_info.image_name, uid=id, data_device=args.data_device, source_path="/".join(cam_info.image_path.split("/")[:-2]))

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test,source_path="/".join(cam_info.image_path.split("/")[:-2]))


def cameraList_from_camInfos(cam_infos, resolution_scale, args,is_test_dataset):
    camera_list = []

    for id, c in enumerate(tqdm(cam_infos)):#tqdm显示进度条
        camera_list.append(loadCam(args, id, c, resolution_scale,is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
