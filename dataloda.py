import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
import sys
from random import randint
import math
from diff_gaussian_rasterization_underwater import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings as GaussianRasterizationSettings2
from diff_gaussian_rasterization import GaussianRasterizer as GaussianRasterizer2
from utils.sh_utils import eval_sh
import matplotlib.pyplot as plt
from utils.loss_utils import l1_loss, ssim, raw_loss, l2_loss, BackscatterLoss, DeattenuateLoss


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, model, medium_mlp, embed_fn, embeddirs_fn,
           scaling_modifier=1.0, override_color=None, c_med=None, sigma_bs=None, sigma_atten=None,colors_enhance = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means# 创建一个与输入点云（高斯模型）大小相同的零张量，用于记录屏幕空间中的点的位置。这个张量将用于计算对于屏幕空间坐标的梯度。
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()  # 尝试保留张量的梯度。这是为了确保可以在反向传播过程中计算对于屏幕空间坐标的梯度。
    except:
        pass

    # Set up rasterization configuration# 计算视场的 tan 值，这将用于设置光栅化配置。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_underwater_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height), # 输出图像高度
        image_width=int(viewpoint_camera.image_width),  # 输出图像宽度
        tanfovx=tanfovx,  # 水平视场角的正切
        tanfovy=tanfovy,  # 垂直视场角的正切
        bg=bg_color,  # 背景颜色
        scale_modifier=scaling_modifier,  # 缩放因子
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False,  # 抗锯齿设置
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_underwater_settings)  # 创建一个高斯光栅化器对象，用于将高斯分布投影到屏幕上。
    means3D = pc.get_xyz  # 获取高斯分布的三维坐标、屏幕空间坐标和不透明度。
    # means3D = torch.float32([[100,100,100],[100,100,100],[100,100,100]])
    means2D = screenspace_points
    opacity = pc.get_opacity

    # colors_enhance = torch.full((H, W, 3), 2, dtype=torch.float32, device="cuda")

    # 如果提供了预先计算的3D协方差矩阵，则使用它。否则，它将由光栅化器根据尺度和旋转进行计算。
    # scales = None
    # rotations = None
    cov3D_precomp = None
    shs = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)  # 获取预计算的三维协方差矩阵。
    else:  # 获取缩放和旋转信息。（对应的就是3D高斯的协方差矩阵了）
        scales = pc.get_scaling  # 缩放三维
        rotations = pc.get_rotation  # 旋转四元数[1,0,0,0]

    # 如果提供了预先计算的颜色，则使用它们。否则，如果希望在Python中从球谐函数中预计算颜色，请执行此操作。如果没有，则颜色将通过光栅化器进行从球谐函数到RGB的转换。

    # 使用python内部写的球鞋，不用submodles

    if override_color is None:
        if pipe.convert_SHs_python:  # 跳进去改ture就行
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    color_attn, color_clr, color_medium, radii, depths = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        medium_rgb=c_med,
        medium_bs=sigma_bs,
        medium_attn=sigma_atten,
        colors_enhance = colors_enhance # 增强的颜色
    )

    rendered_image = color_attn + color_medium
    # 返回一个字典，包含渲染的图像、增强的图像，屏幕空间坐标、可见性过滤器（根据半径判断是否可见）、及每个高斯分布在屏幕上的半径、增强的损失。
    return {"render": rendered_image,  # 渲染的图像
            "viewspace_points": screenspace_points,  # 屏幕空间坐标
            "visibility_filter": radii > 0,  # 可见性过滤器（根据半径判断是否可见)
            "radii": radii,  # 及每个高斯分布在屏幕上的半径
            "medium_rgb": color_medium, # 还要改
            "color_clr": color_clr,  # 还要改
            "depths": depths,  #

            }


def training(dataset, pipe=None, opt=None):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0] #设置背景颜色，根据数据集是否有白色背景来选择
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") #将背景颜色转化为 PyTorch Tensor，并移到 GPU 上。
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    bg = background  # 背景

    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, model=None, medium_mlp=None, embed_fn=None, embeddirs_fn=None)
    net_image = render_pkg["render"]

    # 计算原有的损失
    gt_image = viewpoint_cam.original_image.cuda()  # （3，H，W）

    Ll1 = l1_loss(net_image, gt_image)  # scalar，公式中的 l1，平均绝对误差
    ssim_value = ssim(net_image, gt_image)
    loss1 = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim_value)
    loss1.backward()
    print("dd")


    rendered_image = net_image.permute(1, 2, 0)  # 从[3, H, W]转换为[H, W, 3]

    # 将渲染图像从GPU移动到CPU，并分离梯度，转换为NumPy数组
    rendered_image_np = rendered_image.detach().cpu().numpy()

    # 使用Matplotlib显示渲染结果
    plt.imshow(rendered_image_np)
    plt.axis('off')  # 可选，去掉坐标轴
    plt.show()


    ...


    '''-----------------------原始光栅器---------------------------'''
    # raster_settings = GaussianRasterizationSettings2(
    #     image_height=H,                       # 输出图像高度
    #     image_width=W,                        # 输出图像宽度
    #     tanfovx=tanfovx,                      # 水平视场角的正切
    #     tanfovy=tanfovy,                      # 垂直视场角的正切
    #     bg=bg,                                  # 背景颜色
    #     scale_modifier=1.0,                     # 缩放因子
    #     viewmatrix=viewmatrix,                  # 视图矩阵
    #     projmatrix=projmatrix,                  # 投影矩阵
    #     sh_degree=1,                            # 球谐次数
    #     campos=cam_pos,                         # 相机位置
    #     prefiltered=False,                      # 是否使用预滤波
    #     debug=False,                             # 是否启用调试模式
    #     # antialiasing=False,                     # 抗锯齿设置
    # )
    #
    # # 初始化高斯光栅化器
    # rasterizer = GaussianRasterizer2(raster_settings=raster_settings)
    #
    # colors_precomp =None
    # # 执行高斯光栅化
    # rendered_image, radii, depth_image = rasterizer(
    #     means3D=pts,
    #     means2D=screenspace_points,
    #     shs=shs,
    #     colors_precomp=colors_precomp,
    #     opacities=opacities,
    #     scales=scales,
    #     rotations=rotations,
    #     cov3D_precomp=None)

    '''----------------------------------------------------------'''



if __name__ =="__main__":
    parser = ArgumentParser(description="dataloda")
    lp = ModelParams(parser)#模型参数
    # op = OptimizationParams(parser)#各种优化参数参数
    pp = PipelineParams(parser)
    # args = parser
    args = parser.parse_args()

    args.source_path = "/home/asus/桌面/lowlight_under_watergs/lowlight-gaussian-splatting-underwater/submodules/diff-gaussian-rasterization_underwater/test_colmap_data"
    args.model_path = "/home/asus/桌面/lowlight_under_watergs/lowlight-gaussian-splatting-underwater/submodules/diff-gaussian-rasterization_underwater/test_colmap_data/data"
    training(lp.extract(args),pp.extract(args))
