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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
from typing import Optional


def cpu_deep_copy_tuple(input_tuple):
    # 如果元组中的元素是张量，则在 CPU 上深拷贝，否则保持原样
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians_underwater(
    # 调用自定义 C++/CUDA 函数实现的高斯分布栅格化，封装调用自定义的CUDA 函数 _C.rasterize_gaussians_underwater对应c++RasterizeGaussiansCUDA_underwater
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    medium_rgb,
    medium_bs,
    medium_attn,
    raster_settings,
    colors_enhance
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        medium_rgb,
        medium_bs,
        medium_attn,
        raster_settings,
        colors_enhance
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        medium_rgb,
        medium_bs,
        medium_attn,
        raster_settings,
        colors_enhance,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, #1
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            medium_rgb,
            medium_bs,
            medium_attn,
            colors_enhance,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug #24
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color_image , color_clr, color_cem , radii, geomBuffer, binningBuffer, imgBuffer, depths = _C.rasterize_gaussians_underwater(*args)

        # Keep relevant tensors for backward 保存上下文以便反向传播
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp,  medium_rgb, medium_bs, medium_attn,
                              colors_enhance , radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color_image , color_clr, color_cem , radii, depths

    @staticmethod
    def backward(ctx, grad_color_image , grad_color_clr, grad_color_cem , _ , grad_depths):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, medium_rgb, medium_bs, medium_attn,  \
                              colors_enhance, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them 对应c++RasterizeGaussiansBackwardCUDA_underwater 
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                medium_rgb, 
                medium_bs, 
                medium_attn,  
                colors_enhance,
                grad_color_image,
                grad_color_clr,
                grad_color_cem,
                grad_depths, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_medium_rgb, grad_medium_bs,   \
                grad_medium_attn, grad_colors_enhance, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward_underwater(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_medium_rgb,
            grad_medium_bs,
            grad_medium_attn,
            None,
            grad_colors_enhance,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple): #封装高斯栅格化的渲染参数
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool
    # medium_rgb:torch.Tensor
    # medium_bs: torch.Tensor
    # medium_attn: torch.Tensor
    # colors_enhance: Optional[torch.Tensor]  # 表示可以为 None

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None,
                medium_rgb=None, medium_bs=None, medium_attn=None,colors_enhance =None):
        # forward函数：检查输入并调用核心渲染函数
        raster_settings = self.raster_settings# 获取光栅化设置

        # 检查输入的 shs 和 colors_precomp 必须且只能提供一个
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        # 检查输入的 scales 和 rotations 必须和 cov3D_precomp 一起使用或者两者都不使用
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # 如果 shs 为 None，则初始化为空 Tensor
        if shs is None:
            shs = torch.Tensor([])
        # 如果 colors_precomp 为 None，则初始化为空 Tensor
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        # 如果 scales 为 None，则初始化为空 Tensor
        if scales is None:
            scales = torch.Tensor([])
        # 如果 rotations 为 None，则初始化为空 Tensor
        if rotations is None:
            rotations = torch.Tensor([])
        # 如果 cov3D_precomp 为 None，则初始化为空 Tensor
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine

        if medium_rgb is None:
            medium_rgb = torch.zeros((raster_settings.image_height, raster_settings.image_width, 3),
                                     device=means3D.device)
        if medium_bs is None:
            medium_bs = torch.zeros_like(medium_rgb).to(means3D.device)
        if medium_attn is None:
            medium_attn = torch.zeros_like(medium_rgb).to(means3D.device)

        if colors_enhance is None:
            colors_enhance = torch.zeros_like(medium_rgb).to(means3D.device)
            print("Without colors enhance")

        return rasterize_gaussians_underwater(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            medium_rgb,
            medium_bs,
            medium_attn,
            raster_settings,
            colors_enhance
        )

