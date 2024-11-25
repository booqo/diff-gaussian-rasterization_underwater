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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn


def raw_loss(network_output, gt, mask=None):
    mse = (network_output - gt) ** 2
    scale = 1. / (1e-3 + network_output)
    scale.detach()
    if mask is None:
        return (mse * scale ** 2).mean()
    else:
        return ((mse * scale ** 2) * mask).mean()

def l1_loss(network_output, gt, mask=None):
    if mask is None:
        return torch.abs((network_output - gt)).mean()
    else:
        return torch.abs((network_output - gt)*mask).mean()

def l2_loss(network_output, gt, mask=None):
    if mask is None:
        return ((network_output - gt) ** 2).mean()
    else:
        return (((network_output - gt) * mask) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask=mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        ssim_map = ssim_map * mask

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# 定义损失函数类
# class BackscatterLoss(nn.Module):
#     def __init__(self, cost_ratio=1000.):
#         super().__init__()
#         self.l1 = nn.L1Loss()# 对正值部分使用L1损失
#         self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)# 对负值部分使用平滑L1损失
#         self.relu = nn.ReLU() # 用于取正值
#         self.cost_ratio = cost_ratio# 权重系数，用于调节负值损失的重要性
#
#     def forward(self, direct):
#         """
#         计算散射损失（Backscatter Loss）。
#
#         参数:
#         - direct: 输入图像的直接分量（通常是预测的直接光成分）
#
#         返回:
#         - bs_loss: 计算的散射损失值
#         """
#         pos = self.l1(self.relu(direct), torch.zeros_like(direct))# 计算正值损失，即 D^c(i,j) = max{D^c(i,j), 0}
#         neg = self.smooth_l1(self.relu(-direct), torch.zeros_like(direct))# 计算负值损失，即 min{D^c(i,j), 0} = -max{-D^c(i,j), 0}
#         bs_loss = self.cost_ratio * neg + pos# 综合正值和负值损失，负值损失加权
#         return bs_loss
#
# class DeattenuateLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 定义损失函数
#         self.mse = nn.MSELoss()  # 均方误差损失，用于计算空间变化损失
#         self.relu = nn.ReLU()  # ReLU激活函数，用于计算饱和损失
#         self.target_intensity = 0.5  # 目标强度值，用于控制通道强度的目标
#
#     def forward(self, direct, J):
#         """
#         计算去衰减网络的损失函数。
#
#         参数:
#         - direct: 输入的直接光分量
#         - J: 去衰减后的图像
#
#         返回:
#         - 总损失值，包括饱和损失、强度损失和空间变化损失
#         """
#
#         # 饱和损失：惩罚去衰减图像J的值超出[0, 1]范围
#         saturation_loss = (self.relu(-J) + self.relu(J - 1)).square().mean()
#
#         # 初始直接光分量的空间标准差
#         init_spatial = torch.std(direct, dim=[1, 2])
#
#         # 通道强度：计算去衰减图像J每个通道的平均强度
#         channel_intensities = torch.mean(J, dim=[1, 2], keepdim=True)
#
#         # 通道空间变化：计算去衰减图像J每个通道的标准差
#         channel_spatial = torch.std(J, dim=[1, 2])
#
#         # 强度损失：鼓励每个通道的强度接近目标强度值
#         intensity_loss = (channel_intensities - self.target_intensity).square().mean()
#
#         # 空间变化损失：惩罚去衰减图像的空间变化与直接光分量的空间变化差异
#         spatial_variation_loss = self.mse(channel_spatial, init_spatial)
#
#         # 检查是否有NaN值并打印警告
#         if torch.any(torch.isnan(saturation_loss)):
#             print("NaN saturation loss!")
#         if torch.any(torch.isnan(intensity_loss)):
#             print("NaN intensity loss!")
#         if torch.any(torch.isnan(spatial_variation_loss)):
#             print("NaN spatial variation loss!")
#
#         # 返回总损失，包含三部分
#         return saturation_loss + intensity_loss + spatial_variation_loss


class DeattenuateLoss(nn.Module):
    def __init__(self, target_intensity=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.target_intensity = target_intensity

    def forward(self, direct, J):
        # 裁剪 J 的值以确保它们在 [0, 1] 的范围内
        J = torch.clamp(J, 0, 1)

        # 饱和损失
        saturation_loss = (torch.relu(-J) + torch.relu(J - 1)).square().mean()

        # 初始直接光分量的空间标准差
        init_spatial = torch.std(direct, dim=[1, 2], keepdim=True)

        # 通道强度和空间变化
        channel_intensities = torch.mean(J, dim=[1, 2], keepdim=True)
        channel_spatial = torch.std(J, dim=[1, 2], keepdim=True)

        # 强度损失和空间变化损失
        intensity_loss = (channel_intensities - self.target_intensity).square().mean()
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)

        return saturation_loss + intensity_loss + spatial_variation_loss
class BackscatterLoss(nn.Module):
    def __init__(self, cost_ratio=1000.):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.cost_ratio = cost_ratio

    def forward(self, direct):
        pos = self.l1(torch.where(direct > 0, direct, torch.zeros_like(direct)), torch.zeros_like(direct))
        neg = self.l1(torch.where(direct < 0, -direct, torch.zeros_like(direct)), torch.zeros_like(direct))
        bs_loss = self.cost_ratio * neg + pos
        return bs_loss



def get_gray_loss(img):
    """
    计算图像的灰度损失 (Gray Loss)。
    这个损失函数主要通过像素间的差异来评估图像的一致性。

    参数：
        img: 输入的图像张量 (shape: [3, H, W])，其中通道顺序为 RGB。

    返回：
        loss: 计算得到的灰度损失值。
    """
    # mean_R = torch.mean(img[0, :, :])
    # mean_G = torch.mean(img[1, :, :])
    # mean_B = torch.mean(img[2, :, :])
    #
    # # 计算Gray World Prior Loss
    # loss = (mean_R - mean_G) ** 2 + (mean_G - mean_B) ** 2 + (mean_B - mean_R) ** 2

    gray_loss_clip = 0.05  # 用于限制灰度损失的最大值。

    w1 = 1 # 权重1，用于控制损失的缩放
    w2 = img.var(axis=0, keepdims=True) + 0.5  # 计算图像在各位置的方差。
    # 计算相邻像素之间的平方差（通过在第一个维度上滚动来实现）。
    diffs = (img - torch.roll(img, 1, dims=0)) ** 2

    # if gray_loss_clip > 0:
    #     diffs = torch.minimum(gray_loss_clip, diffs)

    '''
        tensor([[0.9768, 0.9784, 0.9819],
            [0.9706, 0.9718, 0.9789],
            [0.9737, 0.9710, 0.9796],
            ...,
            [0.9750, 0.9751, 0.9835],
            [0.9631, 0.9638, 0.9691],
            [0.9728, 0.9730, 0.9833]], device='cuda:0',
           grad_fn= < ClampMinBackward0 >)
    '''
    # 计算损失：对差异进行平方和、标准化并取均值。
    loss = torch.sqrt(diffs.sum(axis=0, keepdims=True) / w2 / 3).mean()

    return loss

def grad(img):
    """
    计算图像的梯度（包括水平和垂直方向）。

    参数：
        img: 输入图像张量 (shape: [3, H, W])。

    返回：
        grad: 图像的梯度张量 (shape: [1, 3, H, W])。
    """
    # 定义 Sobel 滤波器，用于计算图像的梯度
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32)[None, None, :].expand(1, 3, -1, -1).cuda()  # 扩展为 1x3 通道并移至 GPU。
    # Sobel Y 滤波器用于检测垂直边缘。
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=torch.float32)[None, None, :].expand(1, 3, -1, -1).cuda()  # 扩展为 1x3 通道并移至 GPU。
    # 使用 Sobel X 和 Sobel Y 滤波器计算水平和垂直梯度。
    grad_x = F.conv2d(img[None, :], sobel_x)# 计算水平梯度。
    grad_y = F.conv2d(img[None, :], sobel_y)# 及时垂直梯度
    # 计算总体梯度（使用勾股定理求平方和的平方根）。
    grad = torch.sqrt(grad_x**2 + grad_y**2)
    return grad

def get_grad_loss(img1, img2):
    """
    计算两个图像之间的梯度损失。

    参数：
        img1: 第一个输入图像张量 (shape: [3, H, W])。
        img2: 第二个输入图像张量 (shape: [3, H, W])。

    返回：
        梯度损失值（基于 SSIM 计算）。
    """
    g1 = grad(img1)
    g2 = grad(img2)
    # 根据两个图像的平均值计算缩放系数，使它们的亮度相匹配。
    scale = img1.mean() / img2.mean()
    g1 = g1# 第一个图像的梯度（无需缩放）。
    g2 = g2 * scale# 第二个图像的梯度按比例缩放。
    lambda_dssim = 0.2# DSSIM 损失的权重参数。
    # 计算基于 SSIM 的梯度损失。这里注释掉了 L1 损失的部分。
    # return ((1.0 - lambda_dssim) * l1_loss(g1, g2)  + lambda_dssim * (1.0 - ssim(g1, g2)))
    return 1.0 - ssim(g1, g2)


