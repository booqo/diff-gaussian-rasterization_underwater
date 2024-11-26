# 导入必要的库
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from diff_gaussian_rasterization_underwater import GaussianRasterizationSettings, GaussianRasterizer


# ============================================
# 工具函数定义
# ============================================

def getWorld2View(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """
    生成从世界坐标系到视图坐标系的转换矩阵。

    参数:
        R (np.ndarray): 旋转矩阵（3x3）。
        t (np.ndarray): 平移向量（3,）。
        translate (np.ndarray): 额外的平移向量，用于调整相机位置（默认 [0, 0, 0]）。
        scale (float): 缩放因子（默认 1.0）。

    返回:
        np.ndarray: 转换矩阵（4x4）。
    """
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # 计算相机到世界的逆矩阵
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]  # 获取相机中心
    cam_center = (cam_center + translate) * scale  # 应用额外的平移和缩放
    C2W[:3, 3] = cam_center  # 更新相机中心
    Rt = np.linalg.inv(C2W)  # 再次求逆，得到世界到相机的矩阵
    return np.float32(Rt)  # 返回32位浮点数类型的矩阵


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """
    生成从世界坐标系到视图坐标系的转换矩阵（简化版本）。
    """
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = -np.dot(R.T, t)
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """
    生成标准的透视投影矩阵。

    参数:
        znear (float): 近裁剪面距离。
        zfar (float): 远裁剪面距离。
        fovX (float): 水平视场角（度）。
        fovY (float): 垂直视场角（度）。

    返回:
        np.ndarray: 投影矩阵（4x4）。
    """
    aspect = math.tan(math.radians(fovX) / 2) / math.tan(math.radians(fovY) / 2)
    f = 1.0 / math.tan(math.radians(fovY) / 2)
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (zfar + znear) / (znear - zfar)
    P[2, 3] = (2 * zfar * znear) / (znear - zfar)
    P[3, 2] = -1.0
    return P


def initialize_gaussian_bodies(pts_np, desired_colors_np):
    """
    初始化高斯体的中心点、颜色和球谐函数系数。

    参数:
        pts_np (np.ndarray): 高斯体的3D位置数组，形状为 [n, 3]。
        desired_colors_np (np.ndarray): 高斯体的颜色数组，形状为 [n, 3]。

    返回:
        tuple: 包含转换为PyTorch张量的 pts, shs, opacities, scales, rotations
    """
    n = len(pts_np)
    shs_np = np.zeros((n, 16, 3), dtype=np.float32)
    shs_np[:, 0, :] = desired_colors_np  # 设置零阶系数

    opacities_np = np.ones((n, 1), dtype=np.float32)  # 透明度设为1
    scales_np = np.ones((n, 3), dtype=np.float32)  # 尺度设为1
    rotations_np = np.tile(np.eye(3), (n, 1, 1)).astype(np.float32)  # 旋转矩阵设为单位矩阵

    # 转换为PyTorch张量并移动到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts = torch.tensor(pts_np, dtype=torch.float32, device=device)
    shs = torch.tensor(shs_np, dtype=torch.float32, device=device)
    opacities = torch.tensor(opacities_np, dtype=torch.float32, device=device)
    scales = torch.tensor(scales_np, dtype=torch.float32, device=device)
    rotations = torch.tensor(rotations_np, dtype=torch.float32, device=device)

    return pts, shs, opacities, scales, rotations


def initialize_camera(cam_pos_np, R_np):
    """
    初始化相机的位置和视图矩阵。

    参数:
        cam_pos_np (np.ndarray): 相机的位置，形状为 [3,]。
        R_np (np.ndarray): 相机的旋转矩阵，形状为 [3, 3]。

    返回:
        torch.Tensor: 视图矩阵的PyTorch张量，形状为 [4, 4]。
    """
    viewmatrix_np = getWorld2View2(R=R_np, t=cam_pos_np)
    viewmatrix = torch.tensor(viewmatrix_np, dtype=torch.float32, device="cuda")
    return viewmatrix


def initialize_projection(proj_param):
    """
    初始化投影矩阵和计算tan(fovX/2), tan(fovY/2)。

    参数:
        proj_param (dict): 包含投影参数的字典，键包括 "znear", "zfar", "fovX", "fovY"。

    返回:
        tuple: 包含投影矩阵的PyTorch张量, tanfovx, tanfovy
    """
    projmatrix_np = getProjectionMatrix(**proj_param)
    projmatrix = torch.tensor(projmatrix_np, dtype=torch.float32, device="cuda")
    tanfovx = math.tan(math.radians(proj_param["fovX"] / 2))
    tanfovy = math.tan(math.radians(proj_param["fovY"] / 2))
    return projmatrix, tanfovx, tanfovy


def initialize_medium(H, W, device):
    """
    初始化介质参数。

    参数:
        H (int): 图像高度。
        W (int): 图像宽度。
        device (torch.device): 设备（CPU或CUDA）。

    返回:
        tuple: 包含c_med, sigma_bs, sigma_atten, colors_enhance的张量。
    """
    c_med = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    c_med[:, :, 2] = 130  # 蓝色通道设为130
    sigma_bs = torch.full((H, W, 3), 1, dtype=torch.float32, device=device)
    sigma_atten = torch.full((H, W, 3), 1, dtype=torch.float32, device=device)
    colors_enhance = torch.full((H, W, 3), 2, dtype=torch.float32, device=device)
    return c_med, sigma_bs, sigma_atten, colors_enhance


def initialize_background(device):
    """
    初始化背景颜色。

    参数:
        device (torch.device): 设备（CPU或CUDA）。

    返回:
        torch.Tensor: 背景颜色张量，形状为 [3,]。
    """
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    return bg


def compute_screen_space_points(pts, projmatrix):
    """
    计算屏幕空间坐标。

    参数:
        pts (torch.Tensor): 高斯体的3D位置张量，形状为 [n, 3]。
        projmatrix (torch.Tensor): 投影矩阵，形状为 [4, 4]。

    返回:
        torch.Tensor: 屏幕空间的2D坐标，形状为 [n, 3]。
    """
    n_pts = pts.shape[0]
    pts_h = torch.cat([pts, torch.ones(n_pts, 1, dtype=torch.float32, device=pts.device)], dim=1)  # [n, 4]
    screenspace_points_h = torch.matmul(pts_h, projmatrix.T)  # [n, 4]
    screenspace_points = screenspace_points_h[:, :3] / screenspace_points_h[:, 3:4]  # 透视除法
    return screenspace_points


def setup_gaussian_rasterizer(H, W, tanfovx, tanfovy, viewmatrix, projmatrix, bg, cam_pos):
    """
    配置和初始化高斯光栅化器。

    参数:
        H (int): 图像高度。
        W (int): 图像宽度。
        tanfovx (float): 水平视场角的正切。
        tanfovy (float): 垂直视场角的正切。
        viewmatrix (torch.Tensor): 视图矩阵，形状为 [4, 4]。
        projmatrix (torch.Tensor): 投影矩阵，形状为 [4, 4]。
        bg (torch.Tensor): 背景颜色，形状为 [3,]。
        cam_pos (torch.Tensor): 相机位置，形状为 [3,]。

    返回:
        GaussianRasterizer: 配置好的高斯光栅化器对象。
    """
    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=cam_pos,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    return rasterizer


# ============================================
# 主函数
# ============================================

def initialize_rasterizer(H, W, pts_np, desired_colors_np, cam_pos_np, R_np, proj_param):
    """
    初始化所有参数并配置高斯光栅化器。

    参数:
        H (int): 图像高度。
        W (int): 图像宽度。
        pts_np (np.ndarray): 高斯体的3D位置数组，形状为 [n, 3]。
        desired_colors_np (np.ndarray): 高斯体的颜色数组，形状为 [n, 3]。
        cam_pos_np (np.ndarray): 相机的位置，形状为 [3,]。
        R_np (np.ndarray): 相机的旋转矩阵，形状为 [3, 3]。
        proj_param (dict): 包含投影参数的字典，键包括 "znear", "zfar", "fovX", "fovY"。

    返回:
        tuple: 包含 rasterizer, screenspace_points, c_med, sigma_bs, sigma_atten, colors_enhance, bg, pts, shs, opacities, scales, rotations
    """
    # 初始化高斯体
    pts, shs, opacities, scales, rotations = initialize_gaussian_bodies(pts_np, desired_colors_np)

    # 初始化相机
    viewmatrix = initialize_camera(cam_pos_np, R_np)

    # 初始化投影矩阵
    projmatrix, tanfovx, tanfovy = initialize_projection(proj_param)

    # 计算最终的投影矩阵（投影矩阵 * 视图矩阵）
    projmatrix = torch.matmul(projmatrix, viewmatrix)

    # 初始化介质参数
    c_med, sigma_bs, sigma_atten, colors_enhance = initialize_medium(H, W, pts.device)

    # 初始化背景颜色
    bg = initialize_background(pts.device)

    # 计算屏幕空间坐标
    screenspace_points = compute_screen_space_points(pts, projmatrix)

    # 转换相机位置为PyTorch张量
    cam_pos = torch.tensor(cam_pos_np, dtype=torch.float32, device=pts.device)

    # 配置和初始化高斯光栅化器
    rasterizer = setup_gaussian_rasterizer(H, W, tanfovx, tanfovy, viewmatrix, projmatrix, bg, cam_pos)

    return rasterizer, screenspace_points, c_med, sigma_bs, sigma_atten, colors_enhance, bg, pts, shs, opacities, scales, rotations


def main():
    # --------------------------------------------
    # 1. 定义高斯体的中心点和特征
    # --------------------------------------------
    pts_np = np.array([
        [0, 0, 10],  # 位于正Z轴方向
        # [-0.2, -0, 10],
        # [-0, 0.2, 10]
    ], dtype=np.float32)
    desired_colors_np = np.array([
        [100, 0.0, 0.0],  # 高斯体 1: 红色
        # [0.0, 100, 0.0],   # 高斯体 2: 绿色
        # [0.0, 0.0, 100]    # 高斯体 3: 蓝色
    ], dtype=np.float32)

    # --------------------------------------------
    # 2. 定义相机的位置和朝向
    # --------------------------------------------
    cam_pos_np = np.array([0, 0, 0], dtype=np.float32)
    R_np = np.eye(3, dtype=np.float32)  # 相机朝向正Z轴

    # --------------------------------------------
    # 3. 设置图像的宽度和高度
    # --------------------------------------------
    H = 2
    W = 2

    # --------------------------------------------
    # 4. 定义投影参数
    # --------------------------------------------
    proj_param = {
        "znear": 0.01,
        "zfar": 100,
        "fovX": 90,  # 水平视场角（度）
        "fovY": 90  # 垂直视场角（度）
    }

    # --------------------------------------------
    # 5. 初始化高斯光栅化器和相关参数
    # --------------------------------------------
    rasterizer, screenspace_points, c_med, sigma_bs, sigma_atten, colors_enhance, bg, pts, shs, opacities, scales, rotations = initialize_rasterizer(
        H, W, pts_np, desired_colors_np, cam_pos_np, R_np, proj_param
    )

    # --------------------------------------------
    # 6. 执行高斯光栅化
    # --------------------------------------------
    color_attn, color_clr, color_medium, radii, depths = rasterizer(
        means3D=pts,  # 高斯体的3D位置
        means2D=screenspace_points,  # 高斯体在屏幕空间的2D位置
        shs=shs,  # 球谐颜色
        colors_precomp=None,  # 预计算颜色（如果有）
        opacities=opacities,  # 透明度
        scales=scales,  # 缩放因子
        rotations=rotations,  # 旋转矩阵
        cov3D_precomp=None,  # 预计算协方差（如果有）
        medium_rgb=c_med,  # 介质颜色
        medium_bs=sigma_bs,  # 介质散射系数
        medium_attn=sigma_atten,  # 介质衰减系数
        colors_enhance=colors_enhance  # 增强的颜色
    )

    # --------------------------------------------
    # 7. 将输出从 [C, H, W] 转换为 [H, W, C]，以便于显示
    # --------------------------------------------
    color_attn_np = color_attn.permute(1, 2, 0).detach().cpu().numpy()
    color_clr_np = color_clr.permute(1, 2, 0).detach().cpu().numpy()
    color_medium_np = color_medium.permute(1, 2, 0).detach().cpu().numpy()
    depths_np = depths.permute(1, 2, 0).detach().cpu().numpy()
    rendered_image_np = color_attn_np + color_medium_np

    # --------------------------------------------
    # 8. 使用 Matplotlib 显示渲染结果
    # --------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    axes[0].imshow(rendered_image_np)
    axes[0].axis('off')
    axes[0].set_title("Rendered Image")

    axes[1].imshow(color_clr_np)
    axes[1].axis('off')
    axes[1].set_title("Color Clr")

    axes[2].imshow(color_medium_np)
    axes[2].axis('off')
    axes[2].set_title("Color Medium")

    axes[3].imshow(color_attn_np)
    axes[3].axis('off')
    axes[3].set_title("Color Attn")

    axes[4].imshow(depths_np)
    axes[4].axis('off')
    axes[4].set_title("Depth Map")

    plt.show()


if __name__ == "__main__":
    main()
