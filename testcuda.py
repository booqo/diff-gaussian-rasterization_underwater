# 导入必要的库
from diff_gaussian_rasterization_underwater import GaussianRasterizationSettings, GaussianRasterizer
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

# 定义从世界坐标系到视图坐标系的转换函数
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
    # 初始化4x4矩阵
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()  # 转置旋转矩阵
    Rt[:3, 3] = t               # 设置平移向量
    Rt[3, 3] = 1.0              # 设置齐次坐标

    # 计算相机到世界的逆矩阵
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]     # 获取相机中心
    cam_center = (cam_center + translate) * scale  # 应用额外的平移和缩放
    C2W[:3, 3] = cam_center     # 更新相机中心
    Rt = np.linalg.inv(C2W)      # 再次求逆，得到世界到相机的矩阵
    return np.float32(Rt)        # 返回32位浮点数类型的矩阵

def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """
    生成从世界坐标系到视图坐标系的转换矩阵。
    """
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = -np.dot(R.T, t)
    return Rt

# 定义投影矩阵生成函数
def getProjectionMatrix2(znear, zfar, fovX, fovY):
    """
    生成投影矩阵。

    参数:
        znear (float): 近裁剪面距离。
        zfar (float): 远裁剪面距离。
        fovX (float): 水平视场角（度）。
        fovY (float): 垂直视场角（度）。

    返回:
        np.ndarray: 投影矩阵（4x4）。
    """
    # 将视场角从度转换为弧度，并计算半角的正切值
    tanHalfFovY = math.tan(math.radians(fovY) / 2)
    tanHalfFovX = math.tan(math.radians(fovX) / 2)

    # 计算裁剪平面的边界
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # 初始化投影矩阵
    P = np.zeros((4, 4))

    # 确定z轴的符号（通常为1）
    z_sign = 1.0

    # 填充投影矩阵的各个元素
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = z_sign

    return P

def getProjectionMatrix(znear, zfar, fovX, fovY):
    """
    生成标准的透视投影矩阵。
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




# 定义高斯体的中心点和特征
pts = np.array([
    [0, 0, 100],    # 位于正Z轴方向
    [-0.5, -0, 100],
    [-0, 10, 100]
], dtype=np.float32)
n = len(pts)  # 高斯体数量


# np.random.seed(100)
# shs = np.random.random((n, 16, 3))

# 设置固定颜色（进一步调整以避免超出范围）
desired_colors = np.array([
    [0.4, 0.0, 0.0],   # 高斯体 1: 红色
    [0.0, 0.4, 0.0],   # 高斯体 2: 绿色
    [0.0, 0.0, 0.4]    # 高斯体 3: 蓝色
], dtype=np.float32)

# 初始化球谐函数系数为全零，并设置零阶系数为所需颜色
shs = np.zeros((n, 16, 3), dtype=np.float32)
shs[:, 0, :] = desired_colors  # 设置零阶系数


# 设置其他高斯体属性
opacities = np.ones((n, 1), dtype=np.float32)          # 透明度设为1
scales = np.ones((n, 3), dtype=np.float32) # 增大尺度以确保可见
rotations = np.tile(np.eye(3), (n, 1, 1)).astype(np.float32)  # 旋转矩阵设为单位矩阵

# 将NumPy数组转换为PyTorch张量，并移动到GPU
pts = torch.tensor(pts, dtype=torch.float32, device="cuda")
shs = torch.tensor(shs, dtype=torch.float32, device="cuda")
opacities = torch.tensor(opacities, dtype=torch.float32, device="cuda")
scales = torch.tensor(scales, dtype=torch.float32, device="cuda")
rotations = torch.tensor(rotations, dtype=torch.float32, device="cuda")

# 定义相机的位置和朝向
cam_pos = np.array([0, 0, 0], dtype=np.float32)
R = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=np.float32)  # 相机朝向正Z轴

# 生成视图矩阵和投影矩阵
viewmatrix_np = getWorld2View2(R=R, t=cam_pos)
viewmatrix = torch.tensor(viewmatrix_np, dtype=torch.float32, device="cuda")

# 定义投影参数
proj_param = {
    "znear": 0.01,
    "zfar": 100,
    "fovX": 90,  # 水平视场角（度）
    "fovY": 90   # 垂直视场角（度）
}

# 生成世界到视图的转换矩阵，并转换为PyTorch张量
projmatrix_np = getProjectionMatrix(**proj_param)
projmatrix = torch.tensor(projmatrix_np, dtype=torch.float32, device="cuda")

# 生成投影矩阵，并转换为PyTorch张量
projmatrix_np = getProjectionMatrix(**proj_param)
projmatrix = torch.tensor(projmatrix_np, dtype=torch.float32, device="cuda")

# 计算最终的投影矩阵（投影矩阵 * 视图矩阵）
projmatrix = torch.matmul(projmatrix, viewmatrix)

cam_pos = torch.tensor(cam_pos, dtype=torch.float32, device="cuda")

# 定义介质参数（默认为0）
c_med = torch.tensor([0.0], dtype=torch.float32, device="cuda")
sigma_bs = torch.tensor([0.0], dtype=torch.float32, device="cuda")
sigma_atten = torch.tensor([0.0], dtype=torch.float32, device="cuda")

# 定义背景颜色为黑色
bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

# 计算屏幕空间坐标
n_pts = pts.shape[0]
pts_h = torch.cat([pts, torch.ones(n_pts, 1, dtype=torch.float32, device="cuda")], dim=1)  # [n, 4]
screenspace_points_h = torch.matmul(pts_h, projmatrix.T)  # [n, 4]
screenspace_points = screenspace_points_h[:, :3] / screenspace_points_h[:, 3:4]  # 透视除法

# 配置高斯光栅化设置
raster_settings = GaussianRasterizationSettings(
    image_height=200,                       # 输出图像高度
    image_width=200,                        # 输出图像宽度
    tanfovx=math.tan(math.radians(proj_param["fovX"] / 2)),  # 水平视场角的正切
    tanfovy=math.tan(math.radians(proj_param["fovY"] / 2)),  # 垂直视场角的正切
    bg=bg,                                  # 背景颜色
    scale_modifier=1.0,                     # 缩放因子
    viewmatrix=viewmatrix,                  # 视图矩阵
    projmatrix=projmatrix,                  # 投影矩阵
    sh_degree=1,                            # 球谐次数
    campos=cam_pos,                         # 相机位置
    prefiltered=False,                      # 是否使用预滤波
    debug=False,                            # 是否启用调试模式
    medium_rgb=c_med,                       # 介质颜色
    medium_bs=sigma_bs,                     # 介质散射系数
    medium_attn=sigma_atten,                # 介质衰减系数
    antialiasing=None                      # 抗锯齿设置
)

# 初始化高斯光栅化器
rasterizer = GaussianRasterizer(raster_settings=raster_settings)

# 执行高斯光栅化
rendered_image, radii, _ = rasterizer(
    means3D=pts,               # 高斯体的3D位置
    means2D=screenspace_points,  # 高斯体在屏幕空间的2D位置
    shs=shs,                   # 球谐系数
    colors_precomp=None,       # 预计算颜色（如果有）
    opacities=opacities,       # 透明度
    scales=scales,             # 缩放因子
    rotations=rotations,       # 旋转矩阵
    cov3D_precomp=None,        # 预计算协方差（如果有）
    medium_rgb=c_med,          # 介质颜色
    medium_bs=sigma_bs,        # 介质散射系数
    medium_attn=sigma_atten    # 介质衰减系数
)


rendered_image = rendered_image.permute(1, 2, 0)  # 从[3, H, W]转换为[H, W, 3]

# 将渲染图像从GPU移动到CPU，并分离梯度，转换为NumPy数组
rendered_image_np = rendered_image.detach().cpu().numpy()

# 使用Matplotlib显示渲染结果
plt.imshow(rendered_image_np)
plt.axis('off')  # 可选，去掉坐标轴
plt.show()


