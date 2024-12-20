# 导入必要的库
from diff_gaussian_rasterization_underwater import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings as GaussianRasterizationSettings2
from diff_gaussian_rasterization import GaussianRasterizer as GaussianRasterizer2

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
#from testcuda_forward import getWorld2View2,getProjectionMatrix

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


def gradient_check(variable_name, variable, rasterizer, loss_fn, **kwargs):
    """
    检查指定变量的梯度是否正确。

    Args:
        variable_name (str): 变量的名称，用于打印信息。
        variable (torch.Tensor): 需要检查梯度的变量，要求requires_grad=True。
        rasterizer (GaussianRasterizer): 高斯光栅化器对象。
        loss_fn (callable): 损失函数，接受光栅化输出并返回标量损失。
        kwargs: 传递给rasterizer的其他参数。
    """
    epsilon = 1e-4
    # 确保变量需要计算梯度
    variable.requires_grad = True
    print("gradient check for ", variable_name)
    # 计算自动求导的梯度
    rasterizer.zero_grad()  # 清零之前的梯度
    output = rasterizer(
        means3D=kwargs.get('means3D'),
        means2D=kwargs.get('means2D'),
        shs=kwargs.get('shs'),
        colors_precomp=kwargs.get('colors_precomp'),
        opacities=kwargs.get('opacities'),
        scales=kwargs.get('scales'),
        rotations=kwargs.get('rotations'),
        cov3D_precomp=kwargs.get('cov3D_precomp'),
        medium_rgb=kwargs.get('medium_rgb'),
        medium_bs=kwargs.get('medium_bs'),
        medium_attn=kwargs.get('medium_attn'),
        colors_enhance=kwargs.get('colors_enhance')
    )
    loss = loss_fn(output)
    loss.backward()
    autograd_grad = variable.grad.clone()

    # 数值梯度初始化
    numerical_grad = torch.zeros_like(variable)

    # 使用 flatten 方法进行一维化遍历
    variable_flat = variable.view(-1)                       # 展平的变量
    numerical_grad_flat = numerical_grad.view(-1) # 展平的数值梯度量
    autograd_grad_flat = autograd_grad.view(-1)       #展平的自动梯度

    for idx in range(variable_flat.numel()):
        original_value = variable_flat[idx].item()

        # # 创建变量的副本以避免原地操作
        with torch.no_grad():
            variable_flat[idx] = original_value + epsilon
            render_img_max, _,render_cem_max,_,_,_ = rasterizer(
                means3D=kwargs.get('means3D'),
                means2D=kwargs.get('means2D'),
                shs=kwargs.get('shs'),
                colors_precomp=kwargs.get('colors_precomp'),
                opacities=kwargs.get('opacities'),
                scales=kwargs.get('scales'),
                rotations=kwargs.get('rotations'),
                cov3D_precomp=kwargs.get('cov3D_precomp'),
                medium_rgb=kwargs.get('medium_rgb'),
                medium_bs=kwargs.get('medium_bs'),
                medium_attn=kwargs.get('medium_attn'),
                colors_enhance=kwargs.get('colors_enhance')
            )
            loss_plus = render_img_max.sum()
            #print(original_value)
            #print(render_img_max)
            variable_flat[idx] = original_value - epsilon

            render_img_min, _,render_cem_min,_,_,_ = rasterizer(
                means3D=kwargs.get('means3D'),
                means2D=kwargs.get('means2D'),
                shs=kwargs.get('shs'),
                colors_precomp=kwargs.get('colors_precomp'),
                opacities=kwargs.get('opacities'),
                scales=kwargs.get('scales'),
                rotations=kwargs.get('rotations'),
                cov3D_precomp=kwargs.get('cov3D_precomp'),
                medium_rgb=kwargs.get('medium_rgb'),
                medium_bs=kwargs.get('medium_bs'),
                medium_attn=kwargs.get('medium_attn'),
                colors_enhance=kwargs.get('colors_enhance')
            )
            loss_minus = render_img_min.sum()
            #print(render_img_min)
            variable_flat[idx] = original_value

    #     # 计算数值梯度
            # print("loss plus is " , loss_plus)
            # print("loss minus is " , loss_minus)
            numerical_grad_flat[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        # print("numerical_grad\n", numerical_grad_flat[idx])
        # print("autograd_grad\n", autograd_grad_flat[idx])

    # 重新形状还原
    
    
    numerical_grad = numerical_grad_flat.view_as(variable)
    autograd_grad = autograd_grad_flat.view_as(variable)
    print("numerical_grad\n", numerical_grad.detach().cpu().numpy())
    print("autograd_grad\n", autograd_grad.detach().cpu().numpy())



    # # 计算相对误差
    # relative_error = (autograd_grad - numerical_grad).abs() #/ (numerical_grad.abs() + 1e-8)
    # max_error = relative_error.max().item()
    # mean_error = relative_error.mean().item()
    #
    # print(f"梯度检查 - {variable_name}: 最大相对误差 = {max_error:.6f}, 平均相对误差 = {mean_error:.6f}")
    #
    # if max_error < 1e-3:
    #     print(f"{variable_name} 的梯度实现可能是正确的。")
    # else:
    #     print(f"{variable_name} 的梯度实现可能存在问题，请检查。")


def gradient_check_pix(variable_name, variable, rasterizer, loss_fn, **kwargs):
    """
    检查指定变量的梯度是否正确。

    Args:
        variable_name (str): 变量的名称，用于打印信息。
        variable (torch.Tensor): 需要检查梯度的变量，要求requires_grad=True。
        rasterizer (GaussianRasterizer): 高斯光栅化器对象。
        loss_fn (callable): 损失函数，接受光栅化输出并返回标量损失。
        kwargs: 传递给rasterizer的其他参数。
    """
    idx = 0
    epsilon = 1e-3
    # 确保变量需要计算梯度
    variable.requires_grad = True
    print("gradient check for ", variable_name)
    # 计算自动求导的梯度
    rasterizer.zero_grad()  # 清零之前的梯度
    render_img, _,render_cem,_,_ = rasterizer(
        means3D=kwargs.get('means3D'),
        means2D=kwargs.get('means2D'),
        shs=kwargs.get('shs'),
        colors_precomp=kwargs.get('colors_precomp'),
        opacities=kwargs.get('opacities'),
        scales=kwargs.get('scales'),
        rotations=kwargs.get('rotations'),
        cov3D_precomp=kwargs.get('cov3D_precomp'),
        medium_rgb=kwargs.get('medium_rgb'),
        medium_bs=kwargs.get('medium_bs'),
        medium_attn=kwargs.get('medium_attn'),
        colors_enhance=kwargs.get('colors_enhance')
    )
    loss = (render_cem).sum()
    print(render_cem)
    loss.backward()
    autograd_grad = variable.grad.clone()

    # 数值梯度初始化
    numerical_grad = torch.zeros_like(variable)
    numerical_grad2 = torch.zeros_like(variable)

    # 使用 flatten 方法进行一维化遍历
    variable_flat = variable.view(-1)                       # 展平的变量
    numerical_grad_flat = numerical_grad.view(-1) # 展平的数值梯度量
    numerical_grad2_flat = numerical_grad2.view(-1) # 展平的数值梯度量
    autograd_grad_flat = autograd_grad.view(-1)       #展平的自动梯度

    
    original_value = variable_flat[idx].item()

    # # 创建变量的副本以避免原地操作
    with torch.no_grad():
        variable_flat[idx] = original_value + epsilon
        render_img_max, _,render_cem_max,_,_ = rasterizer(
            means3D=kwargs.get('means3D'),
            means2D=kwargs.get('means2D'),
            shs=kwargs.get('shs'),
            colors_precomp=kwargs.get('colors_precomp'),
            opacities=kwargs.get('opacities'),
            scales=kwargs.get('scales'),
            rotations=kwargs.get('rotations'),
            cov3D_precomp=kwargs.get('cov3D_precomp'),
            medium_rgb=kwargs.get('medium_rgb'),
            medium_bs=kwargs.get('medium_bs'),
            medium_attn=kwargs.get('medium_attn'),
            colors_enhance=kwargs.get('colors_enhance')
        )
        #loss_plus = loss_fn(output_plus).item()
        loss_single_max = (render_cem_max).sum()
        loss_first_max = render_cem_max[0,0,0]
        print(render_cem_max , loss_single_max , loss_first_max)
        #print(original_value)
        variable_flat[idx] = original_value - epsilon

        render_img_min, _,render_cem_min,_,_ = rasterizer(
            means3D=kwargs.get('means3D'),
            means2D=kwargs.get('means2D'),
            shs=kwargs.get('shs'),
            colors_precomp=kwargs.get('colors_precomp'),
            opacities=kwargs.get('opacities'),
            scales=kwargs.get('scales'),
            rotations=kwargs.get('rotations'),
            cov3D_precomp=kwargs.get('cov3D_precomp'),
            medium_rgb=kwargs.get('medium_rgb'),
            medium_bs=kwargs.get('medium_bs'),
            medium_attn=kwargs.get('medium_attn'),
            colors_enhance=kwargs.get('colors_enhance')
        )
        loss_single_min = (render_cem_min ).sum()
        loss_first_min = render_cem_min[0,0,0]
        print(render_cem_min , loss_single_min , loss_first_min)
        variable_flat[idx] = original_value

#     # 计算数值梯度
        # print("loss plus is " , loss_plus)
        # print("loss minus is " , loss_minus)
        numerical_grad_flat[idx] = (loss_single_max - loss_single_min) / (2 * epsilon)
        numerical_grad2_flat[idx] = (loss_first_max - loss_first_min) / (2 * epsilon)
    # print("numerical_grad\n", numerical_grad_flat[idx])
    # print("autograd_grad\n", autograd_grad_flat[idx])

    # 重新形状还原
    
    
    numerical_grad = numerical_grad_flat.view_as(variable)
    autograd_grad = autograd_grad_flat.view_as(variable)
    print("numerical_grad\n", numerical_grad.detach().cpu().numpy())
    print("numerical_grad2\n", numerical_grad2.detach().cpu().numpy())
    print("autograd_grad\n", autograd_grad.detach().cpu().numpy())
    



    # # 计算相对误差
    # relative_error = (autograd_grad - numerical_grad).abs() #/ (numerical_grad.abs() + 1e-8)
    # max_error = relative_error.max().item()
    # mean_error = relative_error.mean().item()
    #
    # print(f"梯度检查 - {variable_name}: 最大相对误差 = {max_error:.6f}, 平均相对误差 = {mean_error:.6f}")
    #
    # if max_error < 1e-3:
    #     print(f"{variable_name} 的梯度实现可能是正确的。")
    # else:
    #     print(f"{variable_name} 的梯度实现可能存在问题，请检查。")


# 定义一个简单的损失函数
def loss_function(outputs):
    color_attn, color_clr, color_medium, radii, depths,_ = outputs
    return (color_attn+color_medium).sum()
def loss_function2(outputs):
    rendered_image, radii, depth_image  = outputs
    return (rendered_image).sum()


# 定义高斯体的中心点和特征
pts = np.array([
    [0, 0, 10.0],    # 位于正Z轴方向
    [-0.2, -0, 10.0],
    [-0.2, -0, 5.0],
    [-0.2, -0, 3.0],
    [-0, 0.2, 10.0],
    [-0, 0.2, 5.0],
    [-0, 0.2, 3.0]
], dtype=np.float32)
n = len(pts)  # 高斯体数量

# 设置固定颜色（进一步调整以避免超出范围）
desired_colors = np.array([
    [100, 0.0, 0.0],   # 高斯体 1: 红色
    [0.0, 100, 0.0],   # 高斯体 2: 绿色
    [0.0, 0.0, 100],    # 高斯体 3: 蓝色
     [0.0, 0.0, 100] ,
      [0.0, 50.0, 100] ,
       [30.0, 0.0, 100] ,
    [30.0, 0.0, 100] ,
], dtype=np.float32)

# 初始化球谐函数系数为全零，并设置零阶系数为所需颜色(固定颜色)
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
H = 4
W = 4


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
tanfovx = math.tan(math.radians(proj_param["fovX"] / 2))  # 水平视场角的正切
tanfovy = math.tan(math.radians(proj_param["fovY"] / 2))  # 垂直视场角的正切


# 生成世界到视图的转换矩阵，并转换为PyTorch张量
projmatrix_np = getProjectionMatrix(**proj_param)
projmatrix = torch.tensor(projmatrix_np, dtype=torch.float32, device="cuda")

# 生成投影矩阵，并转换为PyTorch张量
projmatrix_np = getProjectionMatrix(**proj_param)
projmatrix = torch.tensor(projmatrix_np, dtype=torch.float32, device="cuda")

# 计算最终的投影矩阵（投影矩阵 * 视图矩阵）
projmatrix = torch.matmul(projmatrix, viewmatrix)

cam_pos = torch.tensor(cam_pos, dtype=torch.float32, device="cuda")

# 定义介质参数，形状应该是 H, W, 3
c_med = torch.zeros((H, W, 3), dtype=torch.float32, device="cuda")
c_med[:, :, 0] = 1.0 # 红色通道
c_med[:, :, 1] = 1.0  # 绿色通道
c_med[:, :, 2] = 1.0  # 蓝色通道
sigma_bs = torch.full((H, W, 3), 0.3 , dtype=torch.float32, device="cuda")
sigma_atten = torch.full((H, W, 3), 0.3 , dtype=torch.float32, device="cuda")

# 定义增强颜色参数，形状应该是 H, W, 3
colors_enhance = torch.full((H, W, 3), 1.0 , dtype=torch.float32, device="cuda")

# 定义背景颜色为黑色
bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

# 计算屏幕空间坐标
n_pts = pts.shape[0]
pts_h = torch.cat([pts, torch.ones(n_pts, 1, dtype=torch.float32, device="cuda")], dim=1)  # [n, 4]
screenspace_points_h = torch.matmul(pts_h, projmatrix.T)  # [n, 4]
screenspace_points = screenspace_points_h[:, :3] / screenspace_points_h[:, 3:4]  # 透视除法

# 配置高斯光栅化设置
raster_settings = GaussianRasterizationSettings(
    image_height=H,                       # 输出图像高度
    image_width=W,                        # 输出图像宽度
    tanfovx=tanfovx,                      # 水平视场角的正切
    tanfovy=tanfovy,                      # 垂直视场角的正切
    bg=bg,                                  # 背景颜色
    scale_modifier=1.0,                     # 缩放因子
    viewmatrix=viewmatrix,                  # 视图矩阵
    projmatrix=projmatrix,                  # 投影矩阵
    sh_degree=0,                            # 球谐次数
    campos=cam_pos,                         # 相机位置
    prefiltered=False,                      # 是否使用预滤波
    debug=False,                             # 是否启用调试模式
    antialiasing=False,                     # 抗锯齿设置
)

# 初始化高斯光栅化器
rasterizer = GaussianRasterizer(raster_settings=raster_settings)
# 检查 c_med 的梯度
# gradient_check(
#     variable_name='c_med',
#     variable=c_med,  # 要检查的变量
#     rasterizer=rasterizer,
#     loss_fn=loss_function,
#     means3D=pts,
#     means2D=screenspace_points,
#     shs=shs,
#     colors_precomp=None,
#     opacities=opacities,
#     scales=scales,
#     rotations=rotations,
#     cov3D_precomp=None,
#     medium_rgb=c_med,
#     medium_bs=sigma_bs,
#     medium_attn=sigma_atten,
#     colors_enhance=colors_enhance
# )


# # 检查 sigma_bs 的梯度 目前有问题
gradient_check(
    variable_name='sigma_bs',
    variable=sigma_bs,  # 要检查的变量
    rasterizer=rasterizer,
    loss_fn=loss_function,
    means3D=pts,
    means2D=screenspace_points,
    shs=shs,
    colors_precomp=None,
    opacities=opacities,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=None,
    medium_rgb=c_med,
    medium_bs=sigma_bs,
    medium_attn=sigma_atten,
    colors_enhance=colors_enhance
)
#
# 检查 sigma_atten 的梯度
gradient_check(
    variable_name='sigma_atten',
    variable=sigma_atten,  # 要检查的变量
    rasterizer=rasterizer,
    loss_fn=loss_function,
    means3D=pts,
    means2D=screenspace_points,
    shs=shs,
    colors_precomp=None,
    opacities=opacities,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=None,
    medium_rgb=c_med,
    medium_bs=sigma_bs,
    medium_attn=sigma_atten,
    colors_enhance=colors_enhance
)
#
# # 检查 colors_enhance 的梯度
gradient_check(
    variable_name='colors_enhance',
    variable=colors_enhance,  # 要检查的变量
    rasterizer=rasterizer,
    loss_fn=loss_function,
    means3D=pts,
    means2D=screenspace_points,
    shs=shs,
    colors_precomp=None,
    opacities=opacities,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=None,
    medium_rgb=c_med,
    medium_bs=sigma_bs,
    medium_attn=sigma_atten,
    colors_enhance=colors_enhance
)

gradient_check(
    variable_name='colors_enhance',
    variable=colors_enhance,  # 要检查的变量
    rasterizer=rasterizer,
    loss_fn=loss_function,
    means3D=pts,
    means2D=screenspace_points,
    shs=shs,
    colors_precomp=None,
    opacities=opacities,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=None,
    medium_rgb=c_med,
    medium_bs=sigma_bs,
    medium_attn=sigma_atten,
    colors_enhance=colors_enhance
)


# gradient_check(
#     variable_name='alpha',
#     variable=colors_enhance,  # 要检查的变量
#     rasterizer=rasterizer,
#     loss_fn=loss_function,
#     means3D=pts,
#     means2D=screenspace_points,
#     shs=shs,
#     colors_precomp=None,
#     opacities=opacities,
#     scales=scales,
#     rotations=rotations,
#     cov3D_precomp=None,
#     medium_rgb=c_med,
#     medium_bs=sigma_bs,
#     medium_attn=sigma_atten,
#     colors_enhance=colors_enhance
# )
# """-----------------------------------------------------------------"""
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
# def gradient_check2(variable_name, variable, rasterizer, loss_fn, **kwargs):
#     """
#     检查指定变量的梯度是否正确。
#
#     Args:
#         variable_name (str): 变量的名称，用于打印信息。
#         variable (torch.Tensor): 需要检查梯度的变量，要求requires_grad=True。
#         rasterizer (GaussianRasterizer): 高斯光栅化器对象。
#         loss_fn (callable): 损失函数，接受光栅化输出并返回标量损失。
#         kwargs: 传递给rasterizer的其他参数。
#     """
#     epsilon = 1e-4
#     # 确保变量需要计算梯度
#     variable.requires_grad = True
#
#     # 计算自动求导的梯度
#     rasterizer.zero_grad()  # 清零之前的梯度
#     output = rasterizer(
#         means3D=kwargs.get('means3D'),
#         means2D=kwargs.get('means2D'),
#         shs=kwargs.get('shs'),
#         colors_precomp=kwargs.get('colors_precomp'),
#         opacities=kwargs.get('opacities'),
#         scales=kwargs.get('scales'),
#         rotations=kwargs.get('rotations'),
#         cov3D_precomp=kwargs.get('cov3D_precomp'))
#
#     loss = loss_fn(output)
#     loss.backward()
#     autograd_grad = variable.grad.clone()
#
#     # 数值梯度初始化
#     numerical_grad = torch.zeros_like(variable)
#
#     # 使用 flatten 方法进行一维化遍历
#     variable_flat = variable.view(-1)                       # 展平的变量
#     numerical_grad_flat = numerical_grad.view(-1) # 展平的数值梯度量
#     autograd_grad_flat = autograd_grad.view(-1)       #展平的自动梯度
#
#     for idx in range(variable_flat.numel()):
#         original_value = variable_flat[idx].item()
#
#         # # 创建变量的副本以避免原地操作
#         with torch.no_grad():
#             variable_flat[idx] = original_value + epsilon
#         output_plus = rasterizer(
#             means3D=kwargs.get('means3D'),
#             means2D=kwargs.get('means2D'),
#             shs=kwargs.get('shs'),
#             colors_precomp=kwargs.get('colors_precomp'),
#             opacities=kwargs.get('opacities'),
#             scales=kwargs.get('scales'),
#             rotations=kwargs.get('rotations'),
#             cov3D_precomp=kwargs.get('cov3D_precomp')
#         )
#         loss_plus = loss_fn(output_plus).item()
#
#         with torch.no_grad():
#             variable_flat[idx] = original_value - epsilon
#         output_minus = rasterizer(
#             means3D=kwargs.get('means3D'),
#             means2D=kwargs.get('means2D'),
#             shs=kwargs.get('shs'),
#             colors_precomp=kwargs.get('colors_precomp'),
#             opacities=kwargs.get('opacities'),
#             scales=kwargs.get('scales'),
#             rotations=kwargs.get('rotations'),
#             cov3D_precomp=kwargs.get('cov3D_precomp')
#         )
#         loss_minus = loss_fn(output_minus).item()
#
#         # 恢复原始值
#         with torch.no_grad():
#             variable_flat[idx] = original_value
#
#         # 计算数值梯度
#         numerical_grad_flat[idx] = (loss_plus - loss_minus) / (2 * epsilon)
#
#     # 重新形状还原
#     numerical_grad = numerical_grad_flat.view_as(variable)
#     autograd_grad = autograd_grad_flat.view_as(variable)
#     print("numerical_grad\n",numerical_grad.detach().cpu().numpy())
#     print("autograd_grad\n",autograd_grad.detach().cpu().numpy())
#
#
#     # 计算相对误差
#     # relative_error = (autograd_grad - numerical_grad).abs() #/ (numerical_grad.abs() + 1e-8)
#     # max_error = relative_error.max().item()
#     # mean_error = relative_error.mean().item()
#     #
#     # print(f"梯度检查 - {variable_name}: 最大相对误差 = {max_error:.6f}, 平均相对误差 = {mean_error:.6f}")
#     #
#     # if max_error < 1e-3:
#     #     print(f"{variable_name} 的梯度实现可能是正确的。")
#     # else:
#     #     print(f"{variable_name} 的梯度实现可能存在问题，请检查。")
#
# # 检查 原始高斯梯度
# gradient_check2(
#     variable_name='pts',
#     variable=pts,  # 要检查的变量
#     rasterizer=rasterizer,
#     loss_fn=loss_function2,
#     means3D=pts,
#     means2D=screenspace_points,
#     shs=shs,
#     colors_precomp=None,
#     opacities=opacities,
#     scales=scales,
#     rotations=rotations,
#     cov3D_precomp=None
# )