# 导入必要的库
from diff_gaussian_rasterization_underwater import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings as GaussianRasterizationSettings2
from diff_gaussian_rasterization import GaussianRasterizer as GaussianRasterizer2

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from testcuda_forward import getWorld2View2,getProjectionMatrix
import os

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
    print("water gradient check for ", variable_name)
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

        # 创建变量的副本以避免原地操作
        with torch.no_grad():
            variable_flat[idx] = original_value + epsilon
            output_plus = rasterizer(
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
            loss_plus = loss_fn(output_plus).item()
            #print(original_value)
            variable_flat[idx] = original_value - epsilon

            output_minus = rasterizer(
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
            loss_minus = loss_fn(output_minus).item()

            variable_flat[idx] = original_value

        # 计算数值梯度
        numerical_grad_flat[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        # print("numerical_grad\n", numerical_grad_flat[idx])
        # print("autograd_grad\n", autograd_grad_flat[idx])
    
    numerical_grad = numerical_grad_flat.view_as(variable)
    autograd_grad = autograd_grad_flat.view_as(variable)
    # print("numerical_grad\n", numerical_grad.detach().cpu().numpy())
    # print("autograd_grad\n", autograd_grad.detach().cpu().numpy())

    save_gradient_as_image(numerical_grad, f'grad_image/water_grad/{variable_name}_water_numerical_grad.png',variable_name,"Water")
    save_gradient_as_image(autograd_grad, f'grad_image/water_grad/{variable_name}_water_autograd_grad.png',variable_name,"Water")
    # save_gradient_channels_as_image(numerical_grad, 'water_grad/numerical_grad_channels.png')
    # save_gradient_channels_as_image(autograd_grad, 'water_grad/autograd_grad_channels.png')
def gradient_check_gs(variable_name, variable, rasterizer, loss_fn, **kwargs):
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
    print("gs gradient check for ", variable_name)
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
        cov3D_precomp=kwargs.get('cov3D_precomp'))

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
        output_plus = rasterizer(
            means3D=kwargs.get('means3D'),
            means2D=kwargs.get('means2D'),
            shs=kwargs.get('shs'),
            colors_precomp=kwargs.get('colors_precomp'),
            opacities=kwargs.get('opacities'),
            scales=kwargs.get('scales'),
            rotations=kwargs.get('rotations'),
            cov3D_precomp=kwargs.get('cov3D_precomp')
        )
        loss_plus = loss_fn(output_plus).item()

        with torch.no_grad():
            variable_flat[idx] = original_value - epsilon
        output_minus = rasterizer(
            means3D=kwargs.get('means3D'),
            means2D=kwargs.get('means2D'),
            shs=kwargs.get('shs'),
            colors_precomp=kwargs.get('colors_precomp'),
            opacities=kwargs.get('opacities'),
            scales=kwargs.get('scales'),
            rotations=kwargs.get('rotations'),
            cov3D_precomp=kwargs.get('cov3D_precomp')
        )
        loss_minus = loss_fn(output_minus).item()

        # 恢复原始值
        with torch.no_grad():
            variable_flat[idx] = original_value

        # 计算数值梯度
        numerical_grad_flat[idx] = (loss_plus - loss_minus) / (2 * epsilon)

    # 重新形状还原
    numerical_grad = numerical_grad_flat.view_as(variable)
    autograd_grad = autograd_grad_flat.view_as(variable)
    # print("numerical_grad\n",numerical_grad.detach().cpu().numpy())
    # print("autograd_grad\n",autograd_grad.detach().cpu().numpy())
    save_gradient_as_image(numerical_grad, f'grad_image/gs_grad/{variable_name}_gs_numerical_grad.png',variable_name,"GS")
    save_gradient_as_image(autograd_grad, f'grad_image/gs_grad/{variable_name}_gs_autograd_grad.png',variable_name,"GS")
def save_gradient_as_image_old(gradient, filename,variable_name,chose):
    grad_array = gradient.detach().cpu().numpy()
    grad_norm = (grad_array - np.min(grad_array)) / (np.max(grad_array) - np.min(grad_array) + 1e-5)
    plt.imshow(grad_norm, cmap='viridis')
    plt.colorbar()
    plt.title(f'{chose}-{variable_name} Gradient Normalization Visualization')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
def save_gradient_as_image(gradient, filename, variable_name, chose):
    """
    Normalize and save the gradient tensor as an image, including max and min values.
    Also ensures that the target directory exists; if not, creates it.

    Args:
        gradient (torch.Tensor): The gradient tensor to visualize.
        filename (str): The path to save the image file.
        variable_name (str): Name of the variable being visualized (for title).
        chose (str): Additional identifier for the title.
    """
    # Detach gradient, move to CPU and convert to numpy
    grad_array = gradient.detach().cpu().numpy()

    # Compute actual min and max before normalization
    grad_min = np.min(grad_array)
    grad_max = np.max(grad_array)

    # Normalize the gradient to 0-1 for better visualization
    grad_norm = (grad_array - grad_min) / (grad_max - grad_min + 1e-5)

    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(grad_norm, cmap='viridis')  # You can choose a colormap that fits your needs
    plt.colorbar()

    # Add max and min values as text on the image
    plt.text(0.95, 0.05, f'Max: {grad_max:.4f}\nMin: {grad_min:.4f}',
             verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes,
             color='white', fontsize=12,
             bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))

    plt.title(f'{chose} - {variable_name} Gradient Visualization')
    plt.axis('off')  # Turn off axis numbers and ticks

    # Save the plot as an image file
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
def save_gradient_channels_as_image(gradient, filename):
    """
    Normalize and save each channel of the gradient tensor as an image only for c_med sigma_bs sigma_attn colors_enhance.

    Args:
    gradient (torch.Tensor): The gradient tensor to visualize, expected to have shape [H, W, C].
    filename (str): The path to save the image file.
    """
    if gradient.dim() == 4:
        gradient = gradient.squeeze(0)  # Remove batch dimension if exists
    if gradient.dim() != 3 or gradient.size(2) != 3:
        raise ValueError("Gradient tensor must have 3 channels.")

    # Detach gradient, move to CPU and convert to numpy
    grad_array = gradient.detach().cpu().numpy()

    # Normalize each channel separately to 0-1 for better visualization
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with 3 subplots
    channel_titles = ['Channel 1 - Red', 'Channel 2 - Green', 'Channel 3 - Blue']
    for i in range(3):
        channel_grad = grad_array[:, :, i]
        grad_norm = (channel_grad - np.min(channel_grad)) / (np.max(channel_grad) - np.min(channel_grad) + 1e-5)
        ax = axs[i]
        im = ax.imshow(grad_norm, cmap='viridis')
        ax.set_title(channel_titles[i])
        ax.axis('off')  # Turn off axis numbers and ticks
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
def loss_function(outputs):
    color_attn, color_clr, color_medium, radii, depths = outputs
    return (color_attn+color_medium).sum()
def loss_function_gs(outputs):
    rendered_image, radii, depth_image  = outputs
    return (rendered_image).sum()
def setup_camera_settings():
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
        "fovY": 90  # 垂直视场角（度）
    }
    tanfovx = math.tan(math.radians(proj_param["fovX"] / 2))  # 水平视场角的正切
    tanfovy = math.tan(math.radians(proj_param["fovY"] / 2))  # 垂直视场角的正切

    # 生成世界到视图的转换矩阵，并转换为PyTorch张量
    projmatrix_np = getProjectionMatrix(**proj_param)
    projmatrix = torch.tensor(projmatrix_np, dtype=torch.float32, device="cuda")

    # 计算最终的投影矩阵（投影矩阵 * 视图矩阵）
    projmatrix = torch.matmul(projmatrix, viewmatrix)

    cam_pos = torch.tensor(cam_pos, dtype=torch.float32, device="cuda")
    return cam_pos, viewmatrix, tanfovx,tanfovy,projmatrix
def setup_gaussian_params(projmatrix):# 里面改高斯的个数，手动加
    # # 定义高斯体的中心点和特征
    # pts = np.array([
    #     [0, 0, 10.0],  # 位于正Z轴方向
    #     [-0.2, -0, 10.0],
    #     [-0, 0.2, 10.0],
    #     [-0.2, -0, 10.0],
    #     [-0, 0.2, 10.0],
    #     [-0.2, -0, 10.0],
    #     [-0, 0.2, 10.0],
    #     [-0.2, -0, 10.0],
    #     [-0, 0.2, 10.0],
    #     [-0, 0.2, 10.0],
    #     [-0.2, -0, 10.0],
    #     [-0.2, -0, 10.0]
    # ], dtype=np.float32)
    # n = len(pts)  # 高斯体数量
    #
    # # 设置固定颜色（进一步调整以避免超出范围）
    # desired_colors = np.array([
    #     [100, 0.0, 0.0],  # 高斯体 1: 红色
    #     [0.0, 100, 0.0],  # 高斯体 2: 绿色
    #     [0.0, 0.0, 100],  # 高斯体 3: 蓝色
    #     [100, 0.0, 0.0],  # 高斯体 1: 红色
    #     [0.0, 100, 0.0],  # 高斯体 2: 绿色
    #     [0.0, 0.0, 100],  # 高斯体 3: 蓝色
    #     [100, 0.0, 0.0],  # 高斯体 1: 红色
    #     [0.0, 100, 0.0],  # 高斯体 2: 绿色
    #     [0.0, 0.0, 100],  # 高斯体 3: 蓝色
    #     [100, 0.0, 0.0],  # 高斯体 1: 红色
    #     [0.0, 100, 0.0],  # 高斯体 2: 绿色
    #     [0.0, 0.0, 100],  # 高斯体 3: 蓝色
    # ], dtype=np.float32)
    #
    # # 初始化球谐函数系数为全零，并设置零阶系数为所需颜色(固定颜色)
    # shs = np.zeros((n, 16, 3), dtype=np.float32)
    # shs[:, 0, :] = desired_colors  # 设置零阶系数

    n = 800
    x_range = (-1.0, 1.0)
    y_range = (-1.0, 1.0)
    z_range = (5.0, 15.0)
    np.random.seed(n)
    x = np.random.uniform(low=x_range[0], high=x_range[1], size=n)
    y = np.random.uniform(low=y_range[0], high=y_range[1], size=n)
    z = np.random.uniform(low=z_range[0], high=z_range[1], size=n)
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    shs = np.random.uniform(0,1,(n, 16, 3))
    shs[:, 1:, :]=0


    # 设置其他高斯体属性
    opacities = np.ones((n, 1), dtype=np.float32)  # 透明度设为1
    scales = np.ones((n, 3), dtype=np.float32)  # 增大尺度以确保可见
    rotations = np.tile(np.eye(3), (n, 1, 1)).astype(np.float32)  # 旋转矩阵设为单位矩阵

    # 将NumPy数组转换为PyTorch张量，并移动到GPU
    pts = torch.tensor(pts, dtype=torch.float32, device="cuda")
    shs = torch.tensor(shs, dtype=torch.float32, device="cuda")
    opacities = torch.tensor(opacities, dtype=torch.float32, device="cuda")
    scales = torch.tensor(scales, dtype=torch.float32, device="cuda")
    rotations = torch.tensor(rotations, dtype=torch.float32, device="cuda")
    # 计算屏幕空间坐标
    n_pts = pts.shape[0]
    pts_h = torch.cat([pts, torch.ones(n_pts, 1, dtype=torch.float32, device="cuda")], dim=1)  # [n, 4]
    screenspace_points_h = torch.matmul(pts_h, projmatrix.T)  # [n, 4]
    screenspace_points = screenspace_points_h[:, :3] / screenspace_points_h[:, 3:4]  # 透视除法
    return pts,shs,opacities,scales,rotations,screenspace_points
def setup_medium_params(H, W):
    # 定义介质参数，形状应该是 H, W, 3
    c_med = torch.zeros((H, W, 3), dtype=torch.float32, device="cuda")
    c_med[:, :, 0] = 1.0  # 红色通道
    c_med[:, :, 1] = 1.0  # 绿色通道
    c_med[:, :, 2] = 1.0  # 蓝色通道
    sigma_bs = torch.full((H, W, 3), 0.3, dtype=torch.float32, device="cuda")
    sigma_atten = torch.full((H, W, 3), 0.3, dtype=torch.float32, device="cuda")

    # 定义增强颜色参数，形状应该是 H, W, 3
    colors_enhance = torch.full((H, W, 3), 2.0, dtype=torch.float32, device="cuda")

    # 定义背景颜色为黑色
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    return c_med, sigma_bs, sigma_atten, colors_enhance,bg
def create_rasterizer(H, W ,tanfovx,tanfovy, bg, viewmatrix,projmatrix,cam_pos):
    raster_settings = GaussianRasterizationSettings(
        image_height=H,  # 输出图像高度
        image_width=W,  # 输出图像宽度
        tanfovx=tanfovx,  # 水平视场角的正切
        tanfovy=tanfovy,  # 垂直视场角的正切
        bg=bg,  # 背景颜色
        scale_modifier=1.0,  # 缩放因子
        viewmatrix=viewmatrix,  # 视图矩阵
        projmatrix=projmatrix,  # 投影矩阵
        sh_degree=0,  # 球谐次数
        campos=cam_pos,  # 相机位置
        prefiltered=False,  # 是否使用预滤波
        debug=False,  # 是否启用调试模式
        antialiasing=False,  # 抗锯齿设置
    )
    return GaussianRasterizer(raster_settings=raster_settings)
def create_rasterizer2(H, W ,tanfovx,tanfovy, bg, viewmatrix,projmatrix,cam_pos):
    raster_settings = GaussianRasterizationSettings2(
        image_height=H,  # 输出图像高度
        image_width=W,  # 输出图像宽度
        tanfovx=tanfovx,  # 水平视场角的正切
        tanfovy=tanfovy,  # 垂直视场角的正切
        bg=bg,  # 背景颜色
        scale_modifier=1.0,  # 缩放因子
        viewmatrix=viewmatrix,  # 视图矩阵
        projmatrix=projmatrix,  # 投影矩阵
        sh_degree=0,  # 球谐次数
        campos=cam_pos,  # 相机位置
        prefiltered=False,  # 是否使用预滤波
        debug=False,  # 是否启用调试模式
        # antialiasing=False,  # 抗锯齿设置
    )
    return GaussianRasterizer2(raster_settings=raster_settings)

H, W = 200, 200
cam_pos, viewmatrix, tanfovx,tanfovy,projmatrix = setup_camera_settings()#相机参数

pts,shs,opacities,scales,rotations,screenspace_points = setup_gaussian_params(projmatrix) #高斯参数

c_med, sigma_bs, sigma_atten, colors_enhance,bg = setup_medium_params(H, W) # 介质参数
rasterizer = create_rasterizer(H, W ,tanfovx,tanfovy, bg, viewmatrix,projmatrix,cam_pos) #水下光栅器设置
rasterizer2 = create_rasterizer2(H, W ,tanfovx,tanfovy, bg, viewmatrix,projmatrix,cam_pos) #p普通光栅器设置

# 检查water光栅化的梯度
gs_list = {'pts': pts,'shs': shs,'opacities': opacities,'scales': scales}
water_list = {**gs_list , **{'c_med': c_med,'sigma_bs': sigma_bs,'sigma_atten': sigma_atten,'colors_enhance': colors_enhance}}
for test_param in water_list:
    gradient_check(
        variable_name=test_param,
        variable=water_list[test_param],  # 要检查的变量
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

for test_param in gs_list:
    # 检查 原始高斯光栅化梯度
    gradient_check_gs(
        variable_name=test_param,
        variable=gs_list[test_param],  # 要检查的变量
        rasterizer=rasterizer2,
        loss_fn=loss_function_gs,
        means3D=pts,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )