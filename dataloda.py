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
from MLP.mlp import ray_encoding
from MLP.mlp import run_waternetwork,run_network,get_embedder,position_encoding
import MLP.__init__
from utils.general_utils import safe_state



def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, model, medium_mlp, embed_fn, embeddirs_fn,
           scaling_modifier=1.0, override_color=None, c_med=None, sigma_bs=None, sigma_atten=None,colors_enhance = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means# 创建一个与输入点云（高斯模型）大小相同的零张量，用于记录屏幕空间中的点的位置。这个张量将用于计算对于屏幕空间坐标的梯度。
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()  # 尝试保留张量的梯度。这是为了确保可以在反向传播过程中计算对于屏幕空间坐标的梯度。
    except:
        pass

    # Set up rasterization configuration# 计算视场的 tan 值，这将用于设置光栅化配置。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)
    R =viewpoint_camera.R
    # input_vect = ray_encoding(H,W,tanfovx,tanfovy,R)
    input_vect = position_encoding(pc.get_xyz ,viewpoint_camera,)


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

    cov3D_precomp = None
    shs = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)  # 获取预计算的三维协方差矩阵。
    else:  # 获取缩放和旋转信息。（对应的就是3D高斯的协方差矩阵了）
        scales = pc.get_scaling  # 缩放三维
        rotations = pc.get_rotation  # 旋转四元数[1,0,0,0]

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

    # colors_enhance = run_network(input_vect, model, embeddirs_fn)#直接得到增强v 吃显存
    # colors_enhance = colors_enhance.view(H, W, 3)

    medium_output = run_waternetwork(input_vect, medium_mlp, embeddirs_fn)#这里得到的是 每个通道都有,c_med 3, sigma_bs 3, sigma_atten 3 也吃
    sigma_bs, sigma_atten, c_med = torch.split(medium_output, [3, 3, 3], dim=-1)
    sigma_bs = sigma_bs.view(H, W, 3)
    sigma_atten = sigma_atten.view(H, W, 3)
    c_med = c_med.view(H, W, 3)

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
            "color_attn": color_attn,  # 物体颜色衰减后
            }


def training1(dataset, pipe=None, opt=None):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0] #设置背景颜色，根据数据集是否有白色背景来选择
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") #将背景颜色转化为 PyTorch Tensor，并移到 GPU 上。
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    bg = background  # 背景

    embeddirs_fn = get_embedder(4)  # 获取用于嵌入方向的嵌入函数，维度为 4。

    model = MLP.model
    medium_mlp = MLP.medium_mlp


    for iteration in range(0, 1000):
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, model=model, medium_mlp=medium_mlp, embed_fn=None, embeddirs_fn = embeddirs_fn)

        image, viewspace_point_tensor, visibility_filter, radii, medium_rgb, color_clr, depths, color_attn = \
            (render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"],
             render_pkg["radii"], render_pkg["medium_rgb"], render_pkg["color_clr"],
             render_pkg["depths"], render_pkg["color_attn"])
        '''------------------------------------Loss Function-----------------------------'''
        # 计算 direct 和 J
        # direct = torch.clamp(color_attn, 0., 1.)  # 物体的直接分量，限制的范围待定
        # J = color_clr  # 无衰减的图像干净的恢复图像
        #
        # bs_criterion = BackscatterLoss().to('cuda')
        # da_criterion = DeattenuateLoss().to('cuda')
        #
        # # 计算损失
        # bs_loss = bs_criterion(direct)
        # da_loss = da_criterion(direct, J)
        #
        # # 计算原有的损失
        # gt_image = viewpoint_cam.original_image.cuda()  # （3，H，W）
        #
        # Ll1 = l1_loss(image, gt_image)  # scalar，公式中的 l1，平均绝对误差
        # ssim_value = ssim(image, gt_image)
        # # loss1 = (0.8) * Ll1 + 0.2 * (1.0 - ssim_value)
        # # 总损失
        # loss = bs_loss + da_loss + Ll1# +loss1


        gt_image = viewpoint_cam.original_image.cuda()  # （3，H，W）\

        print (image.shape)
        loss= (image-gt_image)**2
        loss = loss.mean()

        loss.backward()


        MLP.optimizer_enhance.zero_grad(set_to_none=True)  # 增强mlp部分梯度清0
        MLP.optimizer_medium.zero_grad(set_to_none=True)  # 介质部分梯度清0

        # loss.backward()  # pytorch反向传播
        print(f"Iteration {iteration}, Loss: {loss.item()}")
        # print(loss.tolist())

    # with torch.no_grad():
        # Optimizer step
        # gaussians.exposure_optimizer.step()
        # gaussians.exposure_optimizer.zero_grad(set_to_none=True)
        # gaussians.optimizer.step()

        # gaussians.optimizer.step()

        MLP.optimizer_medium.step()
        MLP.optimizer_enhance.step()

        # gaussians.optimizer.zero_grad(set_to_none=True)


    # rendered_image = net_image.permute(1, 2, 0)  # 从[3, H, W]转换为[H, W, 3]
    #
    # # 将渲染图像从GPU移动到CPU，并分离梯度，转换为NumPy数组
    # rendered_image_np = rendered_image.detach().cpu().numpy()
    #
    # # 使用Matplotlib显示渲染结果
    # plt.imshow(rendered_image_np)
    # plt.axis('off')  # 可选，去掉坐标轴
    # plt.show()


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
def training(dataset, pipe=None, opt=None):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    bg = background

    embeddirs_fn = get_embedder(4)

    # model = MLP.model
    medium_mlp = MLP.medium_mlp

    # 将模型设置为训练模式
    # model.train()
    medium_mlp.train()

    losses = []

    for iteration in range(0, 1000):
        # 清零梯度
        MLP.optimizer_enhance.zero_grad(set_to_none=True)
        MLP.optimizer_medium.zero_grad(set_to_none=True)

        # 前向传播和损失计算
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, model=None, medium_mlp=medium_mlp, embed_fn=None, embeddirs_fn=embeddirs_fn)
        image, viewspace_point_tensor, visibility_filter, radii, medium_rgb, color_clr, depths, color_attn = \
            (render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"],
             render_pkg["radii"], render_pkg["medium_rgb"], render_pkg["color_clr"],
             render_pkg["depths"], render_pkg["color_attn"])

        gt_image = viewpoint_cam.original_image.cuda()  # （3，H，W）

        direct = torch.clamp((gt_image-medium_rgb), 0., 1.) #多余
        J = color_clr
        # lossFunction = torch.nn.MSELoss
        bs_criterion = BackscatterLoss().to('cuda')
        da_criterion = DeattenuateLoss().to('cuda')

        bs_loss = bs_criterion(direct)
        da_loss = da_criterion(direct, J)

        # loss1 = bs_loss + da_loss
        # loss1 = (direct- J)**2
        # scalar_loss = loss1.mean()
        # gt_image = viewpoint_cam.original_image.cuda()
        #
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss1 = (0.8) * Ll1 + 0.2 * (1.0 - ssim_value) + bs_loss + da_loss

        # loss =loss1 + bs_loss + da_loss

        # 反向传播
        loss1.backward()
        # torch.autograd.set_detect_anomaly(True)


        # 梯度裁剪
        # torch.nn.utils.clip_grad_norm_(medium_mlp.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # 优化器更新
        MLP.optimizer_medium.step()
        MLP.optimizer_enhance.step()

        # 打印损失
        print(f"Iteration {iteration}, Loss: {loss1.item()}")
        print(f"Iteration {iteration}, Loss: {loss1.item()}")

        losses.append(loss1.item())

    # 训练结束后绘制损失曲线
    plt.figure(figsize=(10, 5))  # 设置图形大小
    plt.plot(losses, label='Training Loss')  # 绘制损失值
    plt.title('Loss Curve over Iterations')  # 设置图表标题
    plt.xlabel('Iterations')  # 设置x轴标签
    plt.ylabel('Loss')  # 设置y轴标签
    plt.legend()  # 显示图例
    plt.show()  # 显示图表


# if __name__ =="__main__":
#     parser = ArgumentParser(description="dataloda")
#     lp = ModelParams(parser)#模型参数
#     # op = OptimizationParams(parser)#各种优化参数参数
#     pp = PipelineParams(parser)
#     # args = parser
#     args = parser.parse_args()
#
#     # args.source_path = "/home/asus/桌面/lowlight_under_watergs/lowlight-gaussian-splatting-underwater/submodules/diff-gaussian-rasterization_underwater/test_colmap_data"
#     # args.model_path = "/home/asus/桌面/lowlight_under_watergs/lowlight-gaussian-splatting-underwater/submodules/diff-gaussian-rasterization_underwater/test_colmap_data/data"
#     args.source_path = "/media/asus/2d50eb40-f432-4954-bc07-5d4b5c5887cd/datasets_process/D3_seathru/colmap"
#     args.model_path = "/media/asus/2d50eb40-f432-4954-bc07-5d4b5c5887cd/datasets_process/document_result/D3_seathru/unnamed/gaussian"
#     training(lp.extract(args),pp.extract(args))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        # network_gui.init(args.ip, args.port)
        pass
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # args.source_path = "/home/asus/桌面/gaussian-splatting/data/campus"
    # args.model_path = "./data/output/"

    # args.source_path = "/media/asus/2d50eb40-f432-4954-bc07-5d4b5c5887cd/datasets_process/2_my_data_light/auto"
    # args.model_path = "/media/asus/2d50eb40-f432-4954-bc07-5d4b5c5887cd/datasets_process/document_result/2_my_data_light/unnamed/gaussian"
    # args.source_path = "/media/asus/2d50eb40-f432-4954-bc07-5d4b5c5887cd/datasets_process/Curasao"
    # args.model_path = "/media/asus/2d50eb40-f432-4954-bc07-5d4b5c5887cd/datasets_process/document_result/Curasao/unnamed/gaussian"
    args.source_path = "/media/asus/2d50eb40-f432-4954-bc07-5d4b5c5887cd/datasets_process/D3_seathru/colmap"
    args.model_path = "/media/asus/2d50eb40-f432-4954-bc07-5d4b5c5887cd/datasets_process/document_result/D3_seathru/unnamed/gaussian"

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")