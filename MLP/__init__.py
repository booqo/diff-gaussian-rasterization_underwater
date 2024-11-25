
import torch
from .mlp import color_mlp, get_embedder, MediumMLP, ray_encoding
import torch.nn as nn


# 初始化颜色 MLP 模型，并将其加载到 GPU 上进行计算。
model = color_mlp()  # 实例化颜色 MLP（多层感知机）模型。
model.to(device="cuda")  # 将模型加载到 GPU 设备上。

# 颜色优化器的学习率设置。
color_lr = 0.0001  # 为颜色相关参数设置的初始学习率。

eps = 1e-5  # Adam 优化器中的数值稳定项 epsilon。

# 初始化颜色参数的 Adam 优化器，并为不同的 MLP 层设置学习率。
optimizer_color = torch.optim.Adam(
    [
        # {"params": model.pts_linears.parameters(), "lr": color_lr, "eps": eps},  # 处理空间点的线性层。
        {"params": model.views_linears.parameters(), "lr": color_lr, "eps": eps},  # 处理视角的线性层。
        {"params": model.nviews_linears.parameters(), "lr": color_lr, "eps": eps},  # 处理多视角信息的线性层。
        {"params": model.rgb_linear.parameters(), "lr": color_lr, "eps": eps},  # 输出 RGB 颜色的线性层。
        {"params": model.v_linear.parameters(), "lr": color_lr, "eps": eps},  # 视角相关线性层。
    ]
)

# 初始化增强模块的 Adam 优化器，为不同的增强层设置较低的学习率。
optimizer_enhance = torch.optim.Adam(
    [
        {"params": model.coeff_linears.parameters(), "lr": color_lr / 5, "eps": eps},  # 系数线性层。
        {"params": model.gamma_mlp1.parameters(), "lr": color_lr / 5, "eps": eps},  # 第一层 Gamma MLP。
        {"params": model.alpha_mlp1.parameters(), "lr": color_lr / 5, "eps": eps},  # 第一层 Alpha MLP。
        {"params": model.gamma_mlp2.parameters(), "lr": color_lr / 5, "eps": eps},  # 第二层 Gamma MLP。
        {"params": model.alpha_mlp2.parameters(), "lr": color_lr / 5, "eps": eps},  # 第二层 Alpha MLP。
    ]
)


# 定义激活函数
density_activation = nn.Softplus()  # 原始文章里使用 Softplus 激活函数，要简单就用nn.ReLU()
rgb_activation = nn.Sigmoid()  # RGB 激活函数使用 Sigmoid
sigma_activation = nn.Softplus()  # 使用 Softplus 激活函数

# 定义其他参数 用于测试介质mlp
input_dim = 27  # 输入维度，根据实际情况调整，方向重新编码，4*3*2+3 = 27
net_depth_water = 8  # 介质网络深度 8
net_width_viewdirs = 256  # 宽度256
num_rgb_channels = 3
water_bias = 0.1
skip_layer_dir = 4

# 实例化 MediumMLP
medium_mlp = MediumMLP(
    input_dim=input_dim,
    net_depth_water=net_depth_water,
    net_width_viewdirs=net_width_viewdirs,
    num_rgb_channels=num_rgb_channels,
    density_activation=density_activation,
    rgb_activation=rgb_activation,
    sigma_activation=sigma_activation,
    water_bias=water_bias,
    skip_layer_dir=skip_layer_dir
).to(device="cuda")

# dir_enc_for_water = torch.ones((batch_size,input_dim)).to(device="cuda")
# 将介质 MLP 的参数添加到优化器中
medium_lr = 1e-5  # 学习率
eps = 1e-8  # Adam 优化器中的数值稳定项 epsilon。
optimizer_medium = torch.optim.Adam(
    medium_mlp.parameters(),
    lr=medium_lr,
    eps=eps
)