from diff_gaussian_rasterization_underwater import GaussianRasterizationSettings, GaussianRasterizer
import torch

import math
import numpy as np



def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2)) #需要将视场角从度转换为弧度，然后计算垂直视场角的一半的正切值

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


# 定义高斯体中心和特征
pts = np.array([[2, 0, -2], [0, 2, -2], [-2, 0, -2]])
n = len(pts)
np.random.seed(100)
shs = np.random.random((n, 16, 3))
opacities = np.ones((n, 1))
scales = np.ones((n, 3))
rotations = np.array([np.eye(3)] * n)

# 将 NumPy 数据转换为 PyTorch 张量
pts = torch.tensor(pts, dtype=torch.float32, device="cuda")
shs = torch.tensor(shs, dtype=torch.float32, device="cuda")
opacities = torch.tensor(opacities, dtype=torch.float32, device="cuda")
scales = torch.tensor(scales, dtype=torch.float32, device="cuda")
rotations = torch.tensor(rotations, dtype=torch.float32, device="cuda")

# 相机参数和投影矩阵
cam_pos = torch.tensor([0, 0, 10], dtype=torch.float32, device="cuda")
R = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32, device="cuda")
proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
viewmatrix = torch.tensor(getWorld2View2(R=R.cpu().numpy(), t=cam_pos.cpu().numpy()), dtype=torch.float32, device="cuda")
projmatrix = torch.tensor(getProjectionMatrix(**proj_param), dtype=torch.float32, device="cuda")
projmatrix = torch.matmul(projmatrix, viewmatrix)

# Medium 参数
c_med = torch.tensor([0.5], dtype=torch.float32, device="cuda")
sigma_bs = torch.tensor([0.5], dtype=torch.float32, device="cuda")
sigma_atten = torch.tensor([0.5], dtype=torch.float32, device="cuda")

# 背景
bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

screenspace_points = torch.zeros_like(pts, requires_grad=True)

raster_settings = GaussianRasterizationSettings(
    image_height=200,
    image_width=200,
    tanfovx=math.tan(proj_param["fovX"] * 0.5),
    tanfovy=math.tan(proj_param["fovY"] * 0.5),
    bg=bg,
    scale_modifier=1.0,
    viewmatrix=viewmatrix,
    projmatrix=projmatrix,
    sh_degree=1,
    campos=cam_pos,
    prefiltered=False,
    debug=False,
    medium_rgb=c_med,
    medium_bs=sigma_bs,
    medium_attn=sigma_atten,
    antialiasing = None
)

rasterizer = GaussianRasterizer(raster_settings=raster_settings)

rendered_image, radii, _ = rasterizer(
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
    medium_attn=sigma_atten
)

import matplotlib.pyplot as plt
rendered_image = rendered_image.permute(1, 2, 0)

# 使用 detach() 分离梯度，然后移动到 CPU 并转换为 NumPy
plt.imshow(rendered_image.detach().cpu().numpy())
plt.axis('off')  # 可选，去掉坐标轴
plt.show()


