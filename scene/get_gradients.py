import torch
from utils.sh_utils import eval_sh
from tqdm import tqdm

def compute_gradient(sh_degree, sh_coeffs, directions):
    colors = []
    gradients = []
    for i, direction in enumerate(tqdm(directions, desc="Processing directions")):
        sh_coeff = sh_coeffs[i].unsqueeze(0)
        sh_coeff = sh_coeff.expand(direction.size(0), 3, 16)
        color = eval_sh(sh_degree, sh_coeff, direction)
        colors.append(color)
        gradient = calculate_gradient(direction, color)
        gradients.append(gradient)
    gradients = torch.stack(gradients)
    gradients_mean = torch.mean(gradients, dim=1).unsqueeze(1)
    return gradients_mean

def calculate_gradient(directions, colors):
    """
    计算每对相邻方向之间的梯度，并返回统一的梯度值。

    参数:
    directions (torch.Tensor): 形状为 (28, 3) 的方向张量。
    colors (torch.Tensor): 形状为 (28, 3) 的颜色张量。

    返回:
    gradients (torch.Tensor): 每对相邻方向的梯度，形状为 (27, 3)。
    uniform_gradient (torch.Tensor): 统一的梯度值，形状为 (3,)。
    """
    # 初始化梯度数组
    gradients = torch.zeros((directions.size(0)-1, 3))  # 因为有27对梯度（从0到27，所以是28-1）

    # 计算每个相邻方向的梯度
    for i in range(directions.size(0)-1):
        # 计算梯度
        gradients[i] = (colors[i + 1] - colors[i]) / torch.norm(directions[i + 1] - directions[i])

    # 统一梯度值的计算：可以考虑取所有梯度的均值
    uniform_gradient = torch.mean(gradients, dim=0)

    return uniform_gradient


def compute_det_trace(scaling):
    """
    对于每个 (N, 3) 的行向量，计算元素的累加与累乘之和。
    输入:
        scaling: (N, 3) 的张量
    输出:
        (N, 1) 的张量，每行是元素和与乘积的相加结果。
    """
    scaling = scaling.clamp(min=1e-6, max=1e6)  # 限制范围
    # 每行求和
    row_sum = scaling.sum(dim=1, keepdim=True)  # (N, 1)

    # 每行求积
    # row_prod = scaling.prod(dim=1, keepdim=True)  # (N, 1)
    row_prod_safe = torch.exp(torch.sum(torch.log(scaling.clamp(min=1e-10)), dim=1, keepdim=True))

    # 累加
    scale_trace = row_sum + row_prod_safe  # (N, 1)

    return scale_trace

