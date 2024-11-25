import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



class Embedder:
    """
    嵌入器 (Embedder) 类：将输入映射到高维空间，
    通过频率编码生成更丰富的特征表示，这在神经辐射场（NeRF）等应用中非常常见。
    """
    def __init__(self, **kwargs):
        """
        初始化嵌入器对象，并根据传入参数创建嵌入函数。

        参数：
            **kwargs：字典形式的可变参数，用于指定嵌入的配置，如输入维度、频率数量等。
        """
        self.kwargs = kwargs # 存储传入的参数配置。
        self.create_embedding_fn() # 创建嵌入函数。

    def create_embedding_fn(self):
        """
        创建嵌入函数，根据指定的频率和周期函数生成多个嵌入形式。
        """
        embed_fns = []# 存储嵌入函数的列表。
        d = self.kwargs['input_dims']# 输入维度（通常为 3，例如 RGB 或 3D 坐标）。
        out_dim = 0 # 初始化输出维度。
        # 可选：是否包含原始输入作为嵌入的一部分。
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)# 将原始输入作为嵌入函数之一。
            out_dim += d # 输出维度增加 d。

        # 获取频率参数配置。
        max_freq = self.kwargs['max_freq_log2']# 最大频率的 log2 值。
        N_freqs = self.kwargs['num_freqs']# 频率数量。

        # 根据是否进行对数采样生成频率带。
        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)# 对数间隔采样。
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs) # 线性间隔采样。

        # 为每个频率和周期函数生成嵌入函数。
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:# 遍历周期函数，如 sin 和 cos。
                # 使用 lambda 表达式捕获当前的频率和周期函数。
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d# 每添加一个嵌入函数，输出维度增加 d。

        self.embed_fns = embed_fns# 存储生成的嵌入函数列表。
        self.out_dim = out_dim# 记录最终的输出维度。

    def embed(self, inputs):
        """
        将输入数据通过所有嵌入函数进行处理，并将结果拼接成一个高维张量。

        参数：
            inputs: 输入数据张量。

        返回：
            高维嵌入张量，将多个嵌入结果沿最后一个维度拼接。
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):#NeRF 位置编码代
    """
    创建并返回嵌入函数，用于对输入进行高频编码。

    参数：
        multires: 多分辨率频率的数量（影响嵌入的频率数量）。
        i: 如果 i 为 -1，则返回恒等映射，否则生成一个嵌入器。

    返回：
        一个嵌入函数，以及其输出维度（默认为 3）。
    """
    if i == -1:
        return nn.Identity(), 3 # 如果 i 为 -1，则返回恒等映射（无嵌入）。

    # 嵌入器的参数配置。
    embed_kwargs = {
        'include_input': True, # 是否包含原始输入。
        'input_dims': 3, # 输入的维度。
        'max_freq_log2': multires - 1,# 最大频率的 log2 值。
        'num_freqs': multires,# 频率数量。# 即论文中 5.1 节位置编码公式中的 L
        'log_sampling': True,# 是否对频率进行对数间隔采样。
        'periodic_fns': [torch.sin, torch.cos],# 周期函数列表（sin 和 cos）。
    }
    # 创建嵌入器对象，并生成嵌入函数。
    embedder_obj = Embedder(**embed_kwargs)
    # 使用 lambda 表达式包装嵌入函数，确保调用时使用 embedder_obj 的 embed 方法。
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed# 返回嵌入函数。

def ray_encoding(H,W,FoVx,FoVy,R):

    # H = int(viewpoint_camera.image_height)
    # W = int(viewpoint_camera.image_width)
    # 计算相机的焦距 fx 和 fy
    # fx = (W / 2) / tan(FoVx / 2)
    # fy = (H / 2) / tan(FoVy / 2)
    fx = (W / 2) / math.tan(FoVx * 0.5)
    fy = (H / 2) / math.tan(FoVy * 0.5)

    # 编码方向
    y = torch.arange(0., H, 1., device="cuda").float()  # 生成 y 坐标（0 到 H-1）
    x = torch.arange(0., W, 1., device="cuda").float()  # 生成 x 坐标（0 到 W-1）
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # 显式指定索引顺序  # 创建网格
    yy = (yy - H / 2) / fy  # 标准化y坐标
    xx = (xx - W / 2) / fx  # 标准化x坐标
    directions = torch.stack([yy, xx, -torch.ones_like(xx)], dim=-1)  # 方向向量形状为 (H, W, 3)
    norms = torch.linalg.norm(directions, dim=-1, keepdim=True)  # 计算方向向量的范数
    directions = directions / norms  # 归一化方向向量

    # 检查旋转矩阵 R 是否为 torch.Tensor，如果不是则转换
    if not isinstance(R, torch.Tensor):
        R = torch.tensor(R, dtype=torch.float32)

    # 将旋转矩阵移动到 CUDA 设备
    R = R.to("cuda")

    # 确保旋转矩阵的形状为 (3, 3)
    assert R.shape == (3, 3), "R 应为 3x3 旋转矩阵"

    # 将方向向量展平成二维张量，进行旋转
    directions_flat = directions.view(-1, 3) @ R.T  # 矩阵乘法，旋转方向向量
    return directions_flat

def batchify(fn, chunk): # nerf/run_nerf.py里面的函数
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(viewdirs, fn, embeddirs_fn=None, netchunk=1024*64):
    # output = run_network(rays_o, rays_d, model, embed_fn, embeddirs_fn)在init中初始化，输入rays_o
    # 是nerf中的函数
    """Prepares inputs and applies network 'fn'.
    Args:
    inputs: 输入张量，通常是场景中点的位置。
    viewdirs: 视图方向张量，表示每个点的观察方向。
    fn: 神经网络模型，用于处理输入并生成输出。
    embed_fn: 可选的嵌入函数，用于对输入进行编码。
    embeddirs_fn: 可选的嵌入函数，用于对视图方向进行编码。
    netchunk: 网络处理的批次大小，默认为1024*64。

    Returns:
    outputs: 网络输出，形状与输入相匹配，但最后一个维度是输出特征的数量。
    """
    # NeRF对坐标3维取L = 10，而视角方向2维取L = 4，则启用PositionalEncoding的模型需要将MLP的输入维度从5改为（3×10×2 + 2×4×2）=76

    if viewdirs is not None:# 如果提供了视图方向，则对它们进行编码并将其与输入编码拼接
        embedded_dirs = embeddirs_fn(viewdirs) # 27，理论24
    viewdirs_shape = viewdirs.shape
    outputs_flat = batchify(fn, netchunk)(embedded_dirs) # 使用batchify函数将输入分块，以便网络可以处理大型输入
    # 将输出重新塑形为与原始输入相同的形状，但最后一个维度是输出特征的数量
    return outputs_flat

def run_waternetwork(viewdirs, medium_mlp, embeddirs_fn=None, netchunk=1024*64):
    # output = run_network(rays_o, rays_d, model, embed_fn, embeddirs_fn)在init中初始化，输入rays_o
    # 是nerf中的函数
    """Prepares inputs and applies network 'fn'.
    Args:
    inputs: 输入张量，通常是场景中点的位置。
    viewdirs: 视图方向张量，表示每个点的观察方向。
    fn: 神经网络模型，用于处理输入并生成输出。
    embed_fn: 可选的嵌入函数，用于对输入进行编码。
    embeddirs_fn: 可选的嵌入函数，用于对视图方向进行编码。
    netchunk: 网络处理的批次大小，默认为1024*64。

    Returns:
    outputs: 网络输出，形状与输入相匹配，但最后一个维度是输出特征的数量。
    """
    # 对于介质编码，输入三个方向，MLP的输入应该是3×4×2，如果不考虑绕Z轴的角，应该是2×4×2
    if viewdirs is not None:# 如果提供了像素方向，则对它们进行编码
        embedded_dirs = embeddirs_fn(viewdirs) # 27，理论24，将原始输入作为嵌入向量之一


    outputs_flat = batchify(medium_mlp, netchunk)(embedded_dirs) # 使用batchify函数将输入分块，以便网络可以处理大型输入
    # # 将输出重新塑形为与原始输入相同的形状，但最后一个维度是输出的张量 sigma_bs, sigma_atten, c_med
    # outputs = torch.reshape(outputs_flat, outputs_flat.shape[-1])
    return outputs_flat


class color_mlp(nn.Module):
    def __init__(self, D=8, W=256, input_ch_views=27, output_ch=5, skips=[4]):
        """
        """
        super(color_mlp, self).__init__()
        self.D = D#指定要使用的全连接层的数量。
        self.W = W #每个全连接层的宽度（即神经元的数量）。
        self.k = 1 #输入特征的通道数。
        #self.input_ch = input_ch #视图特征的通道数。
        self.input_ch_views = input_ch_views #输出通道数
        self.skips = skips #包含跳连的层的索引列表。

        #一个模块列表，包含处理点特征的全连接层。其中包含跳连结构，在指定的层后将输入特征与该层的输出进行拼接。
        # self.pts_linears = nn.ModuleList(
        #     [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
        #                                 range(D - 1)])

        #另一个全连接层，用于增强特征。
        self.enhance_linears = nn.ModuleList(
            [nn.Linear(W, W)]
        )

        self.enhance = True
        # self.alpha = None
        # self.gamma = None
        self.gamma_0 = 2.2

        # Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(2 * input_ch_views, W // 2)]) #定义了第 10 层网络，输入维度 (27+256) = 283，输出维度 128。
        self.nviews_linears = nn.ModuleList([nn.Linear(input_ch_views, W // 2)])
        self.rgb_linear = nn.Linear(W // 2, 3) #定义了 RGB 的解码层。
        self.v_linear = nn.Linear(W // 2, 1)

        self.coeff_linears = nn.Linear(input_ch_views+1, W // 2)
        # self.gamma_linear = nn.Linear(W // 2, 3)
        # self.alpha_linear = nn.Linear(W // 2, 3)

        self.gamma_mlp1 = nn.Linear(W // 2, W // 2)
        self.gamma_mlp2 = nn.Linear(W, 3)
        self.alpha_mlp1 = nn.Linear(W // 2, W // 2)
        self.alpha_mlp2 = nn.Linear(W, 3)

    def forward(self, x):
        #input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        input_views = x
        h = input_views
        # for i, l in enumerate(self.pts_linears):
        #     h = self.pts_linears[i](h)
        #     h = F.relu(h)
        #     if i in self.skips:
        #         h = torch.cat([input_pts, h], -1)

        for i, l in enumerate(self.views_linears):
            v = self.views_linears[i](torch.cat([h, input_views], dim=-1))
            v = F.relu(v)

        # for i, l in enumerate(self.nviews_linears):
        #     c = self.nviews_linears[i](h)
        #     c = F.relu(c)

        # c = torch.sigmoid(self.rgb_linear(c))

        v = torch.sigmoid(self.v_linear(v))
        # v = F.relu(self.v_linear(v))

        # torch.clamp_max(c, 1)
        # torch.clamp_max(v, 1)

        if self.enhance:
            f = self.coeff_linears(torch.cat([h, v], dim=-1))
            f = F.relu(f)

            gamma_mid = F.relu(self.gamma_mlp1(f))
            gamma = torch.sigmoid(self.gamma_mlp2(torch.cat([f, gamma_mid], dim=-1)))

            alpha_mid = F.relu(self.alpha_mlp1(f))
            alpha = torch.sigmoid(self.alpha_mlp2(torch.cat([f, alpha_mid], dim=-1)))

            final_gamma = 1 / (gamma + self.gamma_0)

            v_enhance = (v.expand(-1, 3) / (alpha + 0.0001)) ** final_gamma
            v_enhance = self.k * v_enhance
        else:
            v_enhance = 2*v.expand(-1, 3)

        torch.clamp_max(v_enhance, 1)
        return v_enhance

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


class MediumMLP(nn.Module):
    '''
    MediumMLP 类：实现了介质 MLP，其中包含用于计算 c_med、sigma_bs 和 sigma_atten 的层。
    激活函数：使用了介质模型中提到的激活函数， density_activation、rgb_activation 和 sigma_activation。
    跳跃连接：，实现了跳跃连接（skip connections）。
    '''
    def __init__(self, input_dim = 27, net_depth_water = 8, net_width_viewdirs = 256, num_rgb_channels = 3, density_activation = nn.Softplus(),
                 rgb_activation = nn.Sigmoid() ,
                 sigma_activation = nn.Softplus() , water_bias = 0.1, skip_layer_dir = 4):
        super(MediumMLP, self).__init__()
        self.net_depth_water = net_depth_water
        self.net_width_viewdirs = net_width_viewdirs
        self.num_rgb_channels = num_rgb_channels
        self.density_activation = density_activation
        self.rgb_activation = rgb_activation
        self.sigma_activation = sigma_activation
        self.water_bias = water_bias
        self.skip_layer_dir = skip_layer_dir

        self.layers = nn.ModuleList()
        for i in range(net_depth_water):
            if i == 0:
                in_channels = input_dim
            else:
                in_channels = net_width_viewdirs
                if i % self.skip_layer_dir == 0:
                    in_channels += input_dim  # 由于拼接输入，需要增加维度
            out_channels = net_width_viewdirs
            self.layers.append(nn.Linear(in_channels, out_channels))

        self.output_layer_c_med = nn.Linear(net_width_viewdirs, num_rgb_channels)
        self.output_layer_sigma_bs = nn.Linear(net_width_viewdirs, num_rgb_channels)  # 或 num_rgb_channels
        self.output_layer_sigma_atten = nn.Linear(net_width_viewdirs, num_rgb_channels) #或 num_rgb_channels


    def forward(self, dir_enc_for_water, glo_vec=None):
        if glo_vec is None:
            dir_enc_for_water_1 = dir_enc_for_water
            # print(f"input:{dir_enc_for_water_1.tolist()}")
            for i in range(self.net_depth_water):
                if i % self.skip_layer_dir == 0 and i > 0:
                    dir_enc_for_water_1 = torch.cat([dir_enc_for_water_1, dir_enc_for_water], dim=-1)
                dir_enc_for_water_1 = self.layers[i](dir_enc_for_water_1)


                dir_enc_for_water_1 = self.density_activation(dir_enc_for_water_1)

                # print(
                #     f"Layer {i}, Max: {dir_enc_for_water_1.max()}, Min: {dir_enc_for_water_1.min()}, Mean: {dir_enc_for_water_1.mean()}")
                # # 添加调试信息
                # if torch.isnan(dir_enc_for_water_1).any():
                #     print(f"NaN detected after layer {i}")
                # if torch.isinf(dir_enc_for_water_1).any():
                #     print(f"Inf detected after layer {i}")



            c_med = self.rgb_activation(
                self.output_layer_c_med(dir_enc_for_water_1)
            )

            sigma_bs = self.sigma_activation(
                self.output_layer_sigma_bs(dir_enc_for_water_1) + self.water_bias
            )

            sigma_atten = self.sigma_activation(
                self.output_layer_sigma_atten(dir_enc_for_water_1) + self.water_bias
            )
        else:

            # 使用ReLU激活函数确保所有输出值非负
            sigma_bs = F.relu(glo_vec[..., 0:3])  # 处理散射系数部分
            sigma_atten = F.relu(glo_vec[..., 3:6])  # 处理吸收系数部分
            c_med = F.relu(glo_vec[..., 6:])  # 处理介质的RGB颜色部分
        # 返回计算结果

        # 合并所有处理后的张量
        output_tensor = torch.cat([sigma_bs, sigma_atten, c_med], dim=-1)
        return output_tensor

    def backward(self):
        print("sssss")
if __name__ =="__main__":
    Mediu = MediumMLP()
    input = torch.rand(10000,27)
    output = Mediu(input)
    print(output.shape)

    pass


