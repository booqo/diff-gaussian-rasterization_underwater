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
import numpy as np

# from test import result
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene import get_gradients as gg

class GaussianModel:

    def setup_functions(self):#用于设置一些激活函数和变换函数
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):#构建协方差矩阵，该函数接受 scaling（尺度）、scaling_modifier（尺度修正因子）、rotation（旋转）作为参数
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)# \Sigma = R S S^T R^T
            symm = strip_symmetric(actual_covariance)
            return symm#最终返回对称的协方差矩阵。 (N, 6)
        
        self.scaling_activation = torch.exp#将尺度激活函数设置为指数函数。
        self.scaling_inverse_activation = torch.log#将尺度逆激活函数设置为对数函数。

        self.covariance_activation = build_covariance_from_scaling_rotation#将协方差激活函数设置为上述定义的 build_covariance_from_scaling_rotation 函数。

        self.opacity_activation = torch.sigmoid#将不透明度激活函数设置为 sigmoid 函数。
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize#将协方差激活函数设置为上述定义的 build_covariance_from_scaling_rotation 函数。


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0#球谐阶数
        self.max_sh_degree = sh_degree  #最大球谐阶数
        # 存储不同信息的张量（tensor）
        self._xyz = torch.empty(0)#空间位置
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        # self._rgb = torch.empty(0) ->r>%
        self._scaling = torch.empty(0)  #椭球的形状尺度
        self._rotation = torch.empty(0) #椭球的旋转
        self._opacity = torch.empty(0)  #不透明度
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None  #初始化优化器为 None。
        self.percent_dense = 0  #初始化百分比密度为0。
        self.spatial_lr_scale = 0 #初始化空间学习速率缩放为0。
        self.setup_functions() #调用 setup_functions 方法设置各种激活和变换函数

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float): #用于从给定的点云数据 pcd 创建对象的初始化状态。
        self.spatial_lr_scale = spatial_lr_scale    # scene_info.nerf_normalization
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()     # 稀疏点云的3D坐标，(N, 3)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())   # （N，3），球谐函数表示的颜色特征，公式是什么
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # （N，3，(msd+1)**2）
        # 最大球谐阶数msd决定使用 (msd+1)**2 项拟合颜色
        features[:, :3, 0 ] = fused_color   # 全选第一维N，选第二维的前三个rgb，选第三维的第一个元素 ；将颜色转换为以球谐（SH）形式表示的特征
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) #使用KNN来得到初始化一个合适的高斯缩放大小
        '''
        dist2的大小应该是(N,)。
        首先可以明确的是这句话用来初始化scale，且scale（的平方）不能低于1e-7。
        我阅读了一下submodules/simple-knn/simple_knn.cu，大致猜出来了这个是什么意思。
        (cu文件里面一句注释都没有，读起来真折磨！)
        distCUDA2函数由simple_knn.cu的SimpleKNN::knn函数实现。
        KNN意思是K-Nearest Neighbor，即求每一点最近的K个点。
        simple_knn.cu中令k=3，求得每一点最近的三个点距该点的平均距离。
        算法并没有实现真正的KNN，而是近似KNN。
        原理是把3D空间中的每个点用莫顿编码（Morton Encoding）转化为一个1D坐标
        （详见https://www.fwilliams.info/point-cloud-utils/sections/morton_coding/，
        用到了能够填满空间的Z曲线），
        然后对1D坐标进行排序，从而确定离每个点最近的三个点。
        simple_knn.cu实际上还用了一种加速策略，是将点集分为多个大小为1024的块（box），
        在每个块内确定3个最近邻居和它们的平均距离。用平均距离作为Gaussian的scale。
        '''
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3) #log(dist2^0.5)
        '''因为scale的激活函数是exp，所以这里存的也不是真的scale，而是ln(scale)。
           注意dist2其实是距离的平方，所以这里要开根号。
           repeat(1, 3)标明三个方向上scale的初始值是相等的。
           scales的大小：(N, 3)'''
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")# (N, 4) wxyz
        rots[:, 0] = 1 # 初始化旋转四元数为1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        '''
        不透明度在经历sigmoid前的值，大小为(N, 1)
        inverse_sigmoid是sigmoid的反函数，等于ln(x / (1 - x))。
        这里把不透明度初始化为0.1（经历激活函数以后为0.1，原因不明），但存储的时候要取其经历sigmoid前的值：
        inverse_sigmoid(0.1) = -2.197
        '''

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))    # 高斯椭球体中心坐标，(N, 3)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))   # RGB三个通道的直流分量，(N, 1, 3)，代表视角无关的初始色，l=0，0阶
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))  # RGB三个通道的高阶分量，(N, (最大球谐阶数 + 1)² - 1, 3)
        self._scaling = nn.Parameter(scales.requires_grad_(True))   # 尺寸缩放，(N, 3), ln(真正的scale)
        self._rotation = nn.Parameter(rots.requires_grad_(True))    # 四元数，（N, 4）
        self._opacity = nn.Parameter(opacities.requires_grad_(True))# 不透明度（经过sigmoid之前），(N, 1)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # 最大半径？初始化为0，大小为(N,)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense # 百分比密度, 初始为 0.01; 控制Gaussian的密度，在`densify_and_clone`中被使用
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")    # 各个点坐标的累积梯度，（N，1）
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # 各个点的分母，（N，1）

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # {'params':[self._rgb],lr}
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):#更新Gaussian坐标xyz的学习率
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):#这个方法的目的是从PLY文件中加载各种数据，并将这些数据存储为类中的属性，以便后续的操作和训练。
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # 将新的密集化点的相关特征保存在一个字典中。
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        # 新增Gaussian，把新属性添加到优化器中
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]  # 获取初始点的数量。
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")  # 创建一个长度为初始点数量的梯度张量，并将计算得到的梯度填充到其中。
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)  # 创建一个掩码，标记那些梯度大于等于指定阈值的点。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)
        # 一步过滤掉那些缩放（scaling）大于一定百分比的场景范围的点。
        '''
        被分裂的Gaussians满足两个条件：
        1. （平均）梯度过大；
        2. 在某个方向的最大缩放大于一个阈值。
        参照论文5.2节“On the other hand...”一段，大Gaussian被分裂成两个小Gaussians，
        其放缩被除以φ=1.6，且位置是以原先的大Gaussian作为概率密度函数进行采样的。
        '''

        # 为每个点生成新的样本，其中 stds 是点的缩放，means 是均值。
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)  # 使用均值和标准差生成样本。
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)  # 为每个点构建旋转矩阵，并将其重复 N 次。
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N,
                                                                                                              1)  # 将旋转后的样本点添加到原始点的位置
        # 算出随机采样出来的新坐标
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))  # 生成新的缩放参数
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)  # 将旋转矩阵重复 N 次。
        # 将原始点的特征重复 N 次。
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # 调用另一个方法 densification_postfix，该方法对新生成的点执行后处理操作（此处跟densify_and_clone一样）。
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling,
                                   new_rotation)

        # 创建一个修剪（pruning）的过滤器，将新生成的点添加到原始点的掩码之后。
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # 根据修剪过滤器，修剪模型中的一些参数。
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False) #建一个掩码，标记满足梯度条件的点。具体来说，对于每个点，计算其梯度的L2范数，如果大于等于指定的梯度阈值，则标记为True，否则标记为False。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # 提取出大于阈值`grad_threshold`且缩放参数较小（小于self.percent_dense * scene_extent）的Gaussians，在下面进行克隆

        # 根据掩码选取符合条件的点的其他特征，如颜色、透明度、缩放和旋转等。
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    # 执行密集化和修剪操作
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom  # 计算平均梯度
        grads[grads.isnan()] = 0.0  # 将梯度中的 NaN（非数值）值设置为零，以处理可能的数值不稳定性。

        self.densify_and_clone(grads, max_grad, extent)  # 对under reconstruction的区域进行稠密化和复制操作
        self.densify_and_split(grads, max_grad, extent)  # 对over reconstruction的区域进行稠密化和分割操作

        # 接下来移除一些Gaussians，它们满足下列要求中的一个：
        # 1. 接近透明（不透明度小于min_opacity）
        # 2. 在某个相机视野里出现过的最大2D半径大于屏幕（像平面）大小
        # 3. 在某个方向的最大缩放大于0.1 * extent（也就是说很长的长条形也是会被移除的）
        prune_mask = (self.get_opacity < min_opacity).squeeze()  # 创建一个掩码，标记那些透明度小于指定阈值的点。.squeeze() 用于去除掩码中的单维度。
        if max_screen_size:  # 如何设置了相机的范围，
            big_points_vs = self.max_radii2D > max_screen_size  # 创建一个掩码，标记在图像空间中半径大于指定阈值的点。
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent  # 创建一个掩码，标记在世界空间中尺寸大于指定阈值的点。
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs),
                                          big_points_ws)  # 将这两个掩码与先前的透明度掩码进行逻辑或操作，得到最终的修剪掩码。
        self.prune_points(prune_mask)  # 根据修剪掩码，修剪模型中的一些参数。

        torch.cuda.empty_cache()  # 清理 GPU 缓存，释放一些内存

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def deletepoint(self, viewpoint, max_color_grad, max_opacity,max_scale = 20):
        xyz_coords = self.get_xyz
        shs_coeffs = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree + 1) ** 2)
        opacity = self.get_opacity
        camera_centers = torch.stack([cam.camera_center for cam in viewpoint])
        directions = camera_centers[None, :, :] - xyz_coords[:, None, :]
        directions_normalized = directions / directions.norm(dim=-1, keepdim=True)
        sh_degree = self.active_sh_degree
        scaling = self.get_scaling # 获得协方差矩阵的相对大小
        scaleAddTrace = gg.compute_det_trace(scaling)
        print(scaleAddTrace)

        gradients = gg.compute_gradient(sh_degree, shs_coeffs, directions_normalized) # 获得球鞋函数颜色梯度
        print(gradients)
        selected_color_grad_mask = torch.where(torch.norm(gradients, dim=-1) <= max_color_grad, True, False)
        selected_color_grad_mask = selected_color_grad_mask.to('cuda:0')
        selected_opacity_mask = torch.where(torch.norm(opacity, dim=-1) <= max_opacity, True, False)
        selected_scale_mask = torch.where(torch.norm(scaleAddTrace, dim=-1) <= max_scale, True, False)
        elected_mask = torch.where(selected_color_grad_mask & selected_opacity_mask & selected_scale_mask, True, False)
        self.prune_points(elected_mask)
        torch.cuda.empty_cache()

# if __name__ == "__main__":
#     test = torch.randn(10, 3)
#     result = gg.compute_det_trace(test)
