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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images" #定义图像路径的
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self._depths = ""
        self.train_test_exp = False
        super().__init__(parser, "Loading Parameters", sentinel)


    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # 定义优化器相关的参数，用于控制训练过程中的学习率、密集化策略等。
        self.iterations = 30_000  # 总共进行的迭代次数。

        # 初始化位置（Position）参数的学习率。
        self.position_lr_init = 0.00016  # 初始学习率。
        self.position_lr_final = 0.0000016  # 最终学习率，逐渐减小到此值。
        self.position_lr_delay_mult = 0.01  # 学习率延迟因子，控制学习率更新的速度。
        self.position_lr_max_steps = 30_000  # 位置学习率调整的最大步数。

        # 其他参数的学习率设置。
        self.feature_lr = 0.0025  # 特征（Feature）参数的学习率。
        self.opacity_lr = 0.05  # 不透明度（Opacity）参数的学习率。
        self.scaling_lr = 0.005  # 缩放（Scaling）参数的学习率。
        self.rotation_lr = 0.001  # 旋转（Rotation）参数的学习率。

        # 控制训练过程中的密集化策略。
        self.percent_dense = 0.01  # 初始密集化比例，即模型从多少密集度开始。
        self.lambda_dssim = 0.2  # DSSIM（结构相似性损失）的权重因子。

        # 密集化和重置的时间间隔。
        self.densification_interval = 100  # 每 100 次迭代后进行一次密集化。
        self.opacity_reset_interval = 3000  # 每 3000 次迭代后重置不透明度。

        # 控制密集化的范围和阈值。
        self.densify_from_iter = 500  # 从第 500 次迭代开始密集化。
        self.densify_until_iter = 15_000  # 到第 15,000 次迭代结束密集化。
        self.densify_grad_threshold = 0.0002  # 密集化的梯度阈值，小于此值时停止密集化。

        # 训练过程中是否使用随机背景。
        self.random_background = False  # 如果为 True，则使用随机背景颜色进行训练。

        #高斯体各向异性梯度与不透明度阈值，以及缩放阈值
        self.color_grad = 0.1
        self.opacity_grad = 0.5
        self.scale_trace = 20

        # 调用父类构造函数，初始化 "Optimization Parameters" 模块。
        super().__init__(parser, "Optimization Parameters")


# def get_combined_args(parser : ArgumentParser):
#     cmdlne_string = sys.argv[1:]
#     cfgfile_string = "Namespace()"
#     args_cmdline = parser.parse_args(cmdlne_string)
#
#     try:
#         cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
#         print("Looking for config file in", cfgfilepath)
#         with open(cfgfilepath) as cfg_file:
#             print("Config file found: {}".format(cfgfilepath))
#             cfgfile_string = cfg_file.read()
#     except TypeError:
#         print("Config file not found at")
#         pass
#     args_cfgfile = eval(cfgfile_string)
#
#     merged_dict = vars(args_cfgfile).copy()
#     for k,v in vars(args_cmdline).items():
#         if v != None:
#             merged_dict[k] = v
#     return Namespace(**merged_dict)
