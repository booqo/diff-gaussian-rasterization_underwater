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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup(
    name="diff_gaussian_rasterization_underwater",  # 包名称
    packages=['diff_gaussian_rasterization_underwater'],  # 包含的模块
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_underwater._C",  # 扩展模块名称
            sources=[  # 源文件列表
                "cuda_rasterizer/rasterizer_impl.cu",  # 核心 CUDA 实现
                "cuda_rasterizer/forward.cu",         # 前向传播 CUDA 实现
                "cuda_rasterizer/backward.cu",        # 反向传播 CUDA 实现
                "rasterize_points.cu",                # 点渲染 CUDA 文件
                "ext.cpp"                             # C++ 与 Python 的绑定实现
            ],
            extra_compile_args={
                "nvcc": [
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
                ]  # 额外的编译参数，包含第三方库 glm
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension  # 指定扩展构建方式
    }
)

