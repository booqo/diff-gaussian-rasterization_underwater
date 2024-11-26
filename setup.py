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

            include_dirs=[
                "/usr/local/cuda-11.8/include",  # CUDA include 路径
                "/media/asus/32A6EEC5E12B6F66/anaconda3/envs/gaussian_splatting/include/python3.7m"  # Python 头文件路径
            ],
            library_dirs=[
                "/usr/local/cuda-11.8/lib64",  # CUDA lib 路径
                "/media/asus/32A6EEC5E12B6F66/anaconda3/envs/gaussian_splatting/lib"  # Python 库路径
            ],

            extra_compile_args={
                "nvcc": [
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
                ]  # 额外的编译参数，包含第三方库 glm
            }
            # extra_compile_args={
            #     "nvcc": [
            #         "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),  # 包含路径
            #         "-g",  # 启用 NVCC 的调试符号
            #         "-G",  # 生成调试信息
            #         "-lineinfo",  # 行号信息
            #     ],
            #     "cxx": [
            #         "-g",  # 为 C++ 编译器启用调试信息
            #     ]
            # }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension  # 指定扩展构建方式
    }
)

