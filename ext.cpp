/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {// 定义一个 Python 模块并注册 C++ 函数
  m.def("rasterize_gaussians_underwater"  // Python 中的函数名
  , &RasterizeGaussiansCUDA_underwater,              // 对应的 C++ 函数实现
   "Render underwater Gaussian distributions");         //

  m.def("rasterize_gaussians_backward_underwater",
   &RasterizeGaussiansBackwardCUDA_underwater,
   "Compute gradients for Gaussian rasterization (backward pass)");
  m.def("mark_visible", &markVisible);
}

//实现了三个函数，光栅化，梯度反向传播的光栅化，标记可视化