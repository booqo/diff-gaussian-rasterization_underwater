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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>
#include <stdio.h>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA_underwater(
	const torch::Tensor& background, //1
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& medium_rgb,
    const torch::Tensor& medium_bs,
    const torch::Tensor& medium_attn,
    const torch::Tensor& colors_enhance,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug) //24
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_image = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts); //初始化为0
  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts); //初始化为0
  torch::Tensor out_med = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts); //初始化为0
  torch::Tensor n_touched = torch::full({H, W}, 0, int_opts); //n_touched gaussian points touched in one pixel
  
//   torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts); //初始化为0
//   float* out_invdepthptr = nullptr;
  float* out_depthptr = nullptr;

  //out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  //out_invdepthptr = out_invdepth.data_ptr<float>();
  out_depthptr = out_depth.data_ptr<float>();

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(  //forward函数
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data_ptr<float>(),
		W, H,
		means3D.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data_ptr<float>(),   //colors precomputed
		opacity.contiguous().data_ptr<float>(),
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data_ptr<float>(),
		(float3 *)medium_rgb.contiguous().data_ptr<float>(),   // H W 3
		(float3 *)medium_bs.contiguous().data_ptr<float>(),   // H W 3
		(float3 *)medium_attn.contiguous().data_ptr<float>(),  // H W 3
		(float3 *)colors_enhance.contiguous().data_ptr<float>(),  // H W 3
		out_image.contiguous().data_ptr<float>(),  // 3 H W
		out_color.contiguous().data_ptr<float>(),  // 3 H W
		out_med.contiguous().data_ptr<float>(),  // 3 H W
		out_depthptr,    // 1 H W
		n_touched.contiguous().data_ptr<int>(),  // H W
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		//out_invdepthptr,
		antialiasing,
		radii.contiguous().data_ptr<int>(),
		debug);
  }
  //printf("ffff");
  return std::make_tuple(rendered, out_image , out_color, out_med, radii, geomBuffer, binningBuffer, imgBuffer, out_depth, n_touched );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
	torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA_underwater(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& medium_rgb, //介质颜色
	const torch::Tensor& medium_bs, //介质 \sigma bs
	const torch::Tensor& medium_attn,
	const torch::Tensor& colors_enhance,
    const torch::Tensor& dL_dout_color_image,
	const torch::Tensor& dL_dout_color_clr,
	const torch::Tensor& dL_dout_color_cmed,
	const torch::Tensor& dL_dout_depth,
	//const torch::Tensor& dL_dout_invdepth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color_image.size(1);
  const int W = dL_dout_color_image.size(2);
  //printf("P: %d, H: %d, W: %d\n", P, H, W);
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_ddepths = torch::zeros({0, 1}, means3D.options());

  
  
  torch::Tensor dL_dmedium_rgb = torch::zeros({H, W, 3}, means3D.options());
  torch::Tensor dL_dmedium_bs = torch::zeros({H, W, 3}, means3D.options());
  torch::Tensor dL_dmedium_attn = torch::zeros({H, W, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dcolors_enhance = torch::zeros({H, W, 3}, means3D.options());

  
//   float* dL_dinvdepthsptr = nullptr;
//   float* dL_dout_invdepthptr = nullptr;
//   if(dL_dout_invdepth.size(0) != 0)
//   {
// 	dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
// 	dL_dinvdepths = dL_dinvdepths.contiguous();
// 	dL_dinvdepthsptr = dL_dinvdepths.data_ptr<float>();
// 	dL_dout_invdepthptr = dL_dout_invdepth.data_ptr<float>();  //P 1
//   }

  float* dL_ddepthsptr = nullptr;
  float* dL_dout_depthptr = nullptr;

  if(dL_dout_depth.size(0) != 0)
  {
	dL_ddepths = torch::zeros({P, 1}, means3D.options());
	dL_ddepths = dL_ddepths.contiguous();
	dL_ddepthsptr = dL_ddepths.data_ptr<float>();  // p 1
	dL_dout_depthptr = dL_dout_depth.contiguous().data_ptr<float>();  //1 H W
  }

  //printf("part1\n");
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data_ptr<float>(),
	  W, H, 
	  means3D.contiguous().data_ptr<float>(),
	  sh.contiguous().data_ptr<float>(),
	  colors.contiguous().data_ptr<float>(),
	  opacities.contiguous().data_ptr<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data_ptr<float>(),
	  viewmatrix.contiguous().data_ptr<float>(),
	  projmatrix.contiguous().data_ptr<float>(),
	  campos.contiguous().data_ptr<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data_ptr<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  (float3 *) medium_rgb.contiguous().data_ptr<float>(), //介质颜色
	  (float3 *) medium_bs.contiguous().data_ptr<float>(), //介质 \sigma bs
	  (float3 *) medium_attn.contiguous().data_ptr<float>(),
	  (float3 *) colors_enhance.contiguous().data_ptr<float>(),
	  dL_dout_color_image.contiguous().data_ptr<float>(),
	  dL_dout_color_clr.contiguous().data_ptr<float>(),
	  dL_dout_color_cmed.contiguous().data_ptr<float>(),
	  dL_dout_depthptr,
	  //dL_dout_invdepthptr,
	  dL_dmeans2D.contiguous().data_ptr<float>(),
	  dL_dconic.contiguous().data_ptr<float>(),
	  dL_dopacity.contiguous().data_ptr<float>(),
	  //dL_dinvdepthsptr,
	  dL_ddepthsptr,
	  dL_dmeans3D.contiguous().data_ptr<float>(),
	  dL_dcov3D.contiguous().data_ptr<float>(),
	  dL_dmedium_rgb.contiguous().data_ptr<float>(),
	  dL_dmedium_bs.contiguous().data_ptr<float>(), 
      dL_dmedium_attn.contiguous().data_ptr<float>(), 
	  dL_dcolors.contiguous().data_ptr<float>(),  //sh
	  dL_dcolors_enhance.contiguous().data_ptr<float>(),
	  dL_dsh.contiguous().data_ptr<float>(),
	  dL_dscales.contiguous().data_ptr<float>(),
	  dL_drotations.contiguous().data_ptr<float>(),
	  antialiasing,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors , dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dmedium_rgb, dL_dmedium_bs, 
                dL_dmedium_attn, dL_dcolors_enhance , dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		present.contiguous().data_ptr<bool>());
  }
  
  return present;
}
