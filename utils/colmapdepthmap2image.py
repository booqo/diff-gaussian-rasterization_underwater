#!/usr/bin/env python

# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# from https://github.com/colmap/colmap/blob/main/scripts/python/read_write_dense.py
# All rights reserved.

import argparse
import os
import struct

import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为 SimHei，以支持中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 读取二进制数组并转换为NumPy数组
def read_array(path):
    """
    从二进制文件中读取深度图或法线图数据，并将其转换为NumPy数组。
    文件的前几行包含图像的宽、高和通道数信息，后面是图像数据。

    参数：
    path - 二进制文件的路径

    返回：
    转换后的NumPy数组
    """
    with open(path, "rb") as fid:
        # 从文件头中读取宽度、高度和通道数
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        # 跳过文件头，找到三个分隔符 '&' 后的图像数据
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        # 读取图像数据并转换为NumPy数组
        array = np.fromfile(fid, np.float32)

    # 将数组重塑为 (宽, 高, 通道数) 形式
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


# 将NumPy数组写入二进制文件
def write_array(array, path):
    """
    将NumPy数组以特定格式写入二进制文件。

    参数：
    array - 需要写入的NumPy数组
    path - 输出文件的路径
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    # 写入图像的元数据信息 (宽&高&通道数&)
    with open(path, "w") as fid:
        fid.write(f"{width}&{height}&{channels}&")

    # 以二进制格式写入图像数据
    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        # 小端字节序的二进制数据打包
        byte_data = struct.pack("<" + "f" * len(data_list), *data_list)
        fid.write(byte_data)


# 命令行参数解析
def parse_args():
    """
    解析命令行参数，用于指定深度图和法线图的路径。

    返回：
    解析后的命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--depth_map", help="深度图的路径", type=str, required=True
    )
    parser.add_argument(
        "-n", "--normal_map", help="法线图的路径", type=str, required=True
    )
    parser.add_argument(
        "--min_depth_percentile",
        help="深度图最小可视化百分位",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--max_depth_percentile",
        help="深度图最大可视化百分位",
        type=float,
        default=95,
    )
    args = parser.parse_args()
    return args


# 主函数
def main():
    args = parse_args()

    # 检查最小百分位是否小于等于最大百分位
    if args.min_depth_percentile > args.max_depth_percentile:
        raise ValueError("min_depth_percentile 应该小于或等于 max_depth_percentile。")

    # 检查文件是否存在
    if not os.path.exists(args.depth_map):
        raise FileNotFoundError(f"文件未找到: {args.depth_map}")
    if not os.path.exists(args.normal_map):
        raise FileNotFoundError(f"文件未找到: {args.normal_map}")

    # 读取深度图和法线图
    depth_map = read_array(args.depth_map)
    # depth_map = np.clip(depth_map, 0, 1)  # 如果数据是浮点数并且超出了 [0, 1] 范围
    # # 如果数据类型是整数并且应该在 [0, 255]，可以这样处理：
    depth_map = np.clip(depth_map, 0, 255).astype(np.uint8)
    normal_map = read_array(args.normal_map)
    normal_map = np.clip(normal_map, 0, 255)  # 如果数据是浮点数并且超出了 [0, 1] 范围

    # 根据百分位值调整深度图的范围
    min_depth, max_depth = np.percentile(
        depth_map, [args.min_depth_percentile, args.max_depth_percentile]
    )
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth



    # 可视化深度图
    plt.figure()
    plt.imshow(depth_map)
    plt.title("深度图")

    # 显示原始深度图（归一化后的灰度图）
    plt.figure()
    plt.imshow(depth_map, cmap='gray')  # 使用灰度色彩映射
    plt.title("深度图的灰度图表示")
    plt.colorbar(label="深度值")

    er_ = (depth_map - np.min(depth_map)).astype(np.uint16)
    er = np.max(depth_map) - np.min(depth_map)
    depth_map_normalized = (255*er_ / er).astype(np.uint8)
    cv2.imshow("depth",depth_map_normalized)
    # cv2.waitKey(0)



    # 可视化法线图
    plt.figure()
    plt.imshow(normal_map)
    plt.title("法线图")

    plt.show()


# 程序入口
if __name__ == "__main__":
    main()
