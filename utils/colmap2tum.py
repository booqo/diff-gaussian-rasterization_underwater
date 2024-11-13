import os
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R
import re

# def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
#     """读取指定字节数并解码。"""
#     data = fid.read(num_bytes)
#     return struct.unpack(endian_character + format_char_sequence, data)
#
# def read_images_binary(images_path):
#     """读取 images.bin 文件并解析相机位姿。"""
#     with open(images_path, "rb") as f:
#         num_images = read_next_bytes(f, 8, "Q")[0]
#         images = {}
#
#         for _ in range(num_images):
#             image_id = read_next_bytes(f, 8, "Q")[0]
#             qw, qx, qy, qz = read_next_bytes(f, 32, "dddd")
#             tx, ty, tz = read_next_bytes(f, 24, "ddd")
#             camera_id = read_next_bytes(f, 8, "Q")[0]
#
#             # 读取图像名称，直到遇到 '\x00'
#             image_name_bytes = bytearray()
#             while True:
#                 byte = f.read(1)
#                 if byte == b'\x00':
#                     break
#                 image_name_bytes.extend(byte)
#             image_name = image_name_bytes.decode('utf-8', errors='replace')
#
#             # 使用图像名称作为时间戳，存储位姿信息
#             images[image_name] = (tx, ty, tz, qx, qy, qz, qw)
#         return images
# def read_images_txt(images_path):
#     """读取 images.txt 文件，提取相机轨迹。"""
#     images = {}
#     with open(images_path, 'r') as f:
#         for line in f:
#             if line.startswith('#') or len(line.strip()) == 0:
#                 continue
#             elems = line.split()
#             image_name = elems[9]
#             qw, qx, qy, qz = map(float, elems[1:5])
#             tx, ty, tz = map(float, elems[5:8])
#             images[image_name] = (tx, ty, tz, qx, qy, qz, qw)
#     return images
#
# def export_to_tum(images, output_path):
#     """将解析后的数据导出为 TUM 格式。"""
#     with open(output_path, "w") as f:
#         for image_name, (tx, ty, tz, qx, qy, qz, qw) in images.items():
#             # 去除文件名中的多余空格，并确保每一行有 8 个条目
#             image_name_clean = image_name.strip()
#             line = f"{image_name_clean} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n"
#             f.write(line)
#
# def convert_colmap_to_tum(input_path, output_path):
#     """检测文件类型并转换为 TUM 格式。"""
#     if input_path.endswith(".bin"):
#         print(f"读取二进制文件：{input_path}")
#         images = read_images_binary(input_path)
#     elif input_path.endswith(".txt"):
#         print(f"读取文本文件：{input_path}")
#         images = read_images_txt(input_path)
#     else:
#         raise ValueError("输入文件格式不支持，请提供 .bin 或 .txt 文件。")
#
#     export_to_tum(images, output_path)
#     print(f"TUM 轨迹文件已导出为 {output_path}")

# 使用示例

def read_images_txt(images_path):
    """读取 COLMAP 的 images.txt 文件，并提取位姿数据。"""
    images = {}
    with open(images_path, 'r') as f:
        lines = f.readlines()
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # 跳过注释行和空行
            if line.startswith('#') or len(line) == 0:
                i += 1
                continue

            # 解析第一行中的位姿数据
            elems = line.split()
            if len(elems) < 10:
                i += 2  # 跳过不完整的行及其对应的 2D 点行
                continue

            # 从位姿数据中提取时间戳和位姿
            image_name = elems[0]  # 使用图像名称作为时间戳
            qw, qx, qy, qz = map(float, elems[1:5])
            tx, ty, tz = map(float, elems[5:8])
            # 创建旋转矩阵
            rotation = R.from_quat([qx, qy, qz, qw])
            R_matrix = rotation.as_matrix()
            # COLMAP 的平移向量
            t = np.array([tx, ty, tz])

            # 计算相机在世界坐标系中的位置 C = -R.T @ t
            camera_center = -np.dot(R_matrix.T, t)
            tx, ty, tz = camera_center[0:3]
            # 存储解析后的数据
            images[image_name] = (tx, ty, tz, qx, qy, qz, qw)

            # 跳过接下来的 2D 点信息行
            i += 2  # 每个图像有两行数据，需要跳过第二行
    # 根据图像名称排序
    sorted_images = dict(sorted(images.items(), key=lambda x: extract_number(x[0])))
    return sorted_images

def extract_number(filename):
    """从文件名中提取数字部分，用于排序。"""
    match = re.search(r'\d+', filename)  # 匹配文件名中的数字
    return int(match.group()) if match else float('inf')  # 提取数字部分

def export_to_tum(images, output_path):
    """将解析后的数据导出为 TUM 格式。"""
    with open(output_path, "w") as f:
        for image_name, (tx, ty, tz, qx, qy, qz, qw) in images.items():
            # 严格格式化为 8 个条目
            line = f"{image_name} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n"
            f.write(line)

def convert_colmap_to_tum(input_path, output_path):
    """将 COLMAP 的 images.txt 转换为 TUM 格式。"""
    images = read_images_txt(input_path)
    export_to_tum(images, output_path)
    print(f"TUM 轨迹文件已导出为 {output_path}")


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_tum_trajectory(file_path):
    """读取 TUM 轨迹文件，并返回位置信息。"""
    positions = []  # 存储 (tx, ty, tz)

    with open(file_path, 'r') as f:
        for line in f:
            elems = line.strip().split()
            if len(elems) != 8:
                continue  # 跳过格式不符的行

            tx, ty, tz = map(float, elems[1:4])
            positions.append((tx, ty, tz))

    return positions


def set_equal_aspect(ax, data):
    """设置 3D 图的 XYZ 轴比例一致。"""
    max_range = np.array([np.max(data[:, 0]) - np.min(data[:, 0]),
                          np.max(data[:, 1]) - np.min(data[:, 1]),
                          np.max(data[:, 2]) - np.min(data[:, 2])]).max() / 2.0

    mid_x = (np.max(data[:, 0]) + np.min(data[:, 0])) * 0.5
    mid_y = (np.max(data[:, 1]) + np.min(data[:, 1])) * 0.5
    mid_z = (np.max(data[:, 2]) + np.min(data[:, 2])) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_trajectory(positions):
    """绘制 3D 轨迹，并确保轴比例一致。"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 将位置数据转换为 NumPy 数组以便操作
    data = np.array(positions)

    # 分离出 X、Y、Z 坐标
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]

    # 绘制轨迹
    ax.plot(xs, ys, zs, label='Trajectory', marker='o')
    # 绘制散点图
    ax.scatter(xs, ys, zs, c='r', marker='o', s=10, label='Positions')

    # 设置轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('TUM Trajectory with Equal Axis Scaling')
    ax.legend()

    # 设置 XYZ 轴比例一致
    set_equal_aspect(ax, data)

    plt.show()

input_file = "I:\\0\\images.txt"  # 或 "sparse/0/images.txt"
output_file = "I:\\0\\trajectory_tum.txt"
convert_colmap_to_tum(input_file, output_file)

# 使用示例
trajectory_file = "I:\\0\\trajectory_tum.txt"  # 替换为你的 TUM 文件路径
positions = read_tum_trajectory(trajectory_file)
plot_trajectory(positions)



