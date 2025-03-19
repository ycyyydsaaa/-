import os
import glob
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFile
import socket
import datetime
import time

ImageFile.MAXBLOCK = 1024 * 1024


def read_and_convert_images(input_dir, output_dir, target_format='PNG', target_mode='RGB', target_size=None):
    """
    读取 input_dir 内所有 JPG 图像，转换颜色模式和尺寸，并保存为 target_format 格式。
    如果文件已经转换，则跳过。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有 JPG 文件路径
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    print(f"共找到 {len(image_paths)} 张 JPG 图像，开始检查是否需要转换...")

    processed_paths = []
    skipped_count = 0

    for img_path in image_paths:
        try:
            base_name = os.path.basename(img_path)
            name, _ = os.path.splitext(base_name)
            out_path = os.path.join(output_dir, name + '.' + target_format.lower())

            # **检查是否已经存在转换后的文件**
            if os.path.exists(out_path):
                print(f"✅ 已转换，跳过: {out_path}")
                skipped_count += 1
                processed_paths.append(out_path)
                continue  # 直接跳过当前循环

            with Image.open(img_path) as img:
                # 转换颜色模式
                if img.mode != target_mode:
                    img = img.convert(target_mode)
                # 调整尺寸（如果设置）
                if target_size:
                    img = img.resize(target_size, Image.ANTIALIAS)

                # 保存新格式的图像
                img.save(out_path, target_format)
                processed_paths.append(out_path)
                print(f"✅ 处理完成: {out_path}")

        except Exception as e:
            print(f"❌ 处理 {img_path} 时出错：{e}")

    print(f"🎯 处理完成，总共 {len(processed_paths)} 张图片，其中 {skipped_count} 张已存在，跳过。")
    return processed_paths


def pad_to_square(img):
    """将图像填充为正方形，保持原有长宽比，避免拉伸变形。"""
    h, w = img.shape[:2]
    size = max(h, w)  # 选择较大的边作为目标边长

    # 计算上下和左右需要填充的像素
    top_pad = (size - h) // 2
    bottom_pad = size - h - top_pad
    left_pad = (size - w) // 2
    right_pad = size - w - left_pad

    # 使用黑色填充（也可以设置为其他颜色
    padded_img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad,
                                    borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_img


def apply_clahe(image):
    """
    对输入的彩色图像应用 CLAHE 均衡化，先转换到 LAB 颜色空间，只处理亮度通道。
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    image_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return image_eq


def denoise_image(image):
    """
    对图像进行噪声去除，采用非局部均值去噪 (Non-Local Means Denoising)处理。
    """
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image


def enhance_image(image):
    """
    对图像进行简单的锐化增强处理。
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    enhanced = cv2.filter2D(image, -1, kernel)
    return enhanced


def process_image_in_memory(img):
    # 1. 将图像填充为正方形，避免拉伸变形
    img_square = pad_to_square(img)

    # 2.亮度均衡（CLAHE）
    img_eq = apply_clahe(img_square)

    # 3.噪声去除
    img_denoised = denoise_image(img_eq)

    # 4.图像增强（锐化）
    img_enhanced = enhance_image(img_denoised)

    return img_enhanced


def pad_to_same_height(img1, img2):
    """
    不缩放，只对高度较小的图像进行上下黑边填充，
    使得两张图像的高度相同。
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 以更大的高度作为目标
    target_height = max(h1, h2)

    # 填充第1张图像
    if h1 < target_height:
        # 需要填充多少像素
        pad_total = target_height - h1
        # 也可以只填在下方，这里示例平均填充到上下
        top_pad = pad_total // 2
        bottom_pad = pad_total - top_pad
        img1 = cv2.copyMakeBorder(img1, top_pad, bottom_pad, 0, 0,
                                  borderType=cv2.BORDER_CONSTANT,
                                  value=(0, 0, 0))

    # 填充第2张图像
    if h2 < target_height:
        pad_total = target_height - h2
        top_pad = pad_total // 2
        bottom_pad = pad_total - top_pad
        img2 = cv2.copyMakeBorder(img2, top_pad, bottom_pad, 0, 0,
                                  borderType=cv2.BORDER_CONSTANT,
                                  value=(0, 0, 0))

    return img1, img2


def pair_images_and_concat(image_list, output_dir, concat_axis=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    paired = [image_list[i:i + 2] for i in range(0, len(image_list), 2)]
    for i, pair in enumerate(paired):
        if len(pair) == 2:
            img1 = cv2.imread(pair[0])
            img2 = cv2.imread(pair[1])
            if img1 is None or img2 is None:
                print(f"无法读取图像 {pair[0]} 或 {pair[1]}，跳过拼接")
                continue
            print(f"拼接前尺寸: {pair[0]} -> {img1.shape}, {pair[1]} -> {img2.shape}")

            # 处理图像
            img1_processed = process_image_in_memory(img1)
            img2_processed = process_image_in_memory(img2)

            # 用填充方式统一高度
            img1_padded, img2_padded = pad_to_same_height(img1_processed, img2_processed)
            print(f"填充后尺寸: {pair[0]} -> {img1_padded.shape}, {pair[1]} -> {img2_padded.shape}")

            concatenated = np.concatenate((img1_padded, img2_padded), axis=concat_axis)

            # 以拼接时间命名
            now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            out_path = os.path.join(output_dir, f"{now}.png")
            cv2.imwrite(out_path, concatenated)
            print(f"保存拼接图像：{out_path}")
        else:
            print(f"最后一组图像不足两张，跳过。")


def process_spliced_paths(spliced_paths, output_dir):
    image_paths = spliced_paths.split('|')
    final_output_dir = os.path.join(output_dir, 'paired')
    if not os.path.exists(final_output_dir):
        try:
            os.makedirs(final_output_dir)
        except PermissionError:
            print(f"没有权限创建目录 {final_output_dir}")
            return None
    if len(image_paths) >= 2:
        pair_images_and_concat(image_paths, final_output_dir, concat_axis=1)
    else:
        print("处理后的图像数量不足，无法进行拼接")

    # 选择处理好的文件夹中的最后一张图片返回给前端
    if os.listdir(final_output_dir):
        all_images = os.listdir(final_output_dir)
        all_images.sort()
        final_processed_image_path = os.path.join(final_output_dir, all_images[-1])
        return final_processed_image_path
    else:
        print("没有拼接后的图像，无法返回路径")
        return None


if __name__ == '__main__':
    # 创建 socket 对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 绑定地址和端口（这里假设端口为 8888，你可以根据需要修改）
    server_address = ('localhost', 8888)
    server_socket.bind(server_address)

    # 开始监听
    server_socket.listen(1)
    print('等待连接...')

    print(f"当前工作目录: {os.getcwd()}")

    try:
        while True:
            # 接受连接
            client_socket, client_address = server_socket.accept()
            print(f'连接来自: {client_address}')

            try:
                client_socket.settimeout(10)  # 设置一个超时时间，例如 10 秒
                # 接收客户端发送的拼接图像路径
                data = client_socket.recv(1024).decode('utf-8')
                # 提取拼接路径
                start_index = data.find('paths=')
                if start_index != -1:
                    spliced_paths = data[start_index + len('paths='):].split('\r\n')[0]
                else:
                    print("未找到有效的拼接路径")
                    continue

                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                final_processed_image_path = process_spliced_paths(spliced_paths, output_dir)
                if final_processed_image_path:
                    # 将处理后的图像路径发送回客户端
                    client_socket.send(final_processed_image_path.encode('utf-8'))
            except socket.timeout:
                print(f"客户端 {client_address} 连接超时，已关闭连接。")
            except (socket.error, BrokenPipeError, ConnectionResetError) as e:
                print(f"与客户端 {client_address} 的连接出现错误: {e}，已关闭连接。")
            except Exception as e:
                print(f'处理错误: {e}')
            finally:
                try:
                    client_socket.close()
                except Exception:
                    pass  # 确保在异常情况下也能尝试关闭连接，忽略可能的关闭错误
    except KeyboardInterrupt:
        print("接收到用户中断信号，正在关闭服务器...")
        server_socket.close()
