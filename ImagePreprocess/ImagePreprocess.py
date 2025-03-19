import socket
import os
import glob
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFile
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
    对图像进行噪声去除，采用高斯模糊处理。
    """
    return cv2.GaussianBlur(image, (5, 5), 0)


def enhance_image(image):
    """
    对图像进行简单的锐化增强处理。
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    enhanced = cv2.filter2D(image, -1, kernel)
    return enhanced


def process_image_pipeline(input_img_path, output_path):
    """
    对单张图像进行整体预处理：读取、CLAHE 亮度均衡、噪声去除、图像增强后保存。
    如果输出文件已存在，则跳过处理。
    """
    try:
        # 修正输入图像路径，假设 FrontPage 和 ImagePreprocess 在同一父目录下
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_img_path = os.path.join(parent_dir, 'FrontPage', input_img_path)

        # 如果处理后的文件已存在，直接跳过
        if os.path.exists(output_path):
            print(f"处理后的图像 {output_path} 已存在，跳过处理。")
            return

        # 读取图像（使用 OpenCV 读取，格式为 BGR）
        img = cv2.imread(input_img_path)
        print(f"尝试读取图像 {input_img_path}，返回值: {img}")
        if img is None:
            print(f"无法读取图像 {input_img_path}")
            return

        # 亮度均衡（CLAHE）
        img_eq = apply_clahe(img)
        # 噪声去除
        img_denoised = denoise_image(img_eq)
        # 图像增强（锐化）
        img_enhanced = enhance_image(img_denoised)

        # 保存处理结果（保存为 PNG 格式）
        cv2.imwrite(output_path, img_enhanced)
        processed_img = cv2.imread(output_path)
        if processed_img is not None:
            print(f"处理后尺寸: {processed_img.shape}")
        print(f"保存处理后的图像：{output_path}")
    except Exception as e:
        print(f"处理图像 {input_img_path} 时出错: {e}")


def batch_process_images(input_dir, output_dir):
    """
    对 input_dir 中所有图像批量执行预处理，并保存到 output_dir。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 假设图片已经预先转换格式（也可以直接在此处处理）
    image_paths = glob.glob(os.path.join(input_dir, '*.png'))
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        name, _ = os.path.splitext(base_name)
        out_path = os.path.join(output_dir, name + '.png')
        process_image_pipeline(img_path, out_path)


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
            print(f"拼接前尺寸: {pair[0]} -> {img1.shape}, {pair[1]} -> {img2.shape}")

            # 用填充方式统一高度
            img1_padded, img2_padded = pad_to_same_height(img1, img2)
            print(f"填充后尺寸: {pair[0]} -> {img1_padded.shape}, {pair[1]} -> {img2_padded.shape}")

            concatenated = np.concatenate((img1_padded, img2_padded), axis=concat_axis)

            # 根据左图的文件名提取数字部分作为输出名称
            base_name = os.path.basename(pair[0])
            name, ext = os.path.splitext(base_name)
            tokens = name.split('_')
            if tokens:
                common_num = tokens[0]
            else:
                common_num = name

            out_path = os.path.join(output_dir, f"{common_num}.png")
            cv2.imwrite(out_path, concatenated)
            print(f"保存拼接图像：{out_path}")
        else:
            print(f"最后一组图像不足两张，跳过。")


# 创建 socket 对象
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定地址和端口（这里假设端口为 8888，你可以根据需要修改）
server_address = ('localhost', 8888)
server_socket.bind(server_address)

# 开始监听
server_socket.listen(1)
print('等待连接...')

try:
    while True:
        # 接受连接
        client_socket, client_address = server_socket.accept()
        print(f'连接来自: {client_address}')

        try:
            client_socket.settimeout(10)  # 设置一个超时时间，例如 10 秒
            # 接收客户端发送的多个图像路径
            paths = client_socket.recv(1024).decode('utf-8')
            image_paths = paths.split('|')

            # 假设输入图像已经是处理过格式的（如果需要转换格式，可先调用 read_and_convert_images 函数）
            input_dir = os.path.dirname(image_paths[0])
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')  # 处理后的图像保存目录
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            processed_image_list = []
            for img_path in image_paths:
                base_name = os.path.basename(img_path)
                name, _ = os.path.splitext(base_name)
                output_path = os.path.join(output_dir, name + '.png')
                process_image_pipeline(img_path, output_path)
                processed_image_list.append(output_path)

            # 将处理后的图像进行拼接
            final_output_dir = os.path.join(output_dir, 'paired')
            if not os.path.exists(final_output_dir):
                os.makedirs(final_output_dir)
            pair_images_and_concat(processed_image_list, final_output_dir, concat_axis=1)

            # 假设拼接后的图像路径为拼接目录下的第一张图像路径（你可以根据实际情况调整）
            final_processed_image_path = os.path.join(final_output_dir, os.listdir(final_output_dir)[0])

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