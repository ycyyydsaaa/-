from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import socket
import requests
import logging
import queue
import threading

app = Flask(__name__, static_folder='captcha_images', static_url_path='/captcha_images')

# 配置日志记录
logging.basicConfig(level=logging.INFO)

# 配置文件上传路径
# 获取当前文件所在目录，即 frontPage 目录
front_page_dir = os.path.dirname(os.path.abspath(__file__))

# 获取 allproject 项目根目录
project_root = os.path.dirname(front_page_dir)

# 构建 imagePreprocess 文件夹下的 uploads 目录路径
UPLOAD_FOLDER = os.path.join(project_root, 'imagePreprocess', 'uploads')

# 创建目录，如果目录不存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 配置 Flask 应用的上传文件夹
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 假设大模型的 API 端点
MODEL_API_URL = 'https://1594-2001-250-4400-86-00-1729.ngrok-free.app/predict'

# 创建队列
data_queue = queue.Queue()
# 创建结果队列，用于存储模型返回的结果
result_queue = queue.Queue()

# 队列处理函数
def process_queue():
    while True:
        try:
            # 从队列中获取数据包
            data, files = data_queue.get(timeout=500)
            # 发送数据到模型 API，忽略 SSL 证书验证
            response = requests.post(MODEL_API_URL, data=data, files=files, verify=False)
            response.raise_for_status()
            result = response.json()
            # 注释掉这一行，不再打印大模型返回的数据
            # logging.info(f"大模型返回的数据: {result}")
            # 将模型返回的结果放入结果队列
            result_queue.put(result)
            data_queue.task_done()
        except queue.Empty:
            continue
        except requests.RequestException as e:
            logging.error(f'发送数据到模型时出错: {e}')
            # 将错误信息放入结果队列
            result_queue.put({'status': 'error','message': f'发送数据到模型时出错: {e}'})
        except ValueError as e:
            logging.error(f'模型 API 返回的不是有效的 JSON 数据或格式不正确: {e}')
            # 将错误信息放入结果队列
            result_queue.put({'status': 'error','message': f'模型 API 返回的不是有效的 JSON 数据或格式不正确: {e}'})
        finally:
            # 检查 files 变量是否存在且不为空
            if 'files' in locals() and files:
                for file in files.values():
                    if hasattr(file[1], 'close'):
                        file[1].close()

# 启动队列处理线程
queue_thread = threading.Thread(target=process_queue)
queue_thread.daemon = True
queue_thread.start()

# 路由：显示首页
@app.route("/")
def home():
    return render_template('homePage.html')


@app.route("/accounts")
def accounts():
    return render_template('accounts.html')


# 路由：显示单人图像上传页面
@app.route("/singleImage")
def singleImage():
    return render_template('singleImage.html')


# 路由：显示多张图像上传页面
@app.route("/multipleImage")
def multipleImage():
    return render_template('multipleImage.html')


# 路由：处理上传请求
@app.route('/upload', methods=['POST'])
def upload_data():
    # 获取表单数据
    insurance_id = request.form.get('insurance-id')
    gender_str = request.form.get('gender')  # 先获取字符串形式的性别
    age = request.form.get('age')
    keywords = request.form.get('keywords')

    # 将性别字符串转换为整数
    if gender_str is not None:
        gender = 0 if gender_str.lower() == "女" else 1 if gender_str.lower() == "男" else None
        if gender is None:
            logging.error("获取的性别信息无效")
            return jsonify({'status': 'error','message': '获取的性别信息无效'}), 400
    else:
        logging.error("未获取到性别信息")
        return jsonify({'status': 'error','message': '未获取到性别信息'}), 400

    logging.info(f"insurance_id: {insurance_id}, gender: {gender}, age: {age}, keywords: {keywords}")

    # 获取上传的左眼和右眼图像
    left_eye_image = request.files.get('left-eye-image')
    right_eye_image = request.files.get('right-eye-image')

    if not left_eye_image and not right_eye_image:
        return jsonify({'status': 'error','message': '没有上传图像'}), 400

    left_image_path = None
    right_image_path = None
    processed_image_path = None

    # 处理左眼图像
    if left_eye_image:
        left_filename = secure_filename(left_eye_image.filename)
        left_image_path = os.path.join(app.config['UPLOAD_FOLDER'], left_filename)
        left_eye_image.save(left_image_path)

    # 处理右眼图像
    if right_eye_image:
        right_filename = secure_filename(right_eye_image.filename)
        right_image_path = os.path.join(app.config['UPLOAD_FOLDER'], right_filename)
        right_eye_image.save(right_image_path)

    if left_image_path and right_image_path:
        # 定义 server_address，确保在使用前已赋值
        server_address = ('localhost', 8888)
        # 拼接左右眼图像路径，用 '|' 作为分隔符
        paths = f"{left_image_path}|{right_image_path}"

        # 构建 HTTP POST 请求
        post_data = f"paths={paths}".encode('utf-8')
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Content-Length': str(len(post_data))
        }
        request_lines = [
            "POST /your_endpoint HTTP/1.1",
            f"Host: {server_address[0]}:{server_address[1]}",
        ]
        for header, value in headers.items():
            request_lines.append(f"{header}: {value}")
        request_lines.append("")
        request_lines.append(post_data.decode('utf-8'))
        http_request = "\r\n".join(request_lines).encode('utf-8')

        # 通过 Socket 发送 HTTP POST 请求给服务端进行预处理
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 将超时时间延长到 1800 秒（半小时）
        client_socket.settimeout(1800)
        try:
            client_socket.connect(server_address)
            client_socket.sendall(http_request)

            response = b""
            while True:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                response += chunk

            processed_image_path = response.decode('utf-8').strip()
            client_socket.close()
        except socket.timeout:
            logging.error("Socket 连接超时，请检查服务端是否正常运行和网络连接。")
            return jsonify({'status': 'error','message': 'Socket 连接超时'}), 500
        except socket.error as e:
            logging.error(f"Socket 连接或数据传输错误: {e}")
            return jsonify({'status': 'error','message': f'Socket 连接或数据传输错误: {e}'}), 500
        except Exception as e:
            logging.error(f"其他异常错误: {e}")
            return jsonify({'status': 'error','message': f'其他异常错误: {e}'}), 500

    # 创建要发送的数据
    data = {
        'gender': gender,
        'age': age,
        'keywords': keywords
    }
    files = {}
    # 假设只处理一张预处理后的图片，这里以 processed_image_path 为例
    if processed_image_path:
        try:
            if os.path.exists(processed_image_path):
                files['image'] = (os.path.basename(processed_image_path), open(processed_image_path, 'rb'))
                logging.info(f"预处理文件已找到: {processed_image_path}")
            else:
                logging.error(f"文件未找到: {processed_image_path}")
                return jsonify({'status': 'error','message': '预处理后的图片文件未找到'}), 400
        except Exception as e:
            logging.error(f"打开文件时出错: {e}")
            return jsonify({'status': 'error','message': f'打开文件时出错: {e}'}), 400

    # 将数据包放入队列
    data_queue.put((data, files))

    try:
        # 从结果队列中获取模型返回的结果，将超时时间延长到 1800 秒（半小时）
        result = result_queue.get(timeout=1800)
        return jsonify({
           'status': result.get('status','success'),
           'message': result.get('message', '数据和图像已放入队列等待处理'),
            'data': {
                'insurance_id': insurance_id,
                'gender': gender,
                'age': age,
                'left_eye_image_url': left_image_path,
                'right_eye_image_url': right_image_path,
                'processed_image_url': processed_image_path
            },
            'predictions': result.get('predictions', []),
            'diseases': result.get('diseases', []),
            'heatmaps': result.get('heatmaps', [])
        })
    except queue.Empty:
        logging.error("等待模型结果超时")
        return jsonify({'status': 'error','message': '等待模型结果超时'}), 5000

# 路由：处理批量上传请求
@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    files = request.files.getlist('images')
    if not files:
        logging.error("没有上传图片")
        return jsonify({'status': 'error','message': '没有上传图片'}), 400

    # 获取批量上传对应的年龄和性别信息
    ages = request.form.getlist('age')
    genders_str = request.form.getlist('gender')

    if len(files) % 2!= 0:
        logging.error("上传的图片数量必须是偶数")
        return jsonify({'status': 'error','message': '上传的图片数量必须是偶数'}), 400

    if len(ages)!= len(files) // 2 or len(genders_str)!= len(files) // 2:
        logging.error("年龄、性别信息数量与图片组数不匹配")
        return jsonify({'status': 'error','message': '年龄、性别信息数量与图片组数不匹配'}), 400

    logging.info(f"获取到的年龄信息: {ages}")
    logging.info(f"获取到的性别信息: {genders_str}")

    image_paths = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image_paths.append(file_path)

    results = []
    for i in range(0, len(image_paths), 2):
        left_image_path = image_paths[i]
        right_image_path = image_paths[i + 1]
        age = ages[i // 2]
        gender_str = genders_str[i // 2]

        # 将性别字符串转换为整数
        gender = 0 if gender_str.lower() == "女" else 1 if gender_str.lower() == "男" else None
        if gender is None:
            logging.error("获取的性别信息无效")
            return jsonify({'status': 'error','message': '获取的性别信息无效'}), 400

        # 定义 server_address，确保在使用前已赋值
        server_address = ('localhost', 8888)
        # 拼接左右眼图像路径，用 '|' 作为分隔符
        paths = f"{left_image_path}|{right_image_path}"

        # 构建 HTTP POST 请求
        post_data = f"paths={paths}".encode('utf-8')
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Content-Length': str(len(post_data))
        }
        request_lines = [
            "POST /your_endpoint HTTP/1.1",
            f"Host: {server_address[0]}:{server_address[1]}",
        ]
        for header, value in headers.items():
            request_lines.append(f"{header}: {value}")
        request_lines.append("")
        request_lines.append(post_data.decode('utf-8'))
        http_request = "\r\n".join(request_lines).encode('utf-8')

        # 通过 Socket 发送 HTTP POST 请求给服务端进行预处理
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #将超时时间延长到 1800 秒（半小时）
        client_socket.settimeout(1800)
        try:
            client_socket.connect(server_address)
            logging.info(f"已连接到预处理服务: {server_address}")
            client_socket.sendall(http_request)

            response = b""
            while True:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                response += chunk

            processed_image_path = response.decode('utf-8').strip()
            client_socket.close()
            logging.info(f"预处理服务返回的路径: {processed_image_path}")
        except socket.timeout:
            logging.error("Socket 连接超时，请检查服务端是否正常运行和网络连接。")
            return jsonify({'status': 'error','message': 'Socket 连接超时'}), 500
        except socket.error as e:
            logging.error(f"Socket 连接或数据传输错误: {e}")
            return jsonify({'status': 'error','message': f'Socket 连接或数据传输错误: {e}'}), 500
        except Exception as e:
            logging.error(f"其他异常错误: {e}")
            return jsonify({'status': 'error','message': f'其他异常错误: {e}'}), 500

        # 创建要发送的数据
        data = {
            'gender': gender,
            'age': age
        }
        files = {}
        # 假设只处理一张预处理后的图片，这里以 processed_image_path 为例
        if processed_image_path:
            try:
                if os.path.exists(processed_image_path):
                    files['image'] = (os.path.basename(processed_image_path), open(processed_image_path, 'rb'))
                    logging.info(f"预处理文件已找到: {processed_image_path}")
                else:
                    logging.error(f"文件未找到: {processed_image_path}")
                    return jsonify({'status': 'error','message': '预处理后的图片文件未找到'}), 400
            except Exception as e:
                logging.error(f"打开文件时出错: {e}")
                return jsonify({'status': 'error','message': f'打开文件时出错: {e}'}), 400

        # 将数据包放入队列
        data_queue.put((data, files))

        try:
            # 从结果队列中获取模型返回的结果，将超时时间延长到 1800 秒（半小时）
            result = result_queue.get(timeout=1800)
            results.append(result)
        except queue.Empty:
            logging.error("等待模型结果超时")
            return jsonify({'status': 'error','message': '等待模型结果超时'}), 500

    combined_result = {
       'status':'success',
       'message': '批量图片已处理完成',
       'results': results
    }
    return jsonify(combined_result)

if __name__ == "__main__":
    app.run(debug=True)