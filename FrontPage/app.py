from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_mysqldb import MySQL
import os
import socket

app = Flask(__name__)

# 配置 Flask - MySQLdb
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Bin2222'  # 替换为实际的密码
app.config['MYSQL_DB'] = 'case_db'
app.config['MYSQL_CHARSET'] = 'utf8mb4'

mysql = MySQL(app)

# 配置文件上传路径
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 路由：显示首页
@app.route("/")
def home():
    return render_template('homePage.html')  # 确保 HomePage.html 在 templates 文件夹内


# 路由：显示单人图像上传页面
@app.route("/singleImage")
def singleImage():
    return render_template('singleImage.html')  # 确保 singleImage.html 在 templates 文件夹内


# 路由：显示多张图像上传页面
@app.route("/multipleImage")
def multipleImage():
    return render_template('multipleImage.html')  # 确保 multipleImage.html 在 templates 文件夹内


# 路由：处理上传请求
@app.route('/upload', methods=['POST'])
def upload_data():
    # 获取表单数据
    insurance_id = request.form.get('insurance-id')
    gender = request.form.get('gender')
    age = request.form.get('age')

    print(f"insurance_id: {insurance_id}, gender: {gender}, age: {age}")

    # 获取上传的左眼和右眼图像
    left_eye_image = request.files.get('left-eye-image')
    right_eye_image = request.files.get('right-eye-image')

    if not left_eye_image and not right_eye_image:
        return jsonify({'status': 'error', 'message': '没有上传图像'}), 400

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
        # 拼接左右眼图像路径，用 '|' 作为分隔符，你也可以选择其他合适的分隔符
        paths = f"{left_image_path}|{right_image_path}"

        # 通过 Socket 发送左右眼图像路径给服务端进行预处理
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('localhost', 8888)  # 替换为实际的服务端 IP 和端口
        client_socket.settimeout(5)  # 设置连接超时时间为 5 秒
        try:
            client_socket.connect(server_address)
            client_socket.send(paths.encode('utf-8'))
            processed_image_path = client_socket.recv(1024).decode('utf-8')
            client_socket.close()
        except socket.timeout:
            print("Socket 连接超时，请检查服务端是否正常运行和网络连接。")
        except socket.error as e:
            print(f"Socket 连接或数据传输错误: {e}")
        except Exception as e:
            print(f"其他异常错误: {e}")

    # 将数据存储到数据库
    cursor = mysql.connection.cursor()
    try:
        cursor.execute("""
            INSERT INTO `diagnose` (case_id, case_gender, case_age, left_eye_url, right_eye_url, processed_image_url)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (insurance_id, gender, age, left_image_path, right_image_path, processed_image_path))
        mysql.connection.commit()  # 提交事务
    except Exception as e:
        mysql.connection.rollback()  # 如果插入失败，回滚事务
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        cursor.close()  # 关闭游标

    return jsonify({
        'status': 'success',
        'message': '数据和图像上传成功',
        'data': {
            'insurance_id': insurance_id,
            'gender': gender,
            'age': age,
            'left_eye_image_url': left_image_path,
            'right_eye_image_url': right_image_path,
            'processed_image_url': processed_image_path
        }
    })


# 路由：处理批量上传请求
@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'status': 'error', 'message': '没有上传图片'}), 400

    image_paths = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image_paths.append(file_path)

    paths_str = '|'.join(image_paths)

    # 通过 Socket 发送图片路径给服务端进行预处理
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 8888)  # 替换为实际的服务端 IP 和端口
    client_socket.settimeout(5)  # 设置连接超时时间为 5 秒
    try:
        client_socket.connect(server_address)
        client_socket.send(paths_str.encode('utf-8'))
        processed_image_path = client_socket.recv(1024).decode('utf-8')
        client_socket.close()
        return jsonify({
            'status': 'success',
            'message': '批量图片上传成功',
            'processed_image_path': processed_image_path
        })
    except socket.timeout:
        print("Socket 连接超时，请检查服务端是否正常运行和网络连接。")
        return jsonify({'status': 'error', 'message': 'Socket 连接超时'}), 500
    except socket.error as e:
        print(f"Socket 连接或数据传输错误: {e}")
        return jsonify({'status': 'error', 'message': f'Socket 连接或数据传输错误: {e}'}), 500
    except Exception as e:
        print(f"其他异常错误: {e}")
        return jsonify({'status': 'error', 'message': f'其他异常错误: {e}'}), 500


if __name__ == "__main__":
    app.run(debug=True)
