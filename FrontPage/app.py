
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_mysqldb import MySQL
import os

app = Flask(__name__)

# 配置 Flask-MySQLdb
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

# 路由：显示上传页面
@app.route("/")
def product():
    return render_template('product.html')  # 确保 product.html 在 templates 文件夹内

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

    # 将数据存储到数据库
    cursor = mysql.connection.cursor()
    try:
        cursor.execute("""
            INSERT INTO `diagnose` (case_id, case_gender, case_age, left_eye_url, right_eye_url)
            VALUES (%s, %s, %s, %s, %s)
        """, (insurance_id, gender, age, left_image_path, right_image_path))
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
            'right_eye_image_url': right_image_path
        }
    })

if __name__ == "__main__":
    app.run(debug=True)