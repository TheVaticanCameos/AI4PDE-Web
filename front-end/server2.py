import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename  
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)
root_dir: str = str(Path(__file__).resolve().parent.parent)
model_path: str = os.path.join(root_dir, 'back-end', 'params', 'dr-params.ckpt-20000.pt')
upload_folder: str = os.path.join(root_dir, 'front-end','upload')  # 上传文件的存储路径
png_path: str = os.path.join(root_dir, 'back-end', 'output', 'dr.png')
html_path: str = os.path.join(root_dir, 'front-end', 'Diffusion-Reaction.html')

sys.path.append(os.path.join(root_dir, 'back-end', 'source', 'diffusion_reaction'))
from infer import diffusion_solver # type: ignore

# 设置上传文件的保存路径
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 允许上传的文件扩展名
ALLOWED_EXTENSIONS = {'csv', 'txt'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查请求中是否包含文件
    if 'file' not in request.files:
        return jsonify({"error": "请求中没有文件"}), 400

    file = request.files['file']

    # 如果用户没有选择文件，浏览器会提交一个空部分，没有文件名
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 将文件内容转换为numpy数组
        data = np.loadtxt(filepath, delimiter=',') if filename.endswith('.csv') else np.loadtxt(filepath)

        result = diffusion_solver(data)

        # 打印结果
        print("Result:", result)

        # 将二维数组转换为字典列表
        result_dict_list = []
        for row in result:
            result_dict_list.append({
                "id": int(row[0]),
                "name": str(row[1]),
                "value": float(row[2])
            })

        print("Result Dict List:", result_dict_list)

        # 返回结果给前端
        return jsonify({"result": result_dict_list})

    

@app.route('/')
def main_page():
    return send_file(html_path, mimetype='text/html')

if __name__ == '__main__':
    app.run(debug=True)
