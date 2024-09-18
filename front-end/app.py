import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from werkzeug.utils import secure_filename  

app = Flask(__name__)
root_dir: str = str(Path(__file__).resolve().parent.parent)

sys.path.append(os.path.join(root_dir, 'back-end', 'source', 'poisson1d'))
from infer_poisson1d import poisson1d_solver # type: ignore

sys.path.append(os.path.join(root_dir, 'back-end', 'source', 'diffusion-reaction'))
from infer_diffusion import diffusion_solver # type: ignore

sys.path.append(os.path.join(root_dir, 'back-end', 'source', 'stokes-flow'))
from infer_stokes import stokes_flow_solver # type: ignore

sys.path.append(os.path.join(root_dir, 'back-end', 'source', 'advection'))
from infer_advection import advection_solver # type: ignore

# 设置上传文件的保存路径
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

poisson_model_path: str = os.path.join(root_dir, 'back-end', 'params', 'poisson1d-params.ckpt-1000.pt')
poisson_png_path: str = os.path.join(root_dir, 'back-end', 'output', 'poisson1d-test.png')

diffusion_model_path: str = os.path.join(root_dir, 'back-end', 'params', 'dr-params.ckpt-20000.pt')
diffusion_png_path: str = os.path.join(root_dir, 'back-end', 'output', 'dr.png')

stokes_model_path: str = os.path.join(root_dir, 'back-end', 'params', 'stokes-params-50000.pt')

advection_model_path: str = os.path.join(root_dir, 'back-end', 'params', 'advection-params.ckpt-50000.pt')

# ori: 2 4
def allowed_file(filename: str) -> bool:
    ALLOWED_EXTS: set[str] = {'csv', 'txt'}
    return '.' in filename and filename.rsplit('.', maxsplit=1)[-1].lower() in ALLOWED_EXTS

# ori: 1
@app.route('/solve_poisson1d', methods=['POST'])
def solve_poisson1d():
    data = request.json
    poly = {item['power']: item['value'] for item in data}
    return poisson1d_solver(poly)

# ori: 2
@app.route('/upload_diffusion', methods=['POST'])
def upload_file_diffusion():
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
        #print("Result:", result)

        # 将ndarray转换为嵌套列表
        result_nested_list = result.tolist()
        
        #print("Result nested list:", result_nested_list)

        # 返回结果给前端
        return jsonify({"result": result_nested_list})

# ori: 3
@app.route('/stokes_flow', methods=['POST'])
def solve_stokes():
    try:
        # 从前端请求中获取数据
        data = request.json 
        input_value = float(data['input_value']) 
        result = stokes_flow_solver(input_value)
        
        # 确保返回的结果是101*101*3的numpy数组
        if isinstance(result, np.ndarray) and result.shape == (101, 101, 3):
            # 将numpy数组转换为JSON友好的格式
            result_list = result.tolist()  # numpy数组转成Python列表
            return jsonify({'result': result_list})  # 返回结果给前端
        else:
            return jsonify({'error': 'Invalid output from solver'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ori: 4
@app.route('/upload_advection', methods=['POST'])
def upload_file_advection():
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

        result = advection_solver(data)

        # 打印结果
        #print("Result:", result)

        # 将ndarray转换为嵌套列表
        result_nested_list = result.tolist()
        
        #print("Result nested list:", result_nested_list)

        # 返回结果给前端
        return jsonify({"result": result_nested_list})

@app.route('/')
def main_page() -> Response:
    home_html_path: str = os.path.join(root_dir, 'front-end', 'menu.html')
    return send_file(home_html_path, mimetype='text/html')

@app.route('/poisson1d')
def poisson1d_page() -> Response:
    ps1d_html_path: str = os.path.join(root_dir, 'front-end', 'poisson1d.html')
    return send_file(ps1d_html_path, mimetype='text/html')

@app.route('/diffusion')
def diffusion_page() -> Response:
    dr_html_path: str = os.path.join(root_dir, 'front-end', 'diffusion.html')
    return send_file(dr_html_path, mimetype='text/html')

@app.route('/stokes')
def stokes_page() -> Response:
    stokes_html_path: str = os.path.join(root_dir, 'front-end', 'stokes-flow.html')
    return send_file(stokes_html_path, mimetype='text/html')

@app.route('/advection')
def advection_page() -> Response:
    advection_html_path: str = os.path.join(root_dir, 'front-end', 'advection.html')
    return send_file(advection_html_path, mimetype='text/html')

@app.route('/about')
def advection_page() -> Response:
    advection_html_path: str = os.path.join(root_dir, 'front-end', 'about.html')
    return send_file(advection_html_path, mimetype='text/html')

@app.route('/logo')
def display_logo() -> Response:
    logo_ico_path: str = os.path.join(root_dir, 'front-end', 'logo.ico')
    return send_file(logo_ico_path, mimetype='image/x-icon')

def main() -> None:
    app.run(debug=True)

if __name__ == '__main__':
    main()
