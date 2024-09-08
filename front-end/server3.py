import os
import sys
from pathlib import Path
import numpy as np


from flask import Flask, request, send_file, jsonify

app = Flask(__name__)
root_dir: str = str(Path(__file__).parent.parent)
model_path: str = os.path.join(root_dir, 'back-end', 'params', 'stokes-params-50000.pt')
html_path: str = os.path.join(root_dir, 'front-end', 'stokes-flow.html')

sys.path.append(os.path.join(root_dir, 'back-end', 'source', 'stokes-flow'))
from infer import stokes_flow_solver # type: ignore

@app.route('/stokes_flow', methods=['POST'])
def solve():
    try:
        # 从前端请求中获取数据
        data = request.json 
        input_value = float(data['input_value']) 
        result = stokes_flow_solver(input_value)
        
        # 确保返回的结果是101*101*3的numpy数组
        if isinstance(result, np.ndarray) and result.shape == (101, 101, 3):
            # 将numpy数组转换为JSON友好的格式
            result_list = result.tolist()  # numpy数组转成Python列表
            print(result_list)
            return jsonify({'result': result_list})  # 返回结果给前端
        else:
            return jsonify({'error': 'Invalid output from solver'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def main_page():
    return send_file(html_path, mimetype='text/html')

if __name__ == '__main__':
    app.run(debug=True)