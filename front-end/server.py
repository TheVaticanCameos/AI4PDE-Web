import os
import sys
from pathlib import Path

from flask import Flask, request, send_file

app = Flask(__name__)
root_dir: str = str(Path(__file__).resolve().parent.parent)
model_path: str = os.path.join(root_dir, 'back-end', 'params', 'poisson1d-params.ckpt-1000.pt')
png_path: str = os.path.join(root_dir, 'back-end', 'output', 'poisson1d-test.png')
html_path: str = os.path.join(root_dir, 'front-end', 'PDE-solver3.0.1.html')

sys.path.append(os.path.join(root_dir, 'back-end', 'source', 'poisson1d'))
from infer import poisson1d_solver # type: ignore

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    poly = {item['power']: item['value'] for item in data}
    return poisson1d_solver(poly)


@app.route('/')
def main_page():
    return send_file(html_path, mimetype='text/html')

if __name__ == '__main__':
    app.run(debug=True)
