from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义模型层次
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 加载模型
model = Model()
model.load_state_dict(torch.load("C:/Users/李颖/Desktop/USTC/大一下/afterschool/余庆杯/git/AI4PDE-Web/back-end/params/params.ckpt-1000.pt", map_location=torch.device('cpu')))
model.eval()

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    x = np.linspace(-10, 10, 100)
    y = np.zeros_like(x)

    for item in data:
        power = item['power']
        value = item['value']
        y += value * np.power(x, power)

    x_tensor = torch.from_numpy(x).float().unsqueeze(1)
    y_tensor = model(x_tensor).detach().numpy()

    plt.figure()
    plt.plot(x, y_tensor, label="u(x)")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid()
    plt.savefig('output.png')
    plt.close()

    return jsonify({"x": x.tolist(), "y": y_tensor.flatten().tolist()})

@app.route('/output.png')
def output_file():
    return send_file('output.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
