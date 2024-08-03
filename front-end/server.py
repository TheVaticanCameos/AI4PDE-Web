from flask import Flask, request, jsonify, send_file
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

app = Flask(__name__)
root_dir: str = str(Path(__file__).parent.parent)
model_path: str = os.path.join(root_dir, 'back-end', 'params', 'params.ckpt-1000.pt')
png_path: str = os.path.join(root_dir, 'back-end', 'output', 'poisson1d-test.png')
html_path: str = os.path.join(root_dir, 'front-end', 'PDE-solver2.0.html')

def poisson1d_solver(poly: dict) -> dict[str, list[float]]:
    # Poisson equation: -u_xx = f
    def equation(x, y, f):
        dy_xx = dde.grad.hessian(y, x)
        return -dy_xx - f

    # Domain is interval [0, 1]
    geom = dde.geometry.Interval(0, 1)

    # Zero Dirichlet BC
    def u_boundary(_):
        return 0

    def boundary(_, on_boundary):
        return on_boundary

    bc = dde.icbc.DirichletBC(geom, u_boundary, boundary)

    # Define PDE
    pde = dde.data.PDE(geom, equation, bc, num_domain=100, num_boundary=2)

    # Function space for f(x) are polynomials
    degree = max(poly.keys())
    space = dde.data.PowerSeries(N=degree + 1)

    # Choose evaluation points
    num_eval_points = 10
    evaluation_points = geom.uniform_points(num_eval_points, boundary=True)

    # Define PDE operator
    pde_op = dde.data.PDEOperatorCartesianProd(
        pde,
        space,
        evaluation_points,
        num_function=100,
    )

    # Setup DeepONet
    dim_x = 1
    p = 32
    net = dde.nn.DeepONetCartesianProd(
        [num_eval_points, 32, p],
        [dim_x, 32, p],
        activation="tanh",
        kernel_initializer="Glorot normal",
    )

    # Define and train model
    model = dde.Model(pde_op, net)
    dde.optimizers.set_LBFGS_options(maxiter=1000)
    model.compile("L-BFGS")
    model.restore(model_path, device='cpu')

    max_deg = max(poly.keys())
    features = np.zeros(shape=(1, max_deg+1), dtype=np.float32)
    for key, value in poly.items():
        features[0][key] = value

    fx = space.eval_batch(features, evaluation_points)
    x = geom.uniform_points(100, boundary=True)
    y = model.predict((fx, x))

    fig = plt.figure(figsize=(4, 8))
    z = np.zeros_like(x)
    plt.plot(x, z, 'k-', alpha=0.1)
    plt.plot(evaluation_points, np.transpose(fx), '--', label=r'$f(x)$')
    plt.plot(x, np.transpose(y), '-', label=r'$u(x)$')
    plt.legend()
    plt.title("Solution of 1d Poisson equation")
    plt.savefig(png_path)
    plt.close()
    return {'x': x.ravel().tolist(), 'y': y.ravel().tolist()}

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    poly = {item['power']: item['value'] for item in data}
    return poisson1d_solver(poly)
    #return send_file('../output/poisson1d-test.png', mimetype='image/png')

@app.route('/')
def main_page():
    return send_file(html_path, mimetype='text/html')

if __name__ == '__main__':
    app.run(debug=True)
