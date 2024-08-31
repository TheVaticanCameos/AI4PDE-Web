"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import os
from pathlib import Path
import deepxde as dde
import numpy as np


root_dir: str = str(Path(__file__).parent.parent.parent.parent)
model_path: str = os.path.join(root_dir, 'back-end', 'params', 'poisson1d-params.ckpt-1000.pt')
png_path: str = os.path.join(root_dir, 'back-end', 'output', 'poisson1d-test.png')

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
    model.restore(model_path)

    max_deg = max(poly.keys())
    features = np.zeros(shape=(1, max_deg+1), dtype=np.float32)
    for key, value in poly.items():
        features[0][key] = value

    fx = space.eval_batch(features, evaluation_points)
    x = geom.uniform_points(100, boundary=True)
    y = model.predict((fx, x))

    return {'x': x.ravel().tolist(), 'y': y.ravel().tolist()}


# usage example:
if __name__ == '__main__':
    poisson1d_solver({0: 1, 1: 1, 2: 1, 3: 1})
