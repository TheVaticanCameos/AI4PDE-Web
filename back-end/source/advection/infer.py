"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import os
from pathlib import Path
import deepxde as dde
import numpy as np
import torch


root_dir: str = str(Path(__file__).parent.parent.parent.parent)
model_path: str = os.path.join(root_dir, 'back-end', 'params', 'advection-params.ckpt-50000.pt')

def advection_solver(u_init: np.array) -> np.array:
    
    dim_x = 5
    sin = torch.sin
    cos = torch.cos
    concat = torch.cat

    # PDE
    def pde(x, y, v):
        dy_x = dde.grad.jacobian(y, x, j=0)
        dy_t = dde.grad.jacobian(y, x, j=1)
        return dy_t + dy_x

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def func_ic(x, v):
        return v

    ic = dde.icbc.IC(geomtime, func_ic, lambda _, on_initial: on_initial)

    pde = dde.data.TimePDE(geomtime, pde, ic, num_domain=250, num_initial=50, num_test=500)

    # Function space
    func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)

    # Data
    eval_pts = np.linspace(0, 1, num=50)[:, None]
    data = dde.data.PDEOperatorCartesianProd(
        pde, func_space, eval_pts, 1000, function_variables=[0], num_test=100, batch_size=32
    )

    # Net
    net = dde.nn.DeepONetCartesianProd(
        [50, 128, 128, 128],
        [dim_x, 128, 128, 128],
        "tanh",
        "Glorot normal",
    )

    def periodic(x):
        x, t = x[:, :1], x[:, 1:]
        x = x * 2 * np.pi
        return concat([cos(x), sin(x), cos(2 * x), sin(2 * x), t], 1)

    net.apply_feature_transform(periodic)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.0005)
    model.restore(model_path)

    x = np.linspace(0, 1, num=100)
    t = np.linspace(0, 1, num=100)

    v_branch = u_init.reshape(1, 50)
    xv, tv = np.meshgrid(x, t)
    x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T

    u_pred = model.predict((v_branch, x_trunk))
    u_pred = u_pred.reshape((100, 100))

    return u_pred


# demo
if __name__ == "__main__":
    x = np.linspace(0, 1, 50)
    u_init = np.sin(2 * np.pi * x)
    u_pred = advection_solver(u_init)
    print(u_pred)
