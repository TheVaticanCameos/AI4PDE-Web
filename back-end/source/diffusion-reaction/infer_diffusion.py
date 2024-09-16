"""Backend supported: tensorflow, pytorch, paddle"""
import os
from pathlib import Path
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt


root_dir: str = str(Path(__file__).parent.parent.parent.parent)
model_path: str = os.path.join(root_dir, 'back-end', 'params', 'dr-params.ckpt-20000.pt')


def diffusion_solver(v_vals: np.array) -> np.array:
    """
    Solve the diffusion equation with given parameters.

    Args:
        v_vals: The values of the parameter function of v(x).

    Returns:
        np.array: The solution of the diffusion-reaction equation.
    """
    
    # PDE
    def pde(x, y, v):
        D = 0.01
        k = 0.01
        grad_y = dde.zcs.LazyGrad(x, y)
        dy_t = grad_y.compute((0, 1))
        dy_xx = grad_y.compute((2, 0))
        return dy_t - D * dy_xx + k * y**2 - v


    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)

    pde = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        num_domain=200,
        num_boundary=40,
        num_initial=20,
        num_test=500,
    )

    # Function space
    func_space = dde.data.GRF(length_scale=0.2, N=1000)

    # Data
    eval_pts = np.linspace(0, 1, num=50)[:, None]
    data = dde.zcs.PDEOperatorCartesianProd(
        pde, func_space, eval_pts, 1000, function_variables=[0], num_test=100, batch_size=50
    )

    # Net
    net = dde.nn.DeepONetCartesianProd(
        [50, 128, 128, 128],
        [2, 128, 128, 128],
        "tanh",
        "Glorot normal",
    )

    model = dde.zcs.Model(data, net)
    model.compile("adam", lr=0.0005)
    model.restore(model_path, device='cpu')

    func_feats = v_vals.reshape((1, -1))

    x = np.linspace(0, 1, num=100)
    t = np.linspace(0, 1, num=100)

    v_branch = func_space.eval_batch(func_feats, np.linspace(0, 1, num=50)[:, None])

    xv, tv = np.meshgrid(x, t)
    x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T

    u_pred = model.predict((v_branch, x_trunk))
    u_pred = u_pred.reshape((100, 100))

    return u_pred
    

def plot_surface(x: np.array, y: np.array, z: np.array, title: str = "", path: str = "dr.png", xlabel: str = "", ylabel: str = "", zlabel: str = "") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(x, y, z)

    ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # there is some problem in my interactive environment, you can try to use plt.show() instead of the following line
    plt.savefig(path)


if __name__ == "__main__":
    x = np.linspace(0, 1, num=1000) # should be fixed, not open for user to modify
    v = np.sin(x)   # open to user
    u = diffusion_solver(v) # predicted solution u(x, t)

    # plot surface
    x = np.linspace(0, 1, num=100)
    t = np.linspace(0, 1, num=100)
    x, t = np.meshgrid(x, t)
    plot_surface(x, t, u, title="Diffusion-Reaction", xlabel="x", ylabel="t", zlabel="u")
