"""Backend supported: tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np


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
func_space = dde.data.GRF(length_scale=0.2)

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
losshistory, train_state = model.train(iterations=20000)
dde.utils.plot_loss_history(losshistory)

print("Saving Model...")
model.save("../../params/dr-params.ckpt")
