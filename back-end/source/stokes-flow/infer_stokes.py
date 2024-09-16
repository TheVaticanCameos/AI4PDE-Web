"""Backend supported: tensorflow, pytorch, paddle"""
import os
from pathlib import Path
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np


root_dir: str = str(Path(__file__).parent.parent.parent.parent)
model_path: str = os.path.join(root_dir, 'back-end', 'params', 'stokes-params-50000.pt')


def stokes_flow_solver(velocity: float) -> np.array:
    """
    Solves the Stokes flow problem for a given velocity.

    Parameters:
    - velocity: float, the velocity used in the boundary condition.

    Returns:
    - np.array, the predicted solution including velocity and pressure fields.
    """
    
    # Define the Partial Differential Equation (PDE) for Stokes flow
    def pde(xy, uvp, aux):
        """
        The PDE equation of Stokes flow.

        Parameters:
        - xy: coordinates.
        - uvp: velocity and pressure values.
        - aux: auxiliary variables.

        Returns:
        - Motion equations in x and y directions and the continuity equation (mass conservation).
        """
        mu = 0.01  # Dynamic viscosity
        u, v, p = uvp[..., 0:1], uvp[..., 1:2], uvp[..., 2:3]  # Unpack velocity and pressure
        grad_u = dde.zcs.LazyGrad(xy, u)  # Gradient of u
        grad_v = dde.zcs.LazyGrad(xy, v)  # Gradient of v
        grad_p = dde.zcs.LazyGrad(xy, p)  # Gradient of p

        # First-order derivatives
        du_x = grad_u.compute((1, 0))
        dv_y = grad_v.compute((0, 1))
        dp_x = grad_p.compute((1, 0))
        dp_y = grad_p.compute((0, 1))

        # Second-order derivatives
        du_xx = grad_u.compute((2, 0))
        du_yy = grad_u.compute((0, 2))
        dv_xx = grad_v.compute((2, 0))
        dv_yy = grad_v.compute((0, 2))

        # Motion equations and continuity equation
        motion_x = mu * (du_xx + du_yy) - dp_x
        motion_y = mu * (dv_xx + dv_yy) - dp_y
        mass = du_x + dv_y

        return motion_x, motion_y, mass

    # Define the geometry of the problem domain
    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    # Boundary condition: slip condition on the top
    def bc_slip_top_func(x, aux_var):
        """
        Boundary condition function for the slip condition on the top boundary.

        Parameters:
        - x: coordinates of points on the boundary.
        - aux_var: auxiliary variables.

        Returns:
        - The value of the top boundary condition.
        """
        return (aux_var / 10 + 1.) * dde.backend.as_tensor(x[:, 0:1] * (1 - x[:, 0:1]))

    bc_slip_top = dde.icbc.DirichletBC(
        geom=geom,
        func=bc_slip_top_func,
        on_boundary=lambda x, on_boundary: np.isclose(x[1], 1.),
        component=0)

    # Create the PDE problem object
    pde = dde.data.PDE(
        geom,
        pde,
        bcs=[bc_slip_top],
        num_domain=5000,
        num_boundary=4000,
        num_test=500,
    )

    # Function space for the solution
    func_space = dde.data.GRF(length_scale=0.2)

    # Generate data points for solving the PDE
    n_pts_edge = 101
    eval_pts = np.linspace(0, 1, num=n_pts_edge)[:, None]
    data = dde.zcs.PDEOperatorCartesianProd(
        pde, func_space, eval_pts, num_function=1000,
        function_variables=[0], num_test=100, batch_size=50
    )

    # Neural network architecture for solving the PDE
    net = dde.nn.DeepONetCartesianProd(
        [n_pts_edge, 128, 128, 128],
        [2, 128, 128, 128],
        "tanh",
        "Glorot normal",
        num_outputs=3,
        multi_output_strategy="independent"
    )

    # Output transform to satisfy boundary conditions
    def out_transform(inputs, outputs):
        x, y = inputs[1][:, 0], inputs[1][:, 1]
        u = outputs[:, :, 0] * (x * (1 - x) * y)[None, :]  # Horizontal velocity
        v = outputs[:, :, 1] * (x * (1 - x) * y * (1 - y))[None, :]  # Vertical velocity
        p = outputs[:, :, 2] * y[None, :]  # Pressure
        return dde.backend.stack((u, v, p), axis=2)

    net.apply_output_transform(out_transform)

    # Create the deep learning model
    model = dde.zcs.Model(data, net)
    model.compile("adam", lr=0.001, decay=("inverse time", 10000, 0.5))
    model.restore(model_path, device='cpu')

    # Evaluate the model for the given velocity
    v = np.ones((1, 101)) * velocity
    xv, yv = np.meshgrid(eval_pts[:, 0], eval_pts[:, 0], indexing='ij')
    xy = np.vstack((np.ravel(xv), np.ravel(yv))).T
    sol_pred = model.predict((v, xy))[0]

    # Plot the predicted solution: commit line 138 - 157 if you don't want to plot the solution in the back-end
    def plot_sol(sol: np.array, pressure_lim: float = 0.03, vec_space: float = 4, vec_scale: float = .5, title: str = "") -> None:
        """
        Plot the velocity vector field and pressure field.

        Parameters:
        - sol: The solution array including velocity and pressure.
        - pressure_lim: The limit for pressure color mapping.
        - vec_space: The spacing between vectors in the plot.
        - vec_scale: The scale of vectors in the plot.
        - title: The title of the plot.
        """
        plt.figure()
        plt.imshow(sol[:, :, 2].T, origin="lower", vmin=-pressure_lim, vmax=pressure_lim, cmap="turbo", alpha=.6)
        plt.quiver(xv[::vec_space, ::vec_space] * 100, yv[::vec_space, ::vec_space] * 100, sol[::vec_space, ::vec_space, 0], sol[::vec_space, ::vec_space, 1], color="k", scale=vec_scale)
        plt.axis('off')
        plt.title(title)
        plt.savefig("stokes.png")

    sol_pred = sol_pred.reshape(101, 101, 3)
    plot_sol(sol_pred, title="Predicted Solution")

    return sol_pred


# demo
if __name__ == "__main__":
    sol = stokes_flow_solver(0.5)
