# 算例4：一维对流方程

## 数学背景

### 数学表达

本算例求解一维对流方程（Advection Equation），其表达式如下

$$
\frac{\partial u}{\partial t} + c\cdot\frac{\partial u}{\partial x}=0
$$

其中 $u(x,t)$ 为待求解的函数，$c$ 为常数，$x$ 为空间坐标，$t$ 为时间坐标。

### 物理意义

对流方程（Advection Equation），也称为连续性方程，在流体力学中描述了流体的守恒定律，即质量守恒。它表达了流体在流动过程中，质量既不会被创造也不会被消灭，只会从一个地方移动到另一个地方。

在方程表达式中， $u(x, t)$ 的物理意义是流体的密度场， $c$ 是流体沿着所研究方向的运动速度，即流体的流速，这里认为是一个非零常数。

## 后端代码说明

本算例中固定 $c=1$ 。用户输入散点值（一维 `np.array` 数组）作为方程的初值 $u(x, 0)$ ，要求在 $(0, 1)$ 中等距采样 50 个点值。后端求解函数 `advection_solver` 返回一个二维的 `np.array` 数组，其中第一维为空间步，第二维为时间步，大小为 $100\times 100$ 。

### 目录结构

算例4的后端内容目录如下：
```
advection/
|--train.py     // 网络训练
|--infer.py     // 网络推理
|--README.md    // 说明文档
```
其中 `train.py` 已经预先运行过，得到的网络参数保存在 `../../params/advection-params.ckpt-50000.pt` 中。

### 使用方法

运行 `infer.py`，即可求解一维对流方程。如下示：
```Python
# demo
if __name__ == "__main__":
    x = np.linspace(0, 1, 50)
    u_init = np.sin(2 * np.pi * x)
    u_pred = advection_solver(u_init)
    print(u_pred)
```