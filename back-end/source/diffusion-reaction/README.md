# 算例2：反应扩散方程

## 数学背景

### 数学表达

本算例求解反应扩散方程（Diffusion-Reaction Equation），其数学表达式如下\[u_t-D u_{xx}+ku^2-v=0\]这是一个一维含时方程，其中$u=u(t, x)$，$k, D$为常数，$v=v(x)$是只依赖于空间的函数。求解区域为 $x\in[0, 1], t\in[0, 1]$。

### 物理意义

反应扩散方程是描述某些化学物质在空间中随时间扩散和反应的数学模型。这个方程结合了扩散过程和化学反应两个方面，是理解许多物理、化学和生物过程中物质传输和转化的基础。

反应扩散方程通常可以写成以下形式：\[\frac{\partial C}{\partial t}=D\nabla^2C+R(C)\]其中
- $C$代表化学物质的浓度
- $t$代表时间
- $D$是扩散系数，它描述了物质在单位时间内由于随机运动导致的扩散能力
- $\nabla^2$是拉普拉斯算子，用于描述空间中浓度的二阶变化率
- $R(C)$是反应项，它是一个关于浓度$C$的函数，描述了物质之间的化学反应速率

其中上述各项的物理意义如下：
- 扩散项 $D\nabla^2C$：这部分描述了物质由于分子运动导致的自然扩散过程。在没有化学反应的情况下，物质会从高浓度区域向低浓度区域自发地扩散，直到达到均匀分布
- 反应项 $R(C)$：这部分描述了物质之间的化学反应。反应项可以是线性的，也可以是非线性的，取决于具体的化学反应机制。例如，某些反应可能随着浓度的增加而加速，而有些反应可能随着浓度的增加而减缓。

## 后端代码说明

### 目录结构

算例2的后端内容目录如下：
```
diffusion-reaction/
|--train.py         // 网络训练
|--infer.py         // 网络推理
|--demo.csv         // 输入数据实例
|--README.md        // 说明文档
```
其中 `train.py` 已经预先运行过，得到的网络参数保存在 `../../params/dr-params.ckpt-20000.pt` 中。

### 使用方法

运行 `infer.py`，即可求解一维含时的反应扩散方程。

```python
if __name__ == "__main__":
    x = np.linspace(0, 1, num=1000) # should be fixed, not open for user to modify
    v = np.genfromtxt("demo.csv", delimiter=',')    # open to user
    u = diffusion_solver(v) # predicted solution u(x, t)

    # plot surface (optional)
    x = np.linspace(0, 1, num=100)
    t = np.linspace(0, 1, num=100)
    x, t = np.meshgrid(x, t)
    plot_surface(x, t, u, title="Diffusion-Reaction", xlabel="x", ylabel="t", zlabel="u")
```
上述代码中，`x = np.linspace(0, 1, num=1000)` 表示空间离散，`v = np.genfromtxt("demo.csv", delimiter=',')` 读取用户指定的函数 $v(x)$ 的离散表示，其具体格式见 `demo.csv`。`u = diffusion_solver(v)` 调用 `diffusion_solver` 函数求解一维扩散方程，其返回值是一个 `numpy` 二维数组，表示 $u(x, t)$ 离散化，其第一个维度表示 $x$，第二个维度表示 $t$。
