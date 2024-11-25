# 泊松方程（Poisson Equation）

## 方程背景介绍

### 数学表达

一维泊松方程（Poisson Equation）是偏微分方程中的一种，它描述了在一维空间中，某个物理量（如电势、温度等）如何随位置变化。其数学表达式为：

\[\begin{aligned}
& u''(x)=f(x)\\
& u(0)=u(1)=0\\
\end{aligned}\]

其中 $u(x)$ 是待求解的未知函数，$f(x)$ 是一个给定的函数，称为源项或载荷。

### 物理背景

1. **静电学**
在静电学中，Poisson方程描述了电势$u(x)$在空间中的分布。如果空间中存在电荷密度$\rho(x)$，由库伦定律可得$\nabla^2u=-\frac{\rho}{\epsilon_0}$，其中$\epsilon_0$是真空中的电容率。
2. **热传导**
在热传导问题中，Poisson方程描述了温度分布$u(x)$如何随时间和位置变化。如果存在热源或热汇，那么温度的拉普拉斯算子与热源密度成正比。
3. **流体力学**
在流体力学中，Poisson方程可以用来描述流体中的压力分布。
4. **量子力学**
在量子力学中，Poisson方程与薛定谔方程结合，可以描述粒子在势场中的波函数。