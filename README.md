# AI4PDE-Web

Repo for 余庆杯

## 1. 项目简介

此 Repo 用于中国科大第二届余庆杯校园软件设计大赛。其实现的主要功能是使用深度学习技术搭建的 Web 端 PDE 在线求解平台。

GitHub网页：https://github.com/TheVaticanCameos/AI4PDE-Web

### 1.1 使用方法

基本使用方法：用户在 Web 端选择需要求解的方程，输入相关参数，后端代码将执行求解过程，将求解结果反馈在 Web 端。

目前可求解的方程有：
- 带 Dirichlet 两点边值的一维 Poisson 方程\[\begin{aligned}
    &u''(x)=f(x), \quad x\in\left[0, 1\right]\\
    &u(0)=u(1)=0
\end{aligned}\]其中右端项（源项） $f(x)$ 是多项式，由用户通过指定各项系数的方式输入；$u(x)$是待求解的函数。
- ......

### 1.2 技术原理

目前使用的深度学习框架为深度算子网络 DeepONet [^1], 项目后端代码在实现中使用了Lu Lu等学者开发的 DeepXDE 框架[^2]。

--------------------

## 2. for 开发者

### 2.1 项目组织结构

本项目的组织结构如下：
```
root/
|--.idea/                       // 配置文件（无需修改）
|--back-end/                    // 后端文件（代码，模型参数，后端输出）
|   |--requirements.txt         // 后端代码环境依赖
|   |--output/                  // 后端代码的输出，用于反馈给用户
|       |--xxx.png              // 解的图像
|       |--...
|   |--params/                  // 预训练好的模型参数
|       |--params.ckpt-1000.pt  // 用于一维 Poisson 方程的模型参数
|       |--...
|   |--source/                  // 用于求解的 python 代码
|       |--infer.py             // 使用模型参数，用于一维 Poisson 方程的求解
|       |--...
|--front-end/                   // 前端文件
|   |--pde-solver.html          // Web 前端
```

### 2.2 infer.py中的部分代码解释

`poisson1d_solver(poly: dict)`函数接受一个字典`poly`，其键表示多项式的某一项次数，对应的值表示该项的系数；这一字典用于表示用户指定的右端多项式 $f(x)$。此函数将调用预训练好的模型参数`back-end/params/params.ckpt-1000.pt`，执行对一维 Poisson 方程的求解，并将解的曲线绘制在`back-end/output/poisson1d-test.png`中，如下图所示。其中蓝色的曲线表示用户输入的多项式 $f(x)$，橙色的曲线表示模型输出的解 $u(x)$。

<div  align="center">    
 <img src="back-end/output/poisson1d-test.png" style="zoom:90%" />
</div>


[^1]: [DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators.](https://arxiv.org/abs/1910.03193)
[^2]: [DeepXDE: a library for scientific machine learning and physics-informed learning.](https://github.com/lululxvi/deepxde/tree/master)
