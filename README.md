# AI4PDE-Web

Repo for 余庆杯

## 1. 项目简介

此 Repo 用于中国科大第二届余庆杯校园软件设计大赛。其实现的主要功能是使用深度学习技术搭建的 Web 端 PDE 在线求解平台。

GitHub网页：https://github.com/TheVaticanCameos/AI4PDE-Web

### 1.1 使用方法

基本使用方法：用户在 Web 端选择需要求解的方程，输入相关参数，后端代码将执行求解过程，将求解结果反馈在 Web 端。

目前可求解的方程有：
- 带 Dirichlet 两点边值的一维 Poisson 方程 $$ \begin{aligned}
    & u''(x)=f(x), \quad x\in\left[0, 1\right]\\
    & u(0)=u(1)=0
\end{aligned} $$ 其中右端项（源项） $f(x)$ 是多项式，由用户通过指定各项系数的方式输入； $u(x)$ 是待求解的函数。
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

### 2.2 后端代码运行方法

> 以下步骤可能需要科学上网

#### 2.2.1 准备工作：Git 下载安装

> [官网](https://desktop.github.com/)，[下载链接](https://central.github.com/deployments/desktop/desktop/latest/win32)，[官方文档](https://help.github.com/en/desktop) 

- Git: 是一个面向开源及私有软件项目的托管平台，是一个分布式的版本控制软件，它可以有效、高速地处理各种项目的版本管理。
- GitHub: 是 Git 的一个托管平台（把本地的代码历史上传到云端），相对于传统的SVN(Subversion)，GitHub具有更强大的功能，已成为当前人们用来管理代码及各种文档的利器。

Git 的安装可以参考 [CSDN上的Git安装教程](https://www.bing.com/search?q=git安装&form=ANNTH1&refig=f820d1be924a4e0a9af1ce3ca2b42a9a&pc=W089)

安装好后，打开 Git CMD，cd到一个空目录（cd是一个命令，打开文件夹用），输入命令
```
git clone https://github.com/TheVaticanCameos/AI4PDE-Web.git
```
如果此目录中出现 repo 中的这些文件，则 clone 成功，否则尝试科学上网后重试。


#### 2.2.2 环境依赖搭建

首先安装 python，推荐3.8及以上版本

打开终端（Windows Power Shell或者vscode中的terminal都可），cd 到`back-end`目录，输入如下命令创建虚拟环境
```
python -m venv env
```
接着输入如下命令激活虚拟环境
```
env\Scripts\activate
```
接下来安装第三方库，输入如下命令（此步骤如果报错或者速度太慢可尝试关闭 / 打开科学上网）
```
pip install -r requirements.txt
```

#### 2.2.3 运行方法

运行infer.py，正常的话会在`back-end/output`目录下生成图片`poisson1d-test.png`（repo中本来就有一张，可以将其删除后再运行）

### 2.3 infer.py中的部分代码解释

`poisson1d_solver(poly: dict)`函数接受一个字典`poly`，其键表示多项式的某一项次数，对应的值表示该项的系数；这一字典用于表示用户指定的右端多项式 $f(x)$。此函数将调用预训练好的模型参数`back-end/params/params.ckpt-1000.pt`，执行对一维 Poisson 方程的求解，并将解的曲线绘制在`back-end/output/poisson1d-test.png`中，如下图所示。其中蓝色的曲线表示用户输入的多项式 $f(x)$，橙色的曲线表示模型输出的解 $u(x)$。

<div  align="center">    
 <img src="back-end/output/poisson1d-test.png" style="zoom:90%" />
</div>


[^1]: [DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators.](https://arxiv.org/abs/1910.03193)
[^2]: [DeepXDE: a library for scientific machine learning and physics-informed learning.](https://github.com/lululxvi/deepxde/tree/master)
