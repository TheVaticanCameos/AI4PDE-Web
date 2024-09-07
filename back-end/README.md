# 后端代码说明文档

#### 后端支持的方程

后端目前提供了如何方程的求解：

- 一维 Poisson 方程 [-> 文档](source/poisson1d/README.md)
- 反应扩散方程（Diffusion-Reaction Equation） [-> 文档](source/diffusion-reaction/README.md)
- 斯托克斯流（Stokes Flow）[-> 文档](source/stokes-flow/README.md)

#### 后端代码项目结构

```
back-end/
|--params/                              // 预训练好的网络参数
|   |--dr-params.ckpt-20000.pt          // 反应扩散方程的网络参数
|   |--poisson1d-params.ckpt-1000.pt    // 一维 Poisson 方程的网络参数
|   |--stokes-params-50000.pt           // Stokes 流的网络参数
|--source/                              // Python 代码
|   |--dataset/                         // 训练用数据集
|   |--diffusion-reaction/              // 反应扩散方程
|   |--poisson1d/                       // 一维 Poisson 方程
|   |--stokes-flow/                     // Stokes 流
|   |--utils/                           // 辅助函数
|--README.md                            // 后端说明文档
|--requirements.txt                     // 环境依赖列表
```