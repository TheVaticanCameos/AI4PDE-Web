# 斯托克斯流（Stokes Flow）

## 方程背景介绍

### 数学表达

Stokes流问题是指在流体力学中，当流体的速度较低，以至于流体的惯性力可以忽略不计时的流动问题。这种情况通常发生在流体流动速度很慢，或者流体的粘度很高的场合。在这种情况下，Navier-Stokes方程中的非线性项（即流体速度的对流项）可以忽略，从而简化为Stokes方程。其具体的数学表达形式如下

\[\begin{aligned}
\mu\nabla^2 u-\nabla p=0\\
\nabla \cdot p=0
\end{aligned}\]

自变量$(x, y)\in[0, 1]^2$. 其中各个物理量的含义如下：
- $\mu$ 为动力粘性系数；
- $u$ 是速度矢量；
- $p$ 为压力

### 物理背景

Stokes流问题的物理背景是描述低雷诺数（Reynolds number）下的流体流动。雷诺数是一个无量纲数，用来描述流体流动中惯性力与粘性力的比值。当雷诺数很低时，粘性力占主导地位，流体的流动主要受粘性影响，而不是惯性力。这种情况常见于润滑油流动、微小生物体在水中的运动、微流体学中的流动等现象。

在Stokes流问题中，由于忽略了惯性力，流体的流动更加平稳和可预测，没有湍流现象。这种流动问题在工程和物理研究中非常重要，因为它涉及到许多实际应用，如润滑理论、悬浮粒子的沉降、微尺度流体设备的设计等。

总的来说，Stokes流问题提供了一个理解和预测低速流体流动行为的重要工具，特别是在粘性力起主导作用的情况下。