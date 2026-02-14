## idea.md

### Selective Inversion：Identifiability-Aware 的“反演 + 生成”统一框架（任务无关实现版）

#### 1. 核心动机

现实退化 (Y = HX + N) 中，并非所有图像分量都同等可恢复：有些频率成分在前向算子下依然“可辨识/稳定可逆”，应当用确定性反演恢复；另一些频率被严重衰减或近似置零，属于“不可辨识/null”，只能依赖先验（生成模型）补全。该框架的关键是用频域的可辨识性划分，决定“哪里反演、哪里生成”，避免可恢复内容被生成模块覆写（hallucination）。

#### 2. 频域可辨识性分解（核心定义）

在推理阶段无法获得真实后验方差，因此用“有效频率不确定性/有效噪声” (\sigma_{\text{eff}}(\omega)) 近似，并据阈值 (\tau) 划分稳定/空子空间：

* (\sigma^2_{\text{eff}}(\omega)=\max{\sigma^2_{\min}, f(\hat H(\omega), Y)})
* (\Omega_s={\omega:\sigma_{\text{eff}}(\omega)\le \tau})（稳定可逆频率）
* (\Omega_n={\omega:\sigma_{\text{eff}}(\omega)>\tau})（不可辨识频率）

其中 (f(\cdot)) 可基于算子响应、观测谱能量，或学习式不确定性预测器（可选模块）。

#### 3. 恢复算子（框架输出定义）

* 对稳定频率 (\Omega_s)：做确定性反演
  [
  \hat X_{\Omega_s}(\omega)=\hat Y(\omega)/\hat H(\omega),\quad \omega\in\Omega_s
  ]
  (\Omega_n) 处置零得到仅含稳定分量的频谱。
* 对不可辨识频率 (\Omega_n)：用条件生成模型补全
  [
  \hat X_{\Omega_n}(\omega)=G(\text{IFFT}(\hat X_{\Omega_s}),\Omega_n),\quad \omega\in\Omega_n
  ]
* 合成输出
  [
  X_{\text{rec}}=\text{IFFT}(\hat X_{\Omega_s}+\hat X_{\Omega_n})
  ]


#### 4. 框架的关键“工程不变量”

1. **Controlled Generative Reconstruction**：生成只作用于 (\Omega_n)，机制上避免修改可辨识分量。
2. **Clean-Image Fixed-Point / No False Trigger**：对干净或轻退化输入，(|\hat H(\omega)|) 全频不接近 0 时，(\sigma_{\text{eff}}) 全频小，得到 (\Omega_s=\Omega,\Omega_n=\emptyset)，生成模块不触发，输出近似输入。
3. **对算子失配的稳定性（仅稳定频率反演）**：只对满足 (|H(\omega)|\ge\tau) 的频率反演时，失配与噪声放大均可界定，不出现灾难性放大。

#### 5. 我们的通用实现选择（与论文框架对齐）

* 生成器采用 **方案 A：Uformer-Residual**

  * 输入：(x_{\text{ctx}}=\text{IFFT}(\hat X_{\Omega_s}))（可选拼接 (Y)）
  * 输出：空间域残差 (r)
  * 通过 FFT + 频域门控强制 (r) 只贡献 (\Omega_n)（实现“只生成 (\Omega_n)”）。
* (\hat H) 与 (\sigma_{\text{eff}}) 做成插件接口：先做最小可运行版本（外部提供/已知/简化估计），后续可迭代引入 learned estimator。

---

