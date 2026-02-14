## plan.md

### 目标

实现一个任务无关的 Selective Inversion 代码骨架：给定输入 (Y)，由 **learned 估计器**从 (Y/op_meta) 直接预测 (\hat H) 与 (\sigma_{\text{eff}})，输出 (X_{\text{rec}})。生成器使用 Uformer-Residual，并严格落实“生成只在 (\Omega_n)”的频域约束。推理流程与论文 Algorithm 2/4 一致。

### Phase 0：骨架与单元测试（必须先做）

1. **FFT/RFFT 工具链**

   * 统一使用 `torch.fft.rfft2 / irfft2`（实输入更省、并天然满足共轭对称）。
   * 封装：`fft2c(x)`, `ifft2c(Xhat, size)`，保证归一化一致、梯度可回传。
   * AMP 注意：FFT/复数路径建议强制 float32（必要时 `autocast(enabled=False)` 包裹），避免 dtype/数值问题。

2. **尺寸与 padding/cropping 约定（和本仓库 Uformer 对齐）**

   * 由于当前 Uformer 实现内部用 `sqrt(L)` 推导 `H/W`，推理/训练建议统一做 `expand2square()`（参考 `test/test_sidd.py`）并保留 mask，输出后 crop 回原图。
   * 训练 patch 已是方形（`train_ps`），但真实推理/验证必须显式处理任意尺寸输入。

3. **mask 逻辑与 fixed-point 测试**

   * 构造 “轻退化/干净输入”用例，验证 (\Omega_n) 为空时：输出应等价于确定性反演，且生成器不应改变结果（触发率为 0）。该性质来自 clean-image fixed-point。

### Phase 1：Selective Inversion Wrapper（默认 learned 估计器）

实现核心推理管线（训练同构）：

* `H_hat = estimate_frequency_response(Y, op_meta)`（默认：learned predictor；保留 `identity/known` 仅用于 debug/fixed-point）
* `sigma_eff = compute_sigma_eff(H_hat, Y, params)`（默认：learned uncertainty predictor；可选 fallback 到解析式代理）
* `M_s = (sigma_eff <= tau)`, `M_n = 1 - M_s`
* **数值稳定的频域伪逆（建议）**：`Xinv_hat = (Y_hat * M_s) * conj(H_hat) / (|H_hat|^2 + eps)`
* `x_ctx = irfft2(Xinv_hat)`
* `r = G_theta(x_ctx, optional Y)`
* `r_hat = rfft2(r) * M_n`
* `Xrec = irfft2(Xinv_hat + r_hat)`

该逻辑对应 stable/null 划分与反演/生成/合成公式。

### Phase 2：Uformer-Residual Generator 集成（方案 A）

对现有 Uformer 做“最小侵入式/尽量不改主体”的集成：

* 输入通道使用 `[Y, x_ctx]`（更稳，默认），即 `dd_in = 2C`。
* 该仓库 Uformer 的 `forward` 在 `dd_in != 3` 时直接 `return y`，天然就是 residual 输出（无需额外 global skip）。
* 不改变 Uformer 主体层级结构；只需在构建 arch 时正确设置 `dd_in`（必要时再补一层轻量 wrapper 适配通道数）。

### Phase 3：训练脚本与损失（frequency-separated）

实现论文式两项损失（并保留可选全图重建损失）：

* (L_{\text{stable}})：只监督稳定频段的反演结果
* (L_{\text{gen}})：只监督不可辨识频段的生成残差
  训练算法结构与 Algorithm 3 对齐。

> 备注：在严格门控 `r_hat = FFT(r) * M_n` 下，生成器不会收到来自 (L_{\text{stable}}) 的梯度，这是预期行为（保证不修改 (\Omega_s)）。(L_{\text{stable}}) 更主要用于训练/约束 learned 估计器（\hat H、\sigma_{\text{eff}}）与做诊断指标。

### Phase 4：可插拔估计器迭代（任务无关的关键）

默认走 learned，同时保留 debug 模式（便于回归测试与定位）：

1. `estimate_frequency_response` 接口（推荐至少三模式）：

* `learned`（默认）：从 (Y, op_meta) 直接预测 (\hat H)
* `known`：外部提供 (\hat H)（用于快速对齐/调通管线）
* `identity`：(\hat H \equiv 1)（用于 fixed-point/无算子测试）

2. `compute_sigma_eff` 的实现层级（建议默认 learned）：

* V_learned（默认）：learned uncertainty predictor (U_\phi)，输出 (\sigma_{\text{eff}})（可从 (Y, |\hat H|, \angle\hat H, op\_meta) 等特征预测）
* V_fallback（可选）：基于 (|\hat H|) 的不确定性代理（例如 (\sigma_{\text{eff}} \propto 1/(|\hat H|+\epsilon))），用于 debug 或早期对齐

为了保持 no-false-trigger 语义：建议对 (\sigma_{\text{eff}}) 增加单调性/校准约束（例如鼓励 |\hat H| 大时 (\sigma_{\text{eff}}) 小），并在轻退化样本上约束 `mask_trigger_rate`。

### 需要记录的调试指标（强烈建议）

* `mask_trigger_rate = mean(M_n)`（每 batch / 每图）
* `stable_band_error = ||ifft(Xinv_hat)-ifft(X_gt_hat*M_s)||`
* `null_band_error = ||ifft(r_hat)-ifft(X_gt_hat*M_n)||`
* `full_recon_error = ||Xrec - X||`
  用于定位“mask 划分过激/过宽”“生成侵入稳定频段”“固定点被破坏”等问题。
