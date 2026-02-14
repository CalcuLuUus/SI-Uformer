## impl.md

### 0. 约定与符号

* 输入图像：`Y`，形状 `[B, C, H, W]`，float32，取值范围由外部 pipeline 决定（建议归一化到 `[0,1]` 或 `[-1,1]`）。
* 频域表示：使用 `torch.fft.rfft2`，输出形状 `[B, C, H, W//2+1]`，dtype complex64。
* mask：`M_s, M_n` 形状建议为 `[B, 1, H, W//2+1]`（单通道后 broadcast 到 C，避免颜色通道 mask 不一致导致色偏）。
* 尺寸处理：本仓库当前 Uformer 实现内部用 `sqrt(L)` 推导 `H/W`，实际要求方形输入；建议统一采用 `expand2square()`（参考 `test/test_sidd.py`）并保留 mask，输出后 crop 回原图。
* AMP 注意：FFT/复数路径建议强制 float32（必要时 `autocast(enabled=False)` 包裹），避免 dtype/数值问题。

### 1. 模块 API 设计（给 Codex 的实现接口）

#### 1.1 `estimate_frequency_response`

```python
def estimate_frequency_response(Y: torch.Tensor, op_meta: dict) -> torch.Tensor:
    """
    Args:
      Y: [B,C,H,W] real
      op_meta: dict, may include:
        - mode: "learned"|"known"|"identity" (default: learned)
        - H_hat: optional precomputed complex spectrum [B,1,H,W//2+1] or [1,1,H,W//2+1]
        - other meta/features (optional)
    Returns:
      H_hat: complex tensor [B,1,H,W//2+1] (broadcastable to C)
    """
```

说明：论文允许通过“effective frequency response estimation”将空间变化或 learned operator 统一到频域响应上。

#### 1.2 `compute_sigma_eff`（默认 learned；可选 V0 fallback）

```python
def compute_sigma_eff(H_hat: torch.Tensor,
                      Y: torch.Tensor,
                      sigma_min: float,
                      tau: float,
                      params: dict) -> torch.Tensor:
    """
    Returns:
      sigma_eff: [B,1,H,W//2+1] real
    """
```

默认：learned uncertainty predictor (U_\phi) 输出 (\sigma_{\text{eff}})，并用 `sigma_min` 做下界（例如 `sigma_eff = sigma_min + softplus(raw)`）。建议在训练时加入校准/单调性约束，使 |\hat H| 大时 (\sigma_{\text{eff}}) 倾向更小，以更稳地满足 no-false-trigger / fixed-point 语义。

可选 V0 fallback（debug/早期对齐用）：只依赖 (|H_hat|) 的代理（保证 (|H|) 大时 sigma_eff 小，从而 (\Omega_n) 为空，满足 fixed-point 语义）。建议实现（可配置）：

* `sigma_eff = max(sigma_min, k / (abs(H_hat)+eps))`
  其中 `k` 可由噪声水平估计或常数占位。

> 重要：不要在代码里直接硬写“|H|>=tau”；框架定义是基于 (\sigma_{\text{eff}}) 阈值分割 (\Omega_s,\Omega_n)。

#### 1.3 `SelectiveInversionUformer`（主模型）

```python
class SelectiveInversionUformer(nn.Module):
    def __init__(self, uformer: nn.Module, cfg: dict):
        """
        cfg includes:
          - sigma_min, tau, eps
          - use_y_as_cond: bool (default True)
          - lambda_stable, lambda_gen, lambda_full
          - etc.
        """
    def forward(self, Y: torch.Tensor, op_meta: dict) -> dict:
        """
        Returns dict containing:
          - X_rec: [B,C,H,W]
          - Xinv_hat: complex [B,C,H,W//2+1]
          - r_hat: complex [B,C,H,W//2+1]
          - M_s, M_n: real masks [B,1,H,W//2+1]
          - x_ctx: [B,C,H,W]
        """
```

### 2. 前向推理（Inference）伪代码（与 Algorithm 2/4 对齐）

Algorithm 2/4 的步骤：FFT → 估计 (\hat H) → 计算 (\sigma_{\text{eff}}) → 划分 (\Omega_s,\Omega_n) → 反演稳定频率 → 生成 null 频率 → 合成 IFFT。 

实现伪代码：

```python
Y0 = Y                                         # keep for crop
Y, pad_mask = expand2square(Y, factor=cfg.pad_factor)
B, C, H, W = Y.shape

Y_hat = rfft2(Y)                                # [B,C,H,W2], complex
H_hat = estimate_frequency_response(Y, op_meta) # [B,1,H,W2], complex
sigma_eff = compute_sigma_eff(H_hat, Y, sigma_min, tau, params)  # [B,1,H,W2], real
M_s = (sigma_eff <= tau).float()                # [B,1,H,W2]
M_n = 1.0 - M_s

# deterministic inversion on stable freqs
Hc = H_hat.expand(-1, C, -1, -1)                # broadcast to channels
# numerically-stable pseudo-inverse (mask BEFORE inverse)
Xinv_hat = (Y_hat * M_s) * conj(Hc) / (abs(Hc)**2 + eps)

x_ctx = irfft2(Xinv_hat, s=(H,W))               # [B,C,H,W]

# Uformer-Residual generator (方案 A)
G_in = torch.cat([Y, x_ctx], dim=1) if use_y_as_cond else x_ctx
r = uformer(G_in)                                # [B,C,H,W] residual
r_hat = rfft2(r) * M_n                           # enforce "only Omega_n"

Xrec_hat = Xinv_hat + r_hat
X_rec = irfft2(Xrec_hat, s=(H,W))
X_rec = crop_with_mask(X_rec, pad_mask, ref=Y0)  # back to original H0,W0
return X_rec (+ diagnostics)
```

### 3. 训练（Training）与损失实现

论文训练算法以 (\Omega_s/\Omega_n) 拆分损失：(L_{\text{stable}}) 监督反演稳定分量，(L_{\text{gen}}) 监督生成的 null 分量。

#### 3.1 频段 ground-truth 的构造（关键实现细节）

论文写 (X[\Omega_s])、(X[\Omega_n])（频域子集）。工程上最稳妥做法：

* `X_hat = rfft2(X)`
* `X_s = irfft2(X_hat * M_s)`
* `X_n = irfft2(X_hat * M_n)`

这样避免在空间域做不明确的“索引”。

#### 3.2 损失项

```python
X_inv = irfft2(Xinv_hat, s=(H,W))
R_n   = irfft2(r_hat,   s=(H,W))

L_stable = l1(X_inv, X_s)
L_gen    = l1(R_n,   X_n)

# optional but recommended for stability
L_full   = l1(X_rec, X)

L = lambda_stable*L_stable + lambda_gen*L_gen + lambda_full*L_full
```

> 备注：在严格门控 `r_hat = FFT(r) * M_n` 下，生成器不会收到来自 (L_{\text{stable}}) 的梯度，这是预期行为（保证不修改 (\Omega_s)）。(L_{\text{stable}}) 更主要用于训练/约束 learned 估计器（\hat H、\sigma_{\text{eff}}）与做诊断指标。

### 4. Uformer 的最小改造点（方案 A）

尽量不改 Uformer 主体：本仓库实现已支持可变输入通道 `dd_in`。

1. **输入通道**：使用 `[Y, x_ctx]` 条件输入时，设置 `dd_in = 2C`。
2. **Residual 输出**：该仓库 `Uformer.forward()` 在 `dd_in != 3` 时直接 `return y`，天然就是 residual `r`（无需额外 global skip）。
3. 其余 Uformer block 保持不动（便于 baseline 对齐、便于 ablation）。

### 5. 数值与稳定性注意事项（必须实现）

* 频域伪逆：优先用 `Y_hat * conj(H) / (|H|^2 + eps)` 的形式；并建议 **先乘 mask 再做伪逆**，避免在 null 频段产生巨大数值。
* `eps`：伪逆分母加 `eps`（实数），即使 mask 已过滤稳定频率，也要防止数值尖峰。
* mask 平滑（可选）：V0 可用硬阈值，后续可切换到 soft mask（例如 sigmoid 温度化）以减少频域硬切导致的振铃。
* 通道一致 mask：建议 `M_s/M_n` 用单通道并 broadcast 到 RGB。
* fixed-point 回归测试：对 `H_hat ≡ 1` 且 `sigma_eff` 设置使 `M_n=0` 的情形，验证 `X_rec ≈ Y`，且 `r_hat` 被 mask 完全清零（生成分支“被动失效”）。该性质来自 Theorem/Corollary 的 fixed-point 与 no false trigger。

---
