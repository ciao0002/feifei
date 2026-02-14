# CoSLight 落地修正版（`mhq_coslight`，非 PPO）

## 已确认决策
- 新增独立 Agent，不在旧逻辑上硬改。
- 新 Agent 命名统一为 `mhq_coslight`（类名建议 `MHQCoSLightAgent`）。
- 协作者总数固定为 `5`（包含自己），即 `self + 4`。
- 当前主线是 MHQ 改进版，不是 PPO。

## 已核对事实（已查明）
- MHQ 主线脚本在：
  - `/root/autodl-tmp/Advanced_XLight/run_multihead_algo_line_jnreal.py`
  - `/root/autodl-tmp/Advanced_XLight/run_multihead_algo_line_hzreal.py`
- 关键配置已在脚本中使用：
  - `USE_MULTIHEAD_Q=True`
  - `HEAD_N=5`
  - `HEAD_AGG='mean'`
  - `USE_UCB_ACTION`
  - `USE_HEAD_BOOTSTRAP`
  - `HEAD_BOOTSTRAP_P=0.8`
- MHQ 核心实现位于：
  - `/root/autodl-tmp/Advanced_XLight/models/colight_agent.py`
- env 相关代码已可查看（已解压）：
  - `/root/projects/Advanced_XLight/CoSLight/CoSLight/envs/...`

## 目标
在 MHQ（multi-head Q）框架下接入 CoSLight 风格协作选择（CoS），形成 `mhq_coslight`，并先跑通 `Jinan real / Hangzhou real` 60 轮串行实验。

## 范围与落点
- 新增：`models/mhq_coslight_agent.py`
- 新增类：`MHQCoSLightAgent`
- 不破坏现有：
  - `models/colight_agent.py`
  - 现有 MHQ 脚本（可并行保留）
- 推荐新增脚本：
  - `run_mhq_coslight_line_jnreal.py`
  - `run_mhq_coslight_line_hzreal.py`

## 关键语义（必须分清两个“5”）
- `HEAD_N=5`：Q 网络多头数量（MHQ 里的 head 数）。
- `COS_TOTAL_K=5`：每个路口协作者数量（包含 self）。
- 两者数值都可为 5，但语义完全不同，不能混用。

## `mhq_coslight` 结构约定
### 1) 主干与输出
- 复用 CoLight/MHQ 的状态编码与注意力主干。
- 增加 CoS head：输入 `[B,N,D]`，输出协作 logits `[B,N,N]`。
- 生成协作者 id：`ids [B,N,5]`，且 `ids[...,0]=self_id` 强制包含自己。
- 由 `ids` 构建动态邻接：`adj_dynamic [B,N,5,N]`。
- Q 输出继续走 MHQ 形式：`q_heads [B,N,HEAD_N,A]`。

### 2) 训练目标（Q-learning 口径，不用 PPO）
- 主损失：TD 回归（沿用现有 MHQ 的 target 构造与 `HEAD_AGG/UCB/bootstrap` 逻辑）。
- CoS 辅助正则（建议）：
  - `loss_diag`：鼓励 `P(i,i)` 较高（自协作稳定）
  - `loss_sym`：约束 `P` 与 `P^T` 接近（协作关系更稳）
  - `loss_entropy`：防止协作分布过早塌缩
- 总体示意：

```python
loss_total = loss_td + beta_diag * loss_diag + gamma_sym * loss_sym - eta_ent * entropy_cos
```

## 与现有 MHQ 配置的对齐
- 保留并继续使用：
  - `USE_MULTIHEAD_Q`
  - `HEAD_N`
  - `HEAD_AGG`
  - `USE_UCB_ACTION`
  - `USE_HEAD_BOOTSTRAP`
  - `HEAD_BOOTSTRAP_P`
- 新增 CoS 配置建议：
  - `COS_ENABLED=true`
  - `COS_TOTAL_K=5`
  - `COS_INCLUDE_SELF=true`
  - `COS_BETA_DIAG=0.05`
  - `COS_GAMMA_SYM=0.10`
  - `COS_ENTROPY_COEF=0.005`

## 分阶段实施
### 阶段A：最小可运行
- 新建 `MHQCoSLightAgent`，跑通：
  - `obs -> cos ids -> adj_dynamic -> q_heads`
- 动作选择仍沿用 MHQ：
  - `mean` 或 `mean + UCB*std` 选动作

### 阶段B：联合训练
- 让 CoS 参数参与反向传播，保持端到端训练。
- TD 目标和 CoS 正则同时优化。
- 校验 `HEAD_N` 与 `COS_TOTAL_K` 在前向/训练/日志三处一致。

### 阶段C：60 轮对照
- Baseline：当前 MHQ（无 CoS）
- Exp1：MHQ + CoS（无约束）
- Exp2：MHQ + CoS + diag
- Exp3：MHQ + CoS + diag + sym（默认主推）

## 验收标准
- 形状正确：
  - `ids.shape == [B,N,5]`
  - `adj_dynamic.shape == [B,N,5,N]`
  - `q_heads.shape == [B,N,HEAD_N,A]`
- 稳定性：
  - 60 轮不崩溃
  - TD loss 不发散
  - CoS 熵不过早塌缩
- 性能：
  - ATT 不低于当前 MHQ baseline
  - 最后 10 轮均值有提升或同等稳定

## 易错点清单
- 把 `mhq_coslight` 写成 `ppo_coslight`（错误路线）。
- 把 `HEAD_N=5` 当成协作者数量（语义错误）。
- 把 `COS_TOTAL_K=5` 解释成“self + 5 他者”（会变成总 6）。
- 同时混用固定邻接与动态邻接而未统一语义。
