# 05 水位语义：服务端水位的黑盒探测

> 术语来源：`results/watermark.md`（推理水位定义、指标体系、分层设计）。
> 该文面向**服务端实现**（KV Cache / 批 slot / GPU / 队列的资源水位计算）；
> 本文回答一个不同的问题：**黑盒压测工具能落地其中的哪些语义，以及怎么落地**。

## 一、水位是什么：三层定义

推理水位（Inference Watermark）不是单一数值，而是**用来触发限流、排队、熔断、扩缩容、路由降级的负载状态阈值集合**。大模型推理不能沿用普通 Web 的 CPU/QPS 水位——瓶颈是 KV Cache 显存、GPU 算力、批队列、PD 分离调度，且请求成本极不均匀（同样 1 个请求资源消耗差几十倍）。`watermark.md` 给出三层：

| 层 | 状态 | 服务端行为 | 目的 |
| --- | --- | --- | --- |
| 软水位（Soft / Queue） | BUSY | 不再直接进 GPU 批，转入排队/路由分流，**不拒绝** | 保护 P99，避免进入悬崖区 |
| 硬水位（Hard / Reject） | OVERLOAD | 新请求直接 429 拒绝接入（队列无上限的部署除外，见 2.1） | 防雪崩、防全局延迟击穿 |
| SLA 水位 | — | 以业务承诺（P99 TTFT / ITL / error rate）反向驱动 | 调度/扩缩容的最终验收 |

以及两种计算范式：**范式 A**（多维归一化加权合成单一水位值，用于 Worker 内部限流）与**范式 B**（各维度独立阈值、OR 逻辑任一超限即触发，用于集群调度器——本项目的 NanHuInfer 调度层属于此类）。

## 二、黑盒压测只能观测“水位的投影”

本工具是**黑盒客户端**：它向 Router 入口发请求，只看得到

- 时延：TTFT / TPOT / E2E（客户端视角，含建连）
- 成败：成功 / HTTP 429·503 拒绝 / 超时 / 流错误
- 吞吐：RPS、输出 tok/s、长度分布

它**看不到**服务端内部状态：KV Cache 占用率、批 slot 占用、GPU SM/HBM 利用率、prefill/decode 队列长度。这些资源水位的计算属于服务端职责（暴露 Prometheus 指标 → 监控分析器），压测工具替代不了，也不应替代。

但这不意味着水位语义与压测无关。反过来看：**递增并发 ramp 本质上就是从外部探测服务的水位在哪里**。服务端的软/硬水位一旦被流量触达，必然通过排队与拒绝投影到客户端可观测信号上：

```
服务端内部（黑盒不可见）                 客户端可观测（本工具判定）
─────────────────────────────   ◄──►   ────────────────────────────
KV > 软阈值 / 队列 P95 超时               TTFT p99 显著抬升、A3 出现
   ↓ 触达                                   ↓ 判定
软水位 BUSY：转入排队                    BUSY：排队信号（不拒绝）
   ↓ 压力继续 ↑
KV > 硬阈值 / slot 耗尽                    429/503 拒绝出现、A3 崩塌
   ↓ 触达                                   ↓ 判定
硬水位 OVERLOAD：拒绝接入                OVERLOAD：拒绝/质量崩塌信号
```

即：**本工具判定的是服务端水位的可观测投影，不是水位本身**。软水位投影略滞后于服务端软水位触达（客户端要积累到足够的 A3 样本才能分辨），报告中对这一滞后有说明（见第六节）。

### 2.1 无限队列：硬水位的另一种投影

上图假设硬水位以"拒绝"表达（队列有上限，超限 429/503）。但 Router 配**无限队列**时，服务端永不拒绝：压力全部转嫁为排队，客户端先触达自己的超时预算——**过载不再投影为 429/529，而投影为客户端超时**。

```
硬水位触达（队列无上限）                  客户端可观测
─────────────────────────────   ◄──►   ────────────────────────────
KV > 硬阈值 / slot 耗尽                    无 429/503
   ↓ 触达                                   ↓ 判定
无限排队，请求永不完成                     error_kind=timeout 堆积
                                           OVERLOAD：A1（超时族）占比超硬门限
```

这对压测结论的影响与应对：

- **分类上客户端超时本来就是 A1**（`status != "ok"` → A1，见 [01](01-metrics-slo-sla.md)）：SLA 照常拉低、硬水位照常触发，容量结论不受影响；
- **归因上必须区分 A1 的两个家族**（`bench/metrics.py: split_fail_kinds`）：拒绝族（`rejected`/`http_4xx`/`http_5xx`/`stream_error`）= 服务端明确回绝；超时族（`timeout`/`connect_timeout`）= 服务不拒绝但预算内完不成。报告的过载形态、水位触发原因与治理建议按家族构成给出不同解读（排队型 vs 拒绝型）；
- **识别信号**：全程无 429/529、失败以 `timeout` 为主 ⇒ Router 疑似无限队列。此时"拒绝阈值/熔断线"无从谈起，治理动作是给队列设上限/排队超时，让过载重新可被观测，而不是调高客户端超时——后者只会把可用性问题继续藏起来；
- **自测复现**：`mock_server.py --infinite-queue` + `config.mock-infinite-queue.toml`（把 `service.timeout_s` 压到 10s，高压级全部客户端超时）。

## 三、外部对偶判定规则（范式 B 的对偶）

`watermark.md` 范式 B 的服务端伪逻辑：

```
if kv_usage > kv_hard OR slot_usage > slot_hard OR p99_ttft > sla_ttft_limit:
    mark worker = OVERLOAD (hard)
elif kv_usage > kv_soft OR queue_p95_wait > queue_soft:
    mark worker = BUSY (soft，可排队、优先路由其他节点)
else:
    mark worker = NORMAL
```

其**外部对偶**（本工具实现，`bench/metrics.py: classify_watermark`）：

```
if a1_ratio ≥ hard_reject_ratio OR a3_ratio ≥ hard_slo_violation_ratio:
    OVERLOAD          # 服务在拒绝，或无限队列下排队耗尽（客户端超时），或质量已崩塌
elif a3_ratio ≥ soft_slo_violation_ratio OR ttft_p99 ≥ soft_ttft_p99_ms:
    BUSY              # 排队信号：不拒绝，但时延/质量已劣化
else:
    NORMAL
```

门限（`[watermark]` 配置段）与默认值：

| 参数 | 默认 | 语义 | 默认值理由 |
| --- | --- | --- | --- |
| `soft_slo_violation_ratio` | 0.05 | A3 占比 ≥ 5% → BUSY | p99 门限下正常流量 A3 应 ≈0；20 个请求坏 1 个已远超随机抖动 |
| `hard_slo_violation_ratio` | 0.50 | A3 占比 ≥ 50% → OVERLOAD | 过半请求不达标 = 质量崩塌，服务对新增流量已无意义 |
| `hard_reject_ratio` | 0.01 | A1 占比 ≥ 1% → OVERLOAD | 服务端限流不会零星出现；1% 已是明确的失败信号（拒绝族或超时族，见 2.1） |
| `soft_ttft_p99_ms` | 0（不启用） | TTFT p99 超此值 → BUSY | 可选排队预警；0 表示只用违约率判据，不引入第二套绝对尺子 |

三点设计说明：

1. **OR 逻辑，优先级 OVERLOAD > BUSY > NORMAL**：与范式 B 一致——任一硬条件触发即硬水位，软条件被硬条件覆盖（状态互斥，每级只有一个状态）。
2. **判定只用本级已有的聚合量**（a1/a3 计数、ttft_p99）：不引入新的采集、不改请求路径，`--report-only` 从旧库重建时同样可判（幂等）。
3. **与 SLO 门限解耦但同源**：A3 的"是否达标"由 SLO 门限（absolute 或 baseline_relative）判定，水位分类只是把 A1/A3 占比翻译成水位词汇。门限换尺子（D4），水位边界随之移动——这正是水位语义想要的效果：**水位是服务行为的外部表现，门限是观察它的尺子**。

## 四、与容量模型的关系

水位边界与容量结论（[02](02-pd-capacity-model.md)）是同一组数据的两种陈述：

| 水位边界 | 容量模型对应 | 关系 |
| --- | --- | --- |
| 软水位并发（首个 BUSY） | 容量拐点的"前夜" | 排队开始；TTFT 已显著抬升但 SLA 通常仍 100%（排队型过载陷阱 [01]） |
| 硬水位并发（首个 OVERLOAD） | SLA 拐点（knee）的失败型表现 | 服务开始拒绝，或无限队列下排队耗尽（客户端超时）；A1 出现，SLA 跌破目标 |
| SLA 水位并发 | **质量容量 quality_capacity** | SLA∧SLO 双达标的最高连续前缀，生产并发建议的唯一基准（D2） |

报告中三个边界并列陈述，回答两个不同的问题：容量结论回答"**生产应该开多少并发**"，水位结论回答"**服务在哪个并发开始排队、哪个并发开始拒绝/排队耗尽**"。运维据此配置 Router 的队列上限与拒绝阈值（软水位并发 = 排队预警线，硬水位并发 = 熔断线），与 `watermark.md` 第九节的分工（Analyzer 判水位、Router 消费水位）对齐。

## 五、PD 分离的归因映射

`watermark.md` 第五节：prefill 与 decode 瓶颈完全独立，必须两套水位。黑盒压测无法直接读取两平面的资源指标，但已有的 A3 归因（[01](01-metrics-slo-sla.md)）提供了等价的分平面信号：

| 劣化归因 | 物理含义 | 对应平面水位压力 |
| --- | --- | --- |
| TTFT 主导（a3_ttft_only + a3_both 多） | 排队或 prefill 计算慢 | **prefill 平面**：prefill 算力/HBM 水位、准入排队 |
| TPOT 主导（a3_tpot_only + a3_both 多） | decode 吞吐不足 | **decode 平面**：KV Cache 水位（第一瓶颈）、批 slot |

因此水位报告中对 BUSY/OVERLOAD 级附带 A3 归因，即可把"哪个平面先触达水位"指向治理方向（扩容 PD 副本时扩哪一侧）。

## 六、边界情况与已知局限

1. **未触达任何水位**：已测至最高并发仍全部 NORMAL——只说明梯度上限内未观测到水位，不代表服务无水位（与"未观测到 SLA 拐点"同一处理，报告显式说明）。
2. **软水位投影滞后**：客户端按占比判定，需要样本积累。样本不足的级别判为 `unknown`，不参与边界发现（与 D10 一致：不可信的统计量比没有更危险）。
3. **纯排队型 vs 失败型**：A1=0 而 A3 持续上升 = 服务只有软水位投影（排队消化压力）；A1 出现 = 硬水位投影，但要看 A1 的家族构成（`split_fail_kinds`）：拒绝族主导 = 拒绝型过载（治理：扩容/准入/队列上限）；超时族主导且全程无 429/529 = 无限队列表征（治理：先给队列设上限，见 2.1）。前者 SLA 可能全程 100%，容量结论必须以 SLO 为准（D2）。
4. **绝对时延口径的软预警可选启用**：`soft_ttft_p99_ms > 0` 时可在 A3 占比尚低时提前报 BUSY，适合"排队初期就要预警"的场景；但它是第二套绝对尺子，跨引擎对比时记得调整或关闭。

**明确不做的事**（属于服务端职责，本文档划清边界）：

- KV Cache / 批 slot / GPU 的资源水位计算——需要服务端 Prometheus 指标；
- 范式 A 加权复合水位——是 Worker 内部限流器的实现，压测工具不模仿；
- 水位动态自适应（基线倍数水位）——是服务端运行时的策略，压测只在门限层提供基线（[03](03-threshold-calibration.md)）。

## 七、实现映射

| 语义 | 实现 |
| --- | --- |
| `[watermark]` 配置段（4 参数 + 校验） | `bench/config.py: WatermarkConfig`、`load_config`、`validate` |
| 单级水位判定 `classify_watermark` | `bench/metrics.py` |
| A1 家族拆分 `split_fail_kinds`（拒绝族/超时族，驱动过载形态归因） | `bench/metrics.py` |
| 水位边界发现 `find_watermarks`（软/硬/SLA 三边界 + notes） | `bench/metrics.py: WatermarkFinding` |
| 分级表“水位”列、结论区水位陈述 | `bench/report.py: render_report / build_conclusions` |
| 水位状态落库（审计 + `--report-only` 重算） | `bench/store.py: stages.watermark_state / watermark_reasons_json` |
