# 07 概要设计：部件、时序与关键量的确定

> 回答的问题：**这个工具由哪些部件组成？一次运行按什么顺序发生？
> Phase 0 低压基线是如何分桶、如何确定基线的？SLA/SLO 的值是如何被确定出来的？**
> 实现：整个 `bench/` 包 + `bench.py` 入口；本文所有论断均可在代码中核对。

01–06 是**方法论**（回答"为什么这样设计"），本文是**设计**（回答"由什么组成、
按什么顺序、关键量如何落地"）。阅读对象是要接手、评审或扩展代码的人：
读完本文应能画出模块依赖图、说清一次运行从读配置到出报告的全过程、
并复述基线分桶与 SLA/SLO 定值的完整规则。

## 7.1 设计目标与约束

| # | 约束 | 含义 | 对设计的影响 |
| --- | --- | --- | --- |
| C1 | **黑盒客户端** | 只能通过 OpenAI 兼容接口观测被测服务，看不到 KV/slot/GPU 等内部指标 | 所有结论必须由"请求特征 × 可观测时延/成败"推出；水位只能是投影（[05](05-watermark.md)） |
| C2 | **纯标准库** | 目标运行环境无 numpy/pandas（Python ≥ 3.11，仅 tomllib 可用） | MLP 训练、`.npz` 读写、分位数全部手写（[06](06-qos-model.md) §2） |
| C3 | **可复现、可审计** | 同一数据库重放必须得到同一报告；配置值 vs 实测值必须可区分 | 所有派生量（门限、水位、拐点）由落库的充分统计量现场重算；`--report-only` 与实时跑同结论 |
| C4 | **fail-fast** | 配置错误必须在开跑前暴露，而不是跑几十分钟后 | `validate()` 覆盖全部语义约束（含派生长度与 max_context 的一致性） |

三条贯穿全局的设计原则（在 00 章展开为 D1–D18）：

1. **尺子与压力解耦**——判定 A2/A3 的门限必须来自压力之外的量（Phase 0 基线），
   否则门限随压力抬升、指标永真（[03](03-threshold-calibration.md) §3.3）。
2. **不可信的统计量比没有统计量更危险**——样本不足一律显式降级 + 标注，
   绝不静默使用（D10）。
3. **配置值与实际值分开记录**——自适应延长是真实行为，`duration_s` 与
   `duration_actual_s` 必须都对得上（D11）。

## 7.2 总体架构

```
                        ┌──────────────────────────────────────────┐
                        │               bench.py（CLI 入口）        │
                        │  --probe / --report-only / --baseline-only│
                        │  --no-qos / --retrain-qos / --qos-predict │
                        └───────┬──────────────────────┬───────────┘
                                │ load_config+validate │
                                ▼                      │
┌──────────────┐     ┌───────────────────┐            │
│  config.py   │────►│  runner.py 编排器  │◄───────────┘
│ (TOML/JSON)  │     │  ┌─────────────┐  │
└──────────────┘     │  │ Phase 0     │  │
                     │  │ 低压基线     │  │
┌──────────────┐     │  └──────┬──────┘  │
│ prompts.py   │────►│         │ 分位数   │
│ (请求特征)   │     │  ┌──────▼──────┐  │     ┌─────────────┐
└──────────────┘     │  │ ramp 逐级   │  │     │ baseline.py │
                     │  │ 预热/测量/  │──┼────►│ 分桶+门限   │
┌──────────────┐     │  │ 延长/drain  │  │     └─────────────┘
│ client.py +  │◄────┤  └──────┬──────┘  │
│ httpclient.py│     │         │ 每级统计 │     ┌─────────────┐
│ (执行+计时)  │     │  ┌──────▼──────┐  │────►│ metrics.py  │
└──────────────┘     │  │ store.py    │  │     │ 分类/聚合/  │
                     │  │ SQLite 落库 │  │     │ 容量/水位/  │
                     │  └─────────────┘  │     │ 服务等级    │
                     └───────────────────┘     └──────┬──────┘
                                                     │
                        ┌─────────────┐     ┌────────▼─────┐     ┌──────────┐
                        │ qos.py      │────►│ report.py    │────►│ 报告 .md │
                        │ mlp.py      │     │ 结论+渲染    │     └──────────┘
                        │ npz.py      │     └──────────────┘
                        └─────────────┘
```

模块职责（括号内为 2026-11 时点的代码行数，仅供建立体量感，随迭代会变）：

| 模块 | 职责 | 关键产出 |
| --- | --- | --- |
| `bench.py`（501） | CLI 解析、四种运行模式编排（压测/探针/重算报告/基线直测） | 退出码、日志横幅 |
| `config.py`（675） | TOML/JSON 加载、类型宽容（float 收 int）、语义校验、长度参数推导 | `Config` |
| `runner.py`（439） | Phase 0 + ramp 的异步编排（worker 闭环/开环、预热、自适应延长、drain） | `List[StageStats]` |
| `client.py`（338） / `httpclient.py`（233） | 单请求执行（SSE 流式/非流式）、单调时钟计时、失败分类（10 种 `KIND_*`） | `RequestResult` |
| `prompts.py`（91） | 按长度档 × 权重采样请求特征；无 tokenizer 时按字符比估算 | prompt spec |
| `store.py`（425） | SQLite 落库（5 张表）、旧库迁移、report-only 重建数据源 | 数据库 |
| `baseline.py`（282） | 长度分桶、Phase 0 统计、两模式门限解析 | `BaselineResult` / `Thresholds` |
| `metrics.py`（739） | A1/A2/A3 分类、分级聚合、水位判定、容量/水位/服务等级提炼 | `StageStats` / 三个 Finding |
| `qos.py`（940） / `mlp.py`（180） / `npz.py`（181） | 采样→MLP 训练/加载/预测、残差 ECDF 校准、包络守卫、npz 落盘 | `QoSResult` |
| `report.py`（1273） | 结论生成 + 十章报告渲染 | `report_<ts>.md` |
| `mock_server.py`（281） | 自测用模拟服务（排队/拒绝/无限队列三模式） | E2E 验证 |

依赖方向是严格分层的：`config → prompts/client → runner → store → metrics → report`；
`baseline` 被 `runner`/`metrics`/`report` 共享；`stats`（分位数）独立成模块以打破
`metrics ↔ baseline` 的循环依赖。`qos` 只被 `report` 与 `bench.py` 调用，不参与
判定链路——**预测结论永不回写实测结论**。

## 7.3 一次运行的完整时序

```
load_config ─► validate ─► ── live 模式 ──────────────────────────────┐
                             │                                          │
                             ▼                                          │
              Phase 0 基线（baseline_enabled）                           │
                · 并发=baseline_concurrency，时长=baseline_duration_s    │
                · 只统计分位数，不判 A2/A3                                │
                · 按"最短样本桶"自适应延长，落 baseline 表                 │
                             │                                          │
                             ▼                                          │
              resolve_thresholds（两模式，见 §7.5）                      │
                             │                                          │
                             ▼                                          │
              ramp：for c in concurrency_steps                          │
                ┌─────────────────────────────────────────────┐        │
                │ 预热段 warmup_s（落库 is_warmup=1，不统计）   │        │
                │ 测量段：closed=每 worker 串行 / open=固定 RPS │        │
                │   · 每条请求落 requests 表并回填 latency_class│        │
                │   · 样本不足 → 30s 步长延长至 max_stage_dur   │        │
                │ 到点 → drain_timeout_s 等待在途 → aborted 单列 │        │
                │ compute_stage_stats → 落 stages/stage_buckets │        │
                └─────────────────────────────────────────────┘        │
                             │                                          │
                             ▼                                          │
              find_capacity ┐                                           │
              find_watermarks├─► QoS 训练/加载 ─► render_report          │
              find_service_levels ┘                                      │
                             │                                          │
── report-only 模式 ─────────┘（从 stages/stage_buckets/baseline 重建，   │
                         全部派生量现场重算，不信任旧 Finding）──────────┘
```

四种运行模式的差异只在"数据从哪来、跑到哪一步停"：

| 模式 | Phase 0 | ramp | 分析+报告 |
| --- | --- | --- | --- |
| 默认（live） | 跑 | 跑 | 跑（QoS 强制重训） |
| `--probe` | 不跑 | 不跑 | 只做连通性探针 |
| `--baseline-only` | 跑 | 不跑 | 打印各桶分位数与生效门限 |
| `--report-only` | 不跑（读 `baseline` 表） | 不跑（读 `stages` 表） | 跑（优先加载已训模型，服务指纹不符时警告） |

## 7.4 Phase 0 低压基线剖析：分桶与基线确定

基线是整个指标体系的"尺子工厂"（[03](03-threshold-calibration.md) §3.3.2）。
本节把它的两个核心问题——**怎么分桶、怎么定基线**——讲到可逐行核对代码。

### 7.4.1 分桶：桶从哪来、怎么映射

**桶的定义**：`requests.length_buckets` 是一组严格递增的 token 上界，
把请求按 prompt 长度切成 `len+1` 个桶，左开右闭、**尾桶无上界**：

```
length_buckets = [1024, 4096]
→ 桶 0：≤1024    桶 1：1025-4096    桶 2：>4096（兜底，避免超长请求丢失）
```

**桶的两种来源**（显式优先，推导可审计）：

| 来源 | 规则 | 示例 |
| --- | --- | --- |
| 显式配置 `requests.length_buckets` | 用户按生产流量画像设定 | `[1024, 4096]` |
| 按 `max_context_tokens` 推导 | `[ctx/32, ctx/8]`（下限 64/256） | 32768 → `[1024, 4096]`；8K → `[256, 1024]` |

推导结果记录进 `Config.derived_lengths`，启动横幅与报告"请求特征覆盖"节
都会标注哪些数字是配的、哪些是推的——读者永远知道自己在看什么。

**映射函数** `bucket_of(tokens, buckets)`：未知长度（无 usage 且无计划长度）
归 0 号桶；`row_tokens` 优先取实际 `prompt_tokens`，退回计划长度。
分桶的必要性：TTFT 与 prompt 长度强相关（prefill 计算量近似线性），
全局单门限要么让长 prompt 天然落 A3（长度差异污染），要么让短请求的排队
劣化被掩盖——两种结局下"A3 增长"都无法回答"劣化是排队还是长度"。

**流量画像与桶的耦合**：`prompt_token_targets` × `prompt_weights`
（同样支持显式/推导 `[ctx/64, ctx/16, ctx/4]`，权重 `[0.2, 0.5, 0.3]`，
即"短多长少"）决定每个桶能收到多少样本。权重为 0 的长度档对应的桶
**不会收到样本**，按桶延长时会被排除（`runner.py::reachable`），
避免无样本桶把基线阶段拖到上限。

### 7.4.2 基线怎么确定

**执行参数**（`[load]` 配置段）：

| 参数 | 默认 | 约束 |
| --- | --- | --- |
| `baseline_enabled` | true | `baseline_relative` 模式下禁止关闭（validate 拒绝） |
| `baseline_concurrency` | 1 | 必须低到不触发排队/拒绝；≥ 最小压力级时不阻断但报告提示污染风险 |
| `baseline_duration_s` | 30 | > `baseline_warmup_s` |
| `baseline_warmup_s` | 5 | 冷启动瞬态排除 |

**五条确定规则**：

1. **只统计、不判定**。`compute_baseline` 只输出每桶的样本数、失败数与
   TTFT/TPOT 的 p50/p95/p99（最近秩定义，`stats.percentile`：第 q 分位数 =
   排序后第 `ceil(q·N)` 小的值，不插值）。基线阶段**不回填**
   `latency_class`——门限还没解析，填了反而是错的。这是反循环论证的
   物理隔离：尺子的来源不经过尺子自己。
2. **样本量判据按桶，不按总量**。门限是按桶解析的，因此"够用"的定义是
   **每个有流量的桶各自达到 `analysis.baseline_min_samples`（默认 30）**。
   不足时 runner 以 30s 步长自适应延长本级（上限 `max_stage_duration_s`），
   延长在日志显式提示。按权重反推的总量需求 = `floor / min_weight`
   （默认 30 / 0.2 = 150 个总样本）——这就是默认 30s 在多数部署下会被
   延长、而日志会给出"还差多少样本/约需多少秒/建议并发提到多少"的原因。
3. **被污染要喊出来**。基线阶段出现任何失败请求（`total_a1 > 0`）说明
   "低压"并不低压：p99 虚高、门限虚松。报告与日志明确提示降低
   `baseline_concurrency` 或排查可用性，而不是把污染基线当好尺子。
4. **薄桶显式回退**。桶内成功样本 < `baseline_min_samples` 时，该桶 p99
   置信度不足，`baseline_relative` 门限**回退为 absolute 配置值**并在
   报告标注"（回退）"。绝不静默使用一个不可信的分位数。
5. **每次运行重测，不复用旧基线**。服务会漂移（扩容、升级后基线已变而
   门限未变 = 拿旧尺子量新系统）。`--baseline-only` 提供换引擎后的
   快速重测入口：先确认各桶分位数符合预期，再跑完整压测。

**两个口径分开记**：`duration_s` 记配置值、`duration_actual_s` 记实际值
（含自适应延长），两者不一致时报告同时展示——"配置 30s、实际跑了 3 分钟"
必须可解释（D11）。

**落库**：`baseline` 表按桶一行（分位数 + n/n_ok/a1 + 两个时长）。
`--report-only` 重建报告时从这张表读回基线，与实时跑解析同一份数据，
门限结论跨路径一致。

### 7.4.3 基线产出的消费路径

```
BaselineResult（每桶 p50/p95/p99 + 薄桶清单 + 污染提示）
   │
   ├─► resolve_thresholds：baseline_relative 门限 = 基线 p99 × 系数（§7.5）
   ├─► report.py 第二节：基线表 + 生效门限表（含来源列与回退标注）
   └─► 质量退化率 = 该级该桶 p99 ÷ 基线该桶 p99（与门限无关的跨引擎诊断量）
```

## 7.5 判定门限的解析：两模式 + 按桶回退

`resolve_thresholds(cfg, baseline)` 是"尺子"的最终装配点：

```
target_mode=absolute ──► 所有桶统一 ttft_target_s×1000 / tpot_target_ms
                        （样本量不参与，契约口径）

target_mode=baseline_relative ──► for 每个桶 i：
        基线桶 i 可用（n_ok ≥ baseline_min_samples 且 p99 齐全）？
          是 ──► 门限[i] = 基线 p99[i] × 系数（TTFT×ttft_factor，TPOT×tpot_factor）
          否 ──► 门限[i] = absolute 配置值，记入 fallback_buckets 并标注
        基线整体缺失 ──► 整体回退 absolute + notes
```

设计细节：

- **min_samples 默认值按模式区分**：`baseline_relative` 用
  `analysis.baseline_min_samples`（30），`absolute` 用
  `min_samples_per_stage`（60，虽然 absolute 根本不看样本量）。
  所有调用点（runner 与 report-only）共用这一默认，保证压测时与重建
  报告时解析出的门限一致（C3）。
- **系数必须 ≥ 1.0**（validate 强制）：小于 1 会把无压力时的正常请求
  判为不达标。默认 TTFT 3× / TPOT 2× 的含义是人工唯一的质量价值目标——
  **退化预算**："比无压力时慢多少倍算不达标"。
- **`classify` 按桶取门限**：单请求分类时先 `bucket_of(row_tokens)` 再取
  该桶门限；桶缺失（理论上不会发生）退回全局 absolute 保底。

## 7.6 SLA/SLO 的值如何确定：三层递进

SLA/SLO 不是配置里"设"出来的，是**测出来**的（[01](01-metrics-slo-sla.md)
§1.7）。值的确定分三层，每层的输入输出都是明确的数据结构：

### 7.6.1 第一层：请求级分类（classify）

每条成功请求按**所在长度桶的门限**打标：

| 类别 | 判据 |
| --- | --- |
| A1 | `status != "ok"`（HTTP 错误含 429/503、连接失败、整体超时、流错误、空输出，共 10 种子类型） |
| A2 | TTFT ≤ 桶门限 **且** TPOT ≤ 桶门限 |
| A3 | 成功但 TTFT 或 TPOT 超门限 |

短输出免检：`completion_tokens < min_completion_tokens`（默认 2）时
TPOT 不参与判定（分母太小、噪声极大）。基线阶段不分类（§7.4.2 规则 1）。

### 7.6.2 第二层：级汇总（compute_stage_stats）

每级（或 Phase 0）汇总出统计量，**N 不含 aborted**（存活者偏差控制，[04](04-measurement-validity.md) §4.1）：

```
SLA（可用性） = (A2 + A3) / N        "接得住"
SLO（质量）   =  A2      / N         "接得好"
```

同时产出：TTFT/TPOT 的 p50/p90/p95/p99、E2E 与 connect 的 p50/p99、
RPS/output_tps、A1 的 `fail_kinds` 直方图（拒绝族/超时族拆分的数据源）、
按桶拆分（`stage_buckets`）、水位状态。这一层全部落库，是 report-only
重建的唯一数据源。

### 7.6.3 第三层：结论级——人工价值目标 vs 实测边界

| | 输入（人工） | 输出（实测） | 代码 |
| --- | --- | --- | --- |
| 价值目标 | `sla_target`（几个 9，默认 0.999）、`slo_target`（达标率，默认 0.95）、`ttft_factor`/`tpot_factor`（退化预算 3×/2×） | — | `config.py` |
| 实测边界 | 判据参数（水位软/硬门限：A3≥5%、A1≥1%、A3≥50%） | **A3 拐点 = 质量/SLO 边界**（首个 BUSY 级）、**A1 拐点 = 可用性/SLA 边界**（首个 OVERLOAD 级） | `find_service_levels` |
| 目标口径容量 | 同上价值目标 | 3 个 9 容量（SLA 达标前缀）、SLO 容量、**质量容量**（双达标前缀，生产建议唯一基准） | `find_capacity` |

三条提炼规则（与容量/水位边界共用）：

1. **连续前缀**：按并发升序，只有每一级都达标/未触发的区间算数，
   遇到第一个触发级即停——防中间级别抖动造成的高估（D8）。
2. **样本充足准入**：`n_total < min_samples_per_stage` 的级不参与前缀；
   前缀为空时给"低置信度参考容量"并标注，绝不静默丢弃低压优质级（D10）。
3. **不看 `watermark.enabled`**：边界统计是服务等级结论的一部分，
   关掉水位渲染时照样给出（判据函数复用 `classify_watermark` 的门限，
   但入口独立）。

**两种门限模式下拐点的语义差异**（报告必须说清，否则误读）：

| 模式 | A3 拐点含义 |
| --- | --- |
| `baseline_relative`（默认） | 相对基线的**退化起点**："比无压力时慢到门限以上"从哪里开始 |
| `absolute` | **契约违约起点**："对外承诺的 p99 线"从哪里开始守不住 |

拐点（退化起点）严于目标容量（价值目标达标前缀）：质量容量 ≤ 拐点。
报告先陈述实测拐点（"退化从哪里开始"），再给目标口径容量
（"价值目标内推荐跑多少"），两者不混写。

## 7.7 数据模型

SQLite 单文件，五张表（`store.py::SCHEMA`）：

| 表 | 粒度 | 内容 | 谁写 |
| --- | --- | --- | --- |
| `meta` | 键值 | 运行元信息（配置摘要、版本、时间戳） | store |
| `stages` | 每级一行 | 全部 §7.6.2 汇总量 + `watermark_state`/`watermark_reasons_json`/`fail_kinds_json` + 两个时长口径 | runner |
| `stage_buckets` | 每级×每桶一行 | 桶内 A1/A2/A3、超标侧拆分、桶 p99、**生效门限快照** | runner |
| `baseline` | Phase 0 每桶一行 | 分位数 + n/n_ok/a1 + 两个时长口径 | runner |
| `requests` | 每条请求一行 | 特征（长度/输出）、时延三件套、`status`/`error_kind`/`latency_class`/`is_warmup`/`sent_at` | runner |

三个设计决定：

- **只存特征与统计量，不存原始请求/响应内容**——压测数据库可能含生产
  prompt 内容，落库即扩散。
- **生效门限按桶快照**（`stage_buckets.ttft_limit_ms/tpot_limit_ms`）：
  换配置重放旧库时，读者能看到当时用的哪把尺子。
- **`_migrate` 向前兼容**：旧库缺列时补齐默认值，`--report-only`
  照常重建（水位/家族归因等后加功能对旧库同样可算，因为判据只用
  聚合量与 `fail_kinds_json`）。

## 7.8 报告产出

`report.py` 渲染十章 + 两附录（`results/report_<ts>.md`）：

| 章 | 内容 | 数据来源 |
| --- | --- | --- |
| 一、结论 | **实测服务等级**（两个拐点）→ 质量容量/3 个 9/SLO 容量 → 水位边界 → 容量补充说明 | 三个 Finding |
| 二、基线剖析与判定门限 | 基线表、按桶门限表（来源列 + 回退标注）、质量退化率表 | `baseline` 表 |
| 三、测试方法 | 五步法说明（含"实测服务等级"表述） | 静态 |
| 四、分级压力测试结果 | 每级样本/A1/A2/A3/SLA/SLO/水位状态 | `stages` |
| 五、延迟分布 | 成功请求的分位数表 | `stages` |
| 六、A1 构成 | error_kind Top3 + 家族拆分 | `fail_kinds_json` |
| 七、A3 归因 | 超标侧（TTFT/TPOT/双超）× 长度桶二维拆分 | `stage_buckets` |
| 八、请求特征覆盖 | 长度档/桶覆盖度，标注推导来源 | `Config.derived_lengths` |
| 九、QoS 预测模型 | 模型卡片、预测 vs 实测、参考流量预测曲线、包络外推警告 | `QoSResult` |
| 十、建议 | 扩容/调参/复测三类，按归因家族与过载形态分化 | Finding + stages |
| 附录 A/B | 复现方式；数据库字段说明 | 静态 |

## 7.9 配置体系

九个配置段（`config.example.toml` 有完整注释版），按"人工输入什么"分组：

| 段 | 人工输入 | 默认要点 |
| --- | --- | --- |
| `service` | 被测服务连接 | `timeout_s=600`（分钟级请求必须足够大） |
| `slo` | 质量价值目标：退化预算系数 + 契约门限 | `target_mode=baseline_relative`、3×/2× |
| `load` | 压力施加方式 + Phase 0 参数 | 闭环、`[1..128]` 梯度、基线并发 1/30s |
| `requests` | 流量画像（长度档/权重/分桶/上下文上限） | 留空则按 `max_context_tokens` 推导 |
| `analysis` | 价值目标：几个 9 + 达标率 + 样本门限 | 0.999 / 0.95 / 60 / 30 |
| `watermark` | 水位判据门限 | 软 5% / 硬 50% / 拒绝 1% |
| `qos_model` | 预测模型超参 | 启用、`[12,12]`、120 epoch、150s 预算 |
| `storage` / `report` | 输出路径 | 空则由 CLI 时间戳派生 |

类型宽容策略（`_coerce`）：float 字段接受 int 字面量（TOML 的 `150`
与 `150.0` 数值等价），int 字段严格（`120.5` 报错不截断），bool 显式
排除（`true` 不会静默变成 1）。

## 7.10 实现映射

| 设计概念 | 位置 |
| --- | --- |
| 分桶映射/标签 | `bench/baseline.py::bucket_of / bucket_label / bucket_count` |
| Phase 0 编排（按桶延长、量化提示） | `bench/runner.py::_run_baseline / _baseline_shortfall_hints` |
| 基线统计（只统计不判定） | `bench/baseline.py::compute_baseline` |
| 门限解析（两模式 + 回退） | `bench/baseline.py::resolve_thresholds` |
| 请求级分类 | `bench/metrics.py::classify` |
| 级汇总 | `bench/metrics.py::compute_stage_stats` |
| 实测服务等级（两个拐点） | `bench/metrics.py::find_service_levels` |
| 目标口径容量（连续前缀） | `bench/metrics.py::find_capacity` |
| 水位边界 | `bench/metrics.py::find_watermarks` |
| A1 家族拆分 | `bench/metrics.py::split_fail_kinds` |
| 压力编排（预热/延长/drain） | `bench/runner.py::_run_stage` |
| 存储与迁移 | `bench/store.py` |
| 结论与渲染 | `bench/report.py::build_conclusions / render_report` |
| 长度参数推导 | `bench/config.py::_resolve_length_defaults` |

## 7.11 设计决策摘要

方法论层面的 18 条决策（D1–D18）见 [00-overview.md](00-overview.md)，
每条都标注了所属章节。本文补充的是**架构层面**的约束，修改前需对照：

- 分层依赖方向不可逆（`qos` 不参与判定、`stats` 为打破循环而独立）；
- 所有派生量必须能由落库数据现场重算（C3），新增结论若需要新列，
  先检查能否用已有充分统计量推导（如 A1 家族用 `fail_kinds_json`）；
- 新增判定口径必须同时给出：单元断言、mock E2E（`mock_server.py`）、
  `--report-only` 一致性验证，三者缺一不可提交；
- 纯标准库约束（C2）下任何新依赖需单独论证（如 numpy 缺失是硬事实）。
