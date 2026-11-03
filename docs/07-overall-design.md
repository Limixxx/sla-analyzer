# 07 概要设计：部件、时序与关键量的确定

> 回答的问题：**这个工具由哪些部件组成？一次运行按什么顺序发生？
> Phase 0 低压基线是如何分桶、如何确定基线的？请求特征是如何被采样的？
> SLA/SLO 的值与水位的判定是如何被确定出来的？批量/容量场景如何在同一条
> 管道上完成采集与分析（§7.10）？**
> 实现：整个 `bench/` 包 + `bench.py` 入口；本文所有论断均可在代码中核对。

01–06 是**方法论**（回答"为什么这样设计"），本文是**设计**（回答"由什么组成、
按什么顺序、关键量如何落地"）。阅读对象是要接手、评审或扩展代码的人：
读完本文应能画出模块依赖图、说清一次运行从读配置到出报告的全过程、
复述基线分桶与 SLA/SLO 定值的完整规则，并说清在线与容量两种分析如何
共享一次采集、分流两种报告。

## 7.1 设计目标与约束

| # | 约束 | 含义 | 对设计的影响 |
| --- | --- | --- | --- |
| C1 | **黑盒客户端** | 只能通过 OpenAI 兼容接口观测被测服务，看不到 KV/slot/GPU 等内部指标 | 所有结论必须由"请求特征 × 可观测时延/成败"推出；水位只能是投影（[05](05-watermark.md)） |
| C2 | **纯标准库** | 目标运行环境无 numpy/pandas；TOML 解析用内置 `tomllib`（≥ 3.11），< 3.11 可回退环境已有的 `tomli`/`toml` | MLP 训练、`.npz` 读写、分位数全部手写（[06](06-qos-model.md) §2） |
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
                        ┌─────────────┐     ┌────────▼─────┐     ┌──────────────┐
                        │ qos.py      │────►│ report.py    │────►│ 延迟报告 .md  │
                        │ mlp.py      │     │ (在线分析)   │     │ qos_model.npz│
                        │ npz.py      │     └─────────────┘     └──────────────┘
                        └─────────────┘     ┌─────────────┐     ┌──────────────┐
                        ┌─────────────┐────►│ report_batch│────►│ 吞吐报告 .md  │
                        │ throughput  │     │ .py(容量分析)│     │ throughput   │
                        │ .py         │     └─────────────┘     │ _model.npz   │
                        └─────────────┘                         └──────────────┘
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
| `report_batch.py`（新增，§7.10） | 容量分析报告渲染 + 容量面汇写 | 吞吐报告、`throughput_model.npz` |
| `throughput.py`（新增，§7.10） | 容量面 fit/query/save/load（不训练，纯标准库） | 容量面模型 |
| `mock_server.py`（281） | 自测用模拟服务（排队/拒绝/无限队列三模式） | E2E 验证 |

依赖方向是严格分层的：`config → prompts/client → runner → store → metrics → report`；
`baseline` 被 `runner`/`metrics`/`report` 共享；`stats`（分位数）独立成模块以打破
`metrics ↔ baseline` 的循环依赖。`qos` 只被 `report` 与 `bench.py` 调用，不参与
判定链路——**预测结论永不回写实测结论**。`report_batch`/`throughput` 只被
`bench.py` 组合层调用，同样不参与判定链路；在线模块零容量 import（§7.10.7）。

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

负载形状（长度档 / 控制循环 / 阶梯）与报告选择（`report.kinds`）都是**配置值，
不是运行模式**：同一条管道，固定长度档 + 开环 + 只出吞吐报告，就是一次完整的
容量测量（§7.10）；默认配置的行为与本文所述在线路径逐字节一致。

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

**采样是逐请求独立的，不存在"短 runner / 长 runner"**：全 run 只构造一个
`PromptFactory`（内含一个 `random.Random(cfg.seed)`），被所有 worker 共享；
每个 worker 每轮循环现抽一次规格再发请求（`prompts.py::sample_spec`）：

```
worker 数：closed 模式 = 本级并发数（每 worker 串行"发→等→再发"）
          open   模式 = 1 个发射器，按 target_rps 定时发

每轮循环（任一 worker）：
    slot   = take_slot()             # 额度控制：预热/测量/延长/上限，见 §7.3
    spec   = factory.sample_spec()   # ← 每次调用都重新抽，worker 不绑定长度档
    prompt = factory.build(spec)     # 头部 seq + nonce，规避上游前缀缓存命中

sample_spec：两次相互独立的加权抽样（不是配对表）
    prompt_tokens_target = weighted_choice(prompt_token_targets, prompt_weights)  # [512,2048,8192]×[.2,.5,.3]
    max_tokens           = weighted_choice(output_token_choices,  output_weights) # [128,256]×[.7,.3]
    → 二维组合是独立乘积：512/256 与 8192/128 都会出现
```

两条必然后果（读报告和改配置时都要记住）：

- **比例靠大数定律收敛，不是按桶配额轮转**。所以单级的各桶样本数是随机的，
  权重最小的桶天然最少——这正是基线阶段最短桶（0.2 权重）成为薄桶、进而
  触发 absolute 回退的直接原因（§7.4.2 规则 2）。
- **可复现的是"分布"而非"内容"**：抽样序列共享同一个 RNG，同 seed 且请求
  发出顺序相同才复现出同样的 prompt 文本；worker 交错顺序变了，文本就变，
  但长度档分布不变。

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

   **延长判据与回退判据不同源，必须分开核对**（否则会出现"延长已停止但
   仍有桶回退"的困惑）：

   ```
   延长判据（runner，决定本级要不要继续跑）：
       min( 已发出请求数[桶]  for 权重>0 的桶 ) < baseline_min_samples
       分桶键 = spec.prompt_tokens_target（计划长度）；发出即计数（含随后失败与在途的）

   回退判据（baseline.py，决定该桶能不能用）：
       n_ok[桶] ≥ baseline_min_samples  且  TTFT_p99[桶] 非空  且  TPOT_p99[桶] 非空
       分桶键 = row_tokens(row)（实际 usage 的 prompt_tokens，缺失才退计划长度）
       只计 status == "ok" 的请求
   ```

   分桶键一个是"计划"一个是"实际"，而实际长度普遍短于计划（真实 token 数
   由服务端 tokenizer 决定，工具侧只按 `chars_per_token` 估算），请求会
   向更低的桶漂移——最短桶因此更难达标。

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
        usable = 基线桶 i 的 n_ok ≥ baseline_min_samples
                 且 TTFT_p99[i] 非空 且 TPOT_p99[i] 非空       # 联合门槛，见下
          是 ──► 门限[i] = 基线 p99[i] × 系数（TTFT×ttft_factor，TPOT×tpot_factor）
          否 ──► 门限[i] = absolute 配置值，记入 fallback_buckets 并标注
        基线整体缺失 ──► 整体回退 absolute + notes
```

设计细节：

- **回退是"按桶"而不是"全局"**：薄桶用自己的绝对值尺，其余桶仍走基线口径；
  一次运行里两种口径的桶并存是正常状态，报告第二节"来源"列会逐桶标明
  `基线×系数` 还是 `回退 absolute`。
- **联合门槛**：`usable` 要求 TTFT 与 TPOT 两个 p99 **同时**齐全，任一侧缺失
  （该侧样本序列为空 → `percentile` 返回 None）则该桶**两把尺一起**回退，
  不做"一半可信一半不可信"的混搭判定。反过来的宽松点也要知道：样本量门槛
  看的是 `n_ok`（桶内全部成功请求），而 TPOT 序列还额外被
  `completion_tokens ≥ min_completion_tokens` 过滤过，因此理论上存在
  "`n_ok` 已达 30 但 TPOT 有效样本极少、p99 仍算可用"的情形——审计薄桶时
  应同时看 `baseline.tpot_p99_ms` 是否落在可信区间，必要时提高
  `min_completion_tokens` 或加长基线。
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

## 7.6 SLA/SLO 的值如何确定：三层递进 + 水位公式

SLA/SLO 不是配置里"设"出来的，是**测出来**的（[01](01-metrics-slo-sla.md)
§1.7）。值的确定分三层（§7.6.1–§7.6.3），水位投影用与拐点同源的判据单独判定
（§7.6.4）；每层的输入输出都是明确的数据结构：

### 7.6.1 第一层：请求级分类（classify）

每条成功请求按**所在长度桶的门限**打标：

```
classify(row):
    if row.status != "ok":  return A1                 # 失败只看 status，不与任何门限比较

    idx        = bucket_of(row_tokens(row), buckets)   # 分桶键：实际 prompt_tokens，缺失才退计划长度
    ttft_limit = thresholds.ttft[idx] ?? slo.ttft_target_s × 1000    # 桶门限缺失 → absolute 保底
    tpot_limit = thresholds.tpot[idx] ?? slo.tpot_target_ms

    ttft_ok = (row.ttft is not None) and (row.ttft ≤ ttft_limit)   # TTFT 缺失 → 判不达标
    if   row.completion_tokens < min_completion_tokens:  tpot_ok = True   # 短输出免检（分母太小、噪声极大）
    else:                                                tpot_ok = (row.tpot is None or row.tpot ≤ tpot_limit)
                                                         # 无 TPOT 时序（非流式）也不判超标

    return A2 if (ttft_ok and tpot_ok) else A3
```

| 类别 | 判据 |
| --- | --- |
| A1 | `status != "ok"`（HTTP 错误含 429/503、连接失败、整体超时、流错误、空输出，共 10 种子类型） |
| A2 | TTFT ≤ 桶门限 **且** TPOT ≤ 桶门限 |
| A3 | 成功但 TTFT 或 TPOT 超门限 |

四个容易读错的点：

- **不是统一一把尺**：每个桶有自己的一对门限（§7.5），全平台统一的只有
  "系数"（×3.0 / ×2.0）和模式。同一次运行里各桶达标线可以差一个数量级——
  本次实测 TTFT 线：≤1024 桶 2000ms（该桶薄，回退 absolute）、
  1025-4096 桶 581ms（基线 p99 × 3）、>4096 桶 2296ms（基线 p99 × 3）。
- **分桶键是实际长度**（服务端 usage），所以"计划 2048、实测 900"的请求
  会去比 ≤1024 桶那把更松的尺。这既符合语义（prefill 真实算力按实际
  token 计），也意味着 `chars_per_token` 估偏会改变判定归属，报告第八节
  的长度偏差（本次 -14.3%）要连同这一条一起读。
- **A3 的两条豁免**：短输出免检 TPOT、缺 TPOT 时序免检 TPOT——两者都只会
  让 A2 变多、不会误伤成 A3；反向的严格点是 TTFT 为 None 时直接判不达标。
- **分类结果的写入时机**：正式级在请求结束时回填 `latency_class` 落库，
  基线阶段刻意不回填（§7.4.2 规则 1）；聚合时优先读已落库标签，缺省才
  现场重算，因此 `--report-only` 与实时跑同一结论（C3）。

### 7.6.2 第二层：级汇总（compute_stage_stats）

每级（或 Phase 0）汇总出统计量，**N 不含 aborted**（存活者偏差控制，[04](04-measurement-validity.md) §4.1）：

```
rows    = 本级请求明细（取数时已排除 is_warmup=1）
aborted = [r for r in rows if r.status == "aborted"]     # 本级结束仍在途、drain 超时被放弃
rows    = rows − aborted
n_total = len(rows)                                      # ← SLA/SLO 与水位唯一的分母

A1/A2/A3 = 逐条 classify(rows) 计数                       # 优先读已落库的 latency_class

SLO = A2 / n_total                       # 只有"成功且双达标"才算质量
SLA = (A2 + A3) / n_total ≡ 1 − A1 / n_total   # 成功即算可用，只有 A1 拉低它
```

三条读数纪律：

- **aborted 不进任何比率与分位数**，单列 `n_aborted`。它不是失败也不是成功，
  计入任一侧都会系统性失真（被放弃的恰是最慢的一批）。
- **A1=0 时 SLA 恒为 1**，与 SLO 归零并存不矛盾：闭环施压下一级请求数有上限，
  排队（A3）而非拒绝（A1）消化压力时，"可用性 100% + 质量 0%"是正常读数，
  此时容量结论只能以 SLO 为准。
- **分位数与吞吐的分母口径不同于 SLA/SLO**：TTFT/TPOT/E2E/connect 的
  p50~p99 只在 `status == "ok"` 的行上算；RPS 与输出 tok/s 用请求自身墙钟
  窗口 `[min(sent_at), max(sent_at + e2e)]` 做分母（含 drain 尾巴），不是
  名义稳压窗口——否则跨窗口完成的请求会让吞吐被低估。

**吞吐侧基础指标**（与延迟侧面同批产出——一次采集、两侧消费，§7.10）：

| 量 | 定义 | 状态 |
| --- | --- | --- |
| `rps` | 全量发出速率 = n_total / window（**含失败**） | 已有 |
| `ok_rps` | 有效完成速率 = n_ok / window（只计成功） | **新增** |
| `output_tps` | Σcompletion_tokens / window（失败贡献 0） | 已有 |
| `input_tps` | Σprompt_tokens / window（按成功请求） | **新增** |
| `total_tps` | input_tps + output_tps（全量 token 工作量） | 读取时派生，**不落列** |

`rps` 与 `ok_rps` 必须分列：全部快速失败时 `rps` 反而最高（fast-fail 刷吞吐），
容量结论只认 `ok_rps` 系。

同时产出：A1 的 `fail_kinds` 直方图（拒绝族/超时族拆分的数据源）、按桶拆分
（`stage_buckets`：桶内 A1/A2/A3、超标侧、桶 p99、生效门限快照）、水位状态
（§7.6.4）。这一层全部落库（新增 `ok_rps`/`input_tps` 两列，旧库缺列时
守卫读回 + 报告期推导兜底，§7.7），是 report-only 重建的唯一数据源。

### 7.6.3 第三层：结论级——人工价值目标 vs 实测边界

| | 输入（人工） | 输出（实测） | 代码 |
| --- | --- | --- | --- |
| 价值目标 | `sla_target`（几个 9，默认 0.999）、`slo_target`（达标率，默认 0.95）、`ttft_factor`/`tpot_factor`（退化预算 3×/2×） | — | `config.py` |
| 实测边界 | 判据参数（水位软/硬门限：A3≥5%、A1≥1%、A3≥50%，公式见 §7.6.4） | **A3 拐点 = 质量/SLO 边界**（首个 BUSY 级）、**A1 拐点 = 可用性/SLA 边界**（首个 OVERLOAD 级） | `find_service_levels` |
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

### 7.6.4 水位判定：级内公式与跨级三边界

水位是服务端内部水位的**可观测投影**（[05](05-watermark.md)）：黑盒客户端看
不到 KV Cache / 批 slot / GPU，只能用"是否出现失败、是否出现不达标"反推
服务端在排队还是在拒绝。**级内判定**（`classify_watermark`，只用本级聚合量）：

```
a1_ratio = a1 / n_total
a3_ratio = a3 / n_total

若 n_total == 0                                                -> unknown（不参与边界发现）
若 a1_ratio ≥ hard_reject_ratio(0.01)
   或  a3_ratio ≥ hard_slo_violation_ratio(0.50)               -> OVERLOAD   ← 先判，优先级最高
若 a3_ratio ≥ soft_slo_violation_ratio(0.05)
   或  (soft_ttft_p99_ms > 0 且 TTFT_p99 ≥ soft_ttft_p99_ms)   -> BUSY       ← 软门限（排队预警）
否则                                                            -> NORMAL
```

- **OR 逻辑 + 优先级**，不是打分制：任一硬条件命中即 OVERLOAD，硬条件全否
  才看软条件。多个条件同时命中时**触发原因逐条累积**，报告的"触发原因"列
  因而可能有两行——这比只留一条更可解释（运维能分清是质量崩塌还是出现拒绝）。
- `soft_ttft_p99_ms` 默认 0 = **该条关闭**。因为"TTFT 绝对值多少算排队"与
  引擎/模型强相关，没有跨部署的合理默认；要用就得按本部署显式设定。默认
  配置下 BUSY 只由 A3 占比驱动。
- **A1 触发硬水位时再做家族归因**（`split_fail_kinds`）：超时族 > 拒绝族 ⇒
  排队型过载（Router 疑似无限队列，过载只投影为客户端超时）；拒绝族 > 超时族
  ⇒ 拒绝型过载（429/5xx 明确回绝）。归因不改变水位状态，只改变治理建议方向
  （扩容/调度 vs 熔断/准入）。
- 分母与 SLA/SLO 同一个 `n_total`（§7.6.2），因此水位状态与本级 SLO 读数
  必然自洽：`a3_ratio = 1 − SLO`（在 A1=0 时严格成立）。

**跨级三边界**（`find_watermarks`，与 §7.6.3 的拐点判据同源）：

```
judged = 按并发升序、剔除 unknown（无完成样本）后的级

软水位边界（排队预警线） = judged 中首个 state ∈ {BUSY, OVERLOAD} 的并发
硬水位边界（熔断线）     = judged 中首个 state == OVERLOAD 的并发
SLA 水位（生产并发建议） = find_capacity 的 quality_capacity      # 直接引用，不另算
```

取"首个"而非"最高"同样是连续前缀思路：个别级别抖动不应被当成边界。两条
反常形态会由 notes 明确喊出：硬边界不晚于软边界（压力跳过排队区直接触发
拒绝，提示准入策略激进或无限队列）；全程 NORMAL 未触达软边界（只说明梯度
上限内没观测到排队信号，不等于服务没有软水位）。

## 7.7 数据模型

SQLite 单文件，五张表（`store.py::SCHEMA`）：

| 表 | 粒度 | 内容 | 谁写 |
| --- | --- | --- | --- |
| `meta` | 键值 | 运行元信息（配置摘要、版本、时间戳） | store |
| `stages` | 每级一行 | 全部 §7.6.2 汇产量（含 `ok_rps`/`input_tps`，可空）+ `watermark_state`/`watermark_reasons_json`/`fail_kinds_json` + 两个时长口径 | runner |
| `stage_buckets` | 每级×每桶一行 | 桶内 A1/A2/A3、超标侧拆分、桶 p99、**生效门限快照** | runner |
| `baseline` | Phase 0 每桶一行 | 分位数 + n/n_ok/a1 + 两个时长口径 | runner |
| `requests` | 每条请求一行 | 特征（长度/输出）、时延四列（`connect_ms`/`ttft_ms`/`tpot_ms`/`e2e_ms`）、`status`/`error_kind`/`latency_class`/`is_warmup`/`sent_at` | runner |

三个设计决定：

- **只存特征与统计量，不存原始请求/响应内容**——压测数据库可能含生产
  prompt 内容，落库即扩散。
- **生效门限按桶快照**（`stage_buckets.ttft_limit_ms/tpot_limit_ms`）：
  换配置重放旧库时，读者能看到当时用的哪把尺子。
- **`_migrate` 向前兼容**：旧库缺列时补齐默认值，`--report-only`
  照常重建（水位/家族归因等后加功能对旧库同样可算，因为判据只用
  聚合量与 `fail_kinds_json`）。

两个容量场景增量（§7.10）：

- **`stages` 加 `ok_rps`/`input_tps` 两列**（可空）：沿用 `_migrate` 加列
  模式与 `"n_aborted" in keys` 同款守卫读回；旧库缺列时报告期推导兜底
  （`ok_rps = rps × n_ok/n_total`；`input_tps = prompt_tokens_avg ×
  ok_rps`），新旧库同结论。`requests` 表**零改动**——每请求事实单一
  出处原样覆盖吞吐分析。
- **容量面不进库**：`throughput_model.npz` 是分析层工件，由报告期从
  `stages` 现算汇写、`--report-only` 可重建——与"只存事实、派生量现场
  重算"（C3）一致，不为它加表。

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

**第二渲染器 `report_batch.py`**（容量分析，§7.10.6）：与 `report.py` 共享
结论构建骨架与格式化原语（`_table`/`_pct`/`_num` 等抽为可导入小模块，在线
渲染行为不变），差异全在章节渲染器——头部边界行、结论（SLA=RPM\*、SLO=全量
TPS\* vs 窗口承诺）、容量表、膝点证据、A1 家族归因、建议。报告期从同一批
StageStats 现取现算，不落库、`--report-only` 一致。两份报告可同时产出
（`report.kinds=["online","throughput"]`）。

## 7.9 配置体系

九个配置段（`config.example.toml` 有完整注释版），按"人工输入什么"分组：

| 段 | 人工输入 | 默认要点 |
| --- | --- | --- |
| `service` | 被测服务连接 | `timeout_s=600`（分钟级请求必须足够大） |
| `slo` | 质量价值目标：退化预算系数 + 契约门限 | `target_mode=baseline_relative`、3×/2× |
| `load` | 压力施加方式 + Phase 0 参数 | 闭环、`[1..128]` 梯度、基线并发 1/30s |
| `requests` | 流量画像（长度档/权重/分桶/上下文上限） | 留空则按 `max_context_tokens` 推导 |
| `analysis` | 价值目标：几个 9 + 达标率 + 样本门限 + 窗口承诺 | 0.999 / 0.95 / 60 / 30；`window_requests`/`window_hours` 默认 0（不设） |
| `watermark` | 水位判据门限 | 软 5% / 硬 50% / 拒绝 1% |
| `qos_model` | 预测模型超参 | 启用、`[12,12]`、120 epoch、150s 预算 |
| `storage` / `report` | 输出路径 + **产出哪些报告**（`report.kinds`） | 空则由 CLI 时间戳派生；`kinds=["online"]`（现状），加 `"throughput"` 走 `report_batch.py` + 容量面 |

类型宽容策略（`_coerce`）：float 字段接受 int 字面量（TOML 的 `150`
与 `150.0` 数值等价），int 字段严格（`120.5` 报错不截断），bool 显式
排除（`true` 不会静默变成 1）。

**容量/批量分析不新增配置段**：固定长度 = `requests` 单长度档 + 权重 1.0，
开环 = `load.mode="open"` + `target_rps`，窗口承诺 = `analysis` 两个字段，
报告选择 = `report.kinds`——全部复用既有段。预设文件
`config.throughput.toml`（与 `config.example.toml`/`config/config.mock-infinite-queue.toml`
同级）只是这些既有段的取值组合，不是新机制。

## 7.10 批量/容量场景：一次采集，两套分析

> 本节是最终设计（由议题提案按评审意见合并而来，历史版本从略）。范围：
> **只测推理服务本身的容量**——不涉文件上传/下载、作业生命周期与上层批量
> 调度前端（§7.10.4）；时间尺度为分钟级（每级 `stage_duration_s` 量级，
> 全阶梯 10-20 分钟），不做小时级稳态。

### 7.10.1 核心重构：测量与场景分析解耦

压测工具只做测量，不做场景裁剪。一次运行把请求级事实全量落库——时延四列
（`connect_ms`/`ttft_ms`/`tpot_ms`/`e2e_ms`）、双 token 计数
（`prompt_tokens`/`completion_tokens`）、`status`/`error_kind`——阶段聚合
同时产出**延迟侧面**与**吞吐侧面**两组基础指标。在线分析与容量分析消费
同一批数据，各自产出报告与模型工件：

```
   一次采集（runner + store + metrics，零场景分支）
        │  请求级事实（requests 表）+ 阶段级双侧面指标（stages 表）
        ▼
   ┌───────────────────┬─────────────────────┐
   ▼                   ▼                     │
 在线分析             容量分析                │
 report.py           report_batch.py          │ 两个分析都可选、
 qos_model.npz       throughput_model.npz     │ 亦可同时产出
 （延迟报告）         （吞吐报告 + 容量面）    │ （report.kinds）
```

（图示：采集管道只有一份；两种分析消费同一批数据，各自产出报告与工件。）

三条推论：

1. **不存在"批量模式"开关**。场景差异只在两处：负载形状配置（长度档 /
   控制循环 / 阶梯，全部是既有配置项）与分析选择（`report.kinds`，§7.9）。
   测量管道对两种场景完全同一。
2. **"批量场景不测延迟"是错误框架的残留**。时延采集成本为零，裁剪只发生在
   分析消费端——容量分析不引用质量维度，不等于不采。早期提案里"关闭延迟
   门限"因此不再需要：`classify` 照常跑（A3 照常计数），吞吐报告不引用
   A3/SLO 质量读数。
3. **一次运行可同时产出两份报告**（`report.kinds=["online","throughput"]`），
   `--report-only` 还能事后从旧库补出当时没选的那份——事实全量落库是
   这一切的前提（C3）。

### 7.10.2 基础指标集：一次采集，两侧消费

**请求级**（`requests` 表，逐请求事实，零改动）：

| 字段 | 侧面 |
| --- | --- |
| `connect_ms` / `ttft_ms` / `tpot_ms` / `e2e_ms` | 延迟 |
| `prompt_tokens` / `completion_tokens` / `token_source` | 吞吐（token 双计数是一等基础数据） |
| `status` / `error_kind` / `latency_class` | 两侧（分类的事实来源） |

**阶段级**（`stages` 表 / `StageStats`，一次聚合两侧消费）：

| 量 | 定义 | 状态 |
| --- | --- | --- |
| TTFT/TPOT/E2E/connect 分位（p50~p99） | 仅 `status=="ok"` 行 | 已有 |
| A1/A2/A3 计数、`sla`/`slo` | §7.6.2 口径 | 已有 |
| `rps` | 全量发出速率 = n_total / window（**含失败**） | 已有 |
| `ok_rps` | 有效完成速率 = n_ok / window（只计成功） | **新增** |
| `output_tps` | Σcompletion_tokens / window（失败贡献 0） | 已有 |
| `input_tps` | Σprompt_tokens / window（按成功请求） | **新增** |
| `total_tps` | input_tps + output_tps（全量 token 工作量） | 读取时派生，**不落列** |

input 与 output 都是一等基础数据（预填充与解码两侧的容量归因），total 是
便利派生而非唯一口径。`rps` 与 `ok_rps` 必须分列：全部快速失败时 `rps`
反而最高（fast-fail 刷吞吐），容量结论只认 `ok_rps` 系。

### 7.10.3 吞吐口径的服务等级

容量分析不消费质量维度，A3 恒不进入结论，比率 `SLA=(A2+A3)/N` 与
`SLO=A2/N` 退化为同一数值——服务等级改由吞吐单位承载，且同在 **A1 拐点**
（`find_service_levels.availability_boundary`；加可选参数 `a1_onset=True`
取"首个 A1 即膝点"，默认 False，在线行为一字不变）这一个测量点：

| 口径 | 定义 | 单位 |
| --- | --- | --- |
| **SLA**（可用性/容量） | `ok_rps(c*) × 60` | RPM（请求/分钟） |
| **SLO**（目标/工作量） | `input_tps(c*) + output_tps(c*)` | tokens/秒（全量） |
| 诊断 | `input_tps(c*)` / `output_tps(c*)` 分解 | prefill-bound vs decode-bound 判读 |

膝点处 λ\*≈μ\*（提交速率等于完成速率），"最大可持续负载"与"饱和吞吐"数值
一致。**容量读数取峰值（诚实口径，E2E 逼出的修正）**：上表 `ok_rps(c*)` 成立
的前提是"膝点处吞吐即峰值"——排队型饱和确实如此；但**硬拒绝型**服务（有界
队列 + 429/503）在膝点级即坍塌（该级大部分请求被拒），`ok_rps` 反而远低于
低压峰值。因此容量读数级定义为**膝点级及低压各级中 `ok_rps` 的峰值所在级**
（`report_batch.capacity_stage`）：理想膝点处二者一致；坍塌时取低压峰值，
并在结论中显式说明（"容量读数 @ 负载 X，膝点级 ok_rps 已受过载回落"），
不把坍塌值当成容量。容量面 `req_conc` 与读数级同源（§7.10.5）。
三条纪律：

1. **条件性**：μ 是（负载水平，输入长度）的函数，每个吞吐数字必须带负载
   坐标；报告以容量表逐级呈现，不做单一标量结论。
2. **窗口承诺**：人工价值目标的形式从"几个 9"变为窗口承诺
   （`analysis.window_requests` / `window_hours`，0 = 不设）：24h 内
   100 万请求 → 反推 μ ≥ 694 RPM，加安全余量得验收线；TPS 侧由 token
   预算（请求数 × (L+M̄)）同理折算。
3. **诚实退化**：TPS 的完整性取决于服务端 usage 上报；`token_source`
   显示缺失时报告降级为 RPM 单口径并标注 token 覆盖率，不静默填 0。

Phase 0 基线不进入容量分析（容量承诺是绝对值，相对基线尺子没有消费方）；
纯容量运行可 `load.baseline_enabled=false` 省时——这只是少跑低压级，
不少采任何事实；此时 `target_mode` 按现有规则回退 absolute 并提示
（§7.3 既有行为）。

### 7.10.4 测量边界与范围声明

- 工具测的是 **`service.base_url` 指向的推理服务本身**。报告头记录
  base_url、运行时刻与配置快照摘要；每个吞吐数字都可追溯到这个边界。
- **上层系统（如批量调度前端）的叠加影响不在本工具范围**：其自有队列、
  调度与轮询开销一般使对客可承诺值**低于**本工具实测；如需那个数字，
  属另立测量，不冒充本工具结论。
- 过载形态随服务端队列策略变化，由 A1 家族归因解读（D17）：拒绝族
  （429/5xx，有界队列）vs 超时族（客户端超时，无限队列吸收）——分类
  规则不变，报告按家族给治理建议。

### 7.10.5 容量面模型：`throughput_model.npz`

膝点只是本次运行的一个**点**；容量规划要反问题："扛 X RPM @ 长度 L 需要
多少并发？"把多次容量运行（每次一个固定长度）的膝点汇成容量面并持久化：

| 方面 | 设计 |
| --- | --- |
| 与在线模型的分工 | `qos_model.npz` 预测**没测过的负载点的质量**（MLP，[06](06-qos-model.md)）；`throughput_model.npz` 回答**容量规划反问题**（表格 + 插值，不训练） |
| Schema | `lengths`/`mu_rpm`/`mu_tps`/`input_tps`/`output_tps`/`req_conc`/`n_samples`/`a1_family`（均 (n,) 数组）+ `meta_json`（kind、base_url、生成时刻、每点配置摘要、token 覆盖率） |
| `req_conc` 语义 | 该长度处**取得峰值吞吐的负载水平**（闭环运行即并发数），与容量读数级同源（§7.10.3 峰值口径）；`--capacity-plan` 输出的"所需并发"是**参考并发**（该长度处取得峰值时的负载水平），不是精确最优解——反问题问"扛得住要多少"，v1 由既有测量点直接回答，不做事后优化 |
| 载体 | `npz.py` 零改动（纯机制，`meta_json` 走 bytes 通道与在线同款） |
| 写入 | 报告期由 `report_batch.py` 汇写（派生量不落库，C3）；`--report-only` 可从库重建；路径经 `--throughput-model` 指定 |
| 合并 | 同一文件内按 L 覆盖更新（replace-per-L），meta 记录来源运行；不同边界/配置的面对比用两个文件，不混装 |
| 查询 | `--capacity-plan <npz> --target-rpm X --input-len L`（平行于 `--qos-predict`）：输出该 L 处预测 μ（RPM/TPS 分解）、所需并发、余量；L 超出测量范围时 clamp 到端点并标 `extrapolated`（对应在线模型的包络守卫角色） |
| 使用 | ① 容量规划反查 ② CI 门禁工件（用绝对验收线，不用插值面）③ 多长度/多配置运行的容量汇总与趋势 ④ 二期运行护栏（实测 `ok_rps` 持续低于预测 μ×安全系数时早警） |
| 不上 MLP 的理由 | v1 的反问题由 `req_conc` 曲线直接回答；2-D 面 μ(c,L) 是优化视角、收益递减，列为二期可选（届时复用 `mlp.py` 方法论，换 `FEATURES`/`TARGETS` 定义） |

### 7.10.6 吞吐报告：`report_batch.py`

结构（共享 `report.py` 的结论构建骨架与格式化原语，后者抽为可导入小模块、
在线渲染行为不变）：

| 章 | 内容 | 计算位置 |
| --- | --- | --- |
| 〇、头部 | 边界行（base_url + 运行时刻 + 配置快照摘要） | 配置与 meta |
| 一、结论 | SLA(RPM\*)/SLO(全量 TPS\*) vs 窗口承诺验收线；达标判定；input/output 分解的瓶颈判读 | 报告期现算 |
| 二、容量表 | 每级负载水平 → ok_rps/total_tps/input_tps/output_tps/A1 占比/样本数；`rps` 全量列单独标注 | StageStats 现列 |
| 三、膝点证据 | 首个 A1 级（`a1_onset`）+ 平台佐证（连续两级 `ok_rps` 增幅 <5%） | 报告期现算 |
| 四、A1 归因 | `error_kind`/`fail_kinds` 家族拆分（D17） | 现成机制 |
| 五、建议 | 扩容 / 负载调参 / 窗口重谈，按归因分化 | Finding + stages |

```
c* = findings.availability_boundary            # a1_onset=True（首个 A1 级）
cap = capacity_stage(stages, c*)                # 膝点级及低压各级 ok_rps 峰值所在级（§7.10.3）
SLA = cap.ok_rps × 60                           # RPM
SLO = cap.input_tps + cap.output_tps            # 全量 TPS
判定 = SLA ≥ window_requests/(window_hours×60) × safety_factor
```

### 7.10.7 架构决策与隔离规则（已定，不再复议）

**决策：单一采集管道 + 可插拔分析/渲染**。曾评估"入口两模式"方案：此时
要分离的只剩两个模块，隔离收益趋零；而共享收益——一份 runner/metrics/
store、同库两种分析、`--report-only` 事后补报告——无法通过复制取得。
三条隔离规则：

1. **无场景分支**：`runner.py`/`client.py`/`metrics.py`/`store.py` 禁止
   出现场景判断；差异只经配置（长度档 / `mode` / `report.kinds` /
   `base_url`）与报告期表达。测试断言守护：共享模块不得 import
   `report_batch`/`throughput`。
2. **依赖方向**：`report_batch`/`throughput` import 在线侧数据结构与
   格式化原语；在线模块零容量 import；`bench.py` 组合层是唯一同时知晓
   两种分析的地方。
3. **窄接口**：渲染器接口 = `(cfg, store, stats, findings, meta) → 报告
   路径`，与 `write_report` 同形；容量面接口 = `fit(points) → 面` /
   `query(面, 目标) → 答案 + 外插标记`。

**在线路径零影响（硬约束）**：本增强只允许加字段、加列、加模块、加 flag。
默认配置（`report.kinds=["online"]`）的输出与增强前逐字节一致，由回归
测试锁定；新列可空、旧库守卫读回（§7.7）。

### 7.10.8 分期实施与验收

| 阶段 | 内容 | 工作量 |
| --- | --- | --- |
| 0 指标与存储 | `ok_rps`/`input_tps` 聚合 + `stages` 两列 + `_stats_from_store` 守卫 | 1 天 |
| 1 报告与画像 | `report_batch.py` + 格式化原语抽取 + `config.throughput.toml` 预设 + `a1_onset` 参数 + `report.kinds` | 1-2 天 |
| 2 验证 | 单测（聚合/守卫/报告期计算）+ mock E2E（固定长度端到端）+ **真实服务冒烟**（验证前提：批量路径复用同一推理引擎）+ `--report-only` 新旧库一致 + 默认配置回归 | 1 天 |
| 3 容量面 | `throughput.py` + `throughput_model.npz` schema + `--throughput-model`/`--capacity-plan` | 1.5-2 天 |

每个阶段的提交必须满足 §7.12 的三件套（单元断言 / mock E2E /
`--report-only` 一致性）。v1 完成标志：`python bench.py --config
config.throughput.toml` 一条命令产出含边界行、每级 SLA/SLO、容量表、
膝点证据与达标判定的吞吐报告并落盘容量面；`--capacity-plan` 可对任意
（目标 RPM，长度）反查所需并发。合计 4.5-6 人天。

## 7.11 实现映射

| 设计概念 | 位置 |
| --- | --- |
| 分桶映射/标签 | `bench/baseline.py::bucket_of / bucket_label / bucket_count` |
| 请求特征采样（逐请求独立加权，§7.4.1） | `bench/prompts.py::PromptFactory.sample_spec / build` |
| Phase 0 编排（按桶延长、量化提示） | `bench/runner.py::_run_baseline / _baseline_shortfall_hints` |
| 基线统计（只统计不判定） | `bench/baseline.py::compute_baseline` |
| 门限解析（两模式 + 回退） | `bench/baseline.py::resolve_thresholds` |
| 请求级分类 | `bench/metrics.py::classify` |
| 级汇总 | `bench/metrics.py::compute_stage_stats` |
| 实测服务等级（两个拐点） | `bench/metrics.py::find_service_levels` |
| 目标口径容量（连续前缀） | `bench/metrics.py::find_capacity` |
| 级内水位判定（§7.6.4 公式） | `bench/metrics.py::classify_watermark` |
| 水位边界 | `bench/metrics.py::find_watermarks` |
| A1 家族拆分 | `bench/metrics.py::split_fail_kinds` |
| 压力编排（预热/延长/drain） | `bench/runner.py::_run_stage` |
| 存储与迁移 | `bench/store.py` |
| 结论与渲染 | `bench/report.py::build_conclusions / render_report` |
| 长度参数推导 | `bench/config.py::_resolve_length_defaults` |
| 吞吐侧聚合字段 | `bench/metrics.py::compute_stage_stats`（+`ok_rps`/`input_tps`） |
| 容量膝点（首个 A1） | `bench/metrics.py::find_service_levels`（`a1_onset=True`，默认 False） |
| 容量面构建/查询 | `bench/throughput.py`（fit/query/save/load，经 `npz.py`） |
| 吞吐报告 | `bench/report_batch.py`（格式化原语抽自 `report.py`） |
| 报告选择 | `bench/config.py`（`report.kinds`）+ `bench.py` 组合层 |
| 容量存储增量 | `bench/store.py`（`stages.ok_rps`/`stages.input_tps` 两列 + 守卫读回） |

## 7.12 设计决策摘要

方法论层面的 18 条决策（D1–D18）见 [00-overview.md](00-overview.md)，
每条都标注了所属章节。本文补充的是**架构层面**的约束，修改前需对照：

- 分层依赖方向不可逆（`qos` 不参与判定、`stats` 为打破循环而独立）；
- 所有派生量必须能由落库数据现场重算（C3），新增结论若需要新列，
  先检查能否用已有充分统计量推导（如 A1 家族用 `fail_kinds_json`）；
- 新增判定口径必须同时给出：单元断言、mock E2E（`mock_server.py`）、
  `--report-only` 一致性验证，三者缺一不可提交；
- 纯标准库约束（C2）下任何新依赖需单独论证（如 numpy 缺失是硬事实）；
- **容量增强不得改动在线路径**：只允许加字段、加列、加模块、加 flag；
  默认配置（`report.kinds=["online"]`）输出与增强前逐字节一致，回归
  测试锁定（§7.10.7）；
- **共享模块禁出现场景判断**：场景差异只经配置与报告期表达，测试断言
  守护 import 方向（§7.10.7）；
- **派生量不落库**：`total_tps`、`throughput_model.npz` 均由报告期从
  既有事实现算（§7.7）。
