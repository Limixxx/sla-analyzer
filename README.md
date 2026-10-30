# PD 分离推理服务质量（SLO/SLA）压测评估工具

针对 **Grove 模式部署的 PD 分离大模型推理服务**，通过**递增并发压力**逐步试探出系统的
SLA 与 SLO，并评估达到 **3 个 9 可用性（99.9%）** 所能承受的最高并发。

零第三方依赖（Python ≥ 3.11 标准库），可直接拷贝进集群 Pod 内运行。

---

## 1. 指标定义

大模型推理请求处理时间达分钟级，传统"成功率"无法反映质量，因此将请求三分类：

| 类别 | 定义 |
| --- | --- |
| **A1** | 失败请求：HTTP 错误（含 Router 过载拒绝 429/503）、连接失败、整体超时、流内错误、空输出 |
| **A2** | 成功请求，且 **TTFT ≤ 门限** 且 **TPOT ≤ 门限**（门限可按 prompt 长度分桶，见第 5 章） |
| **A3** | 成功请求，但不满足 TTFT / TPOT 门限（"慢但成功"） |

派生指标：

- **SLA（可用性）** = (A2 + A3) / 总请求数 —— 用户视角"请求有没有被正确服务"
- **SLO（质量）** = A2 / 总请求数 —— 用户视角"请求是否被快速服务"

**系统优化就是优化 A3**：A3 是成功但不达标的请求，压测的核心产出就是定位 A3 随并发增长的位置。

> ⚠️ **两个口径必须分开看**：SLA 达标只表示"接得住"，SLO 达标才表示"接得好"。
> 过载时若服务选择排队而非拒绝，会出现 **SLA 100% 但 SLO 归零** 的陷阱形态
> ——此时任何并发都"可用"，但没有任何质量可言。因此本工具的生产并发建议以
> **质量容量**（SLA 与 SLO 同时达标的最高连续前缀并发）为唯一基准，
> SLA 容量仅作为 3 个 9 可用性参考单独陈述。

时间口径（流式采集）：

- **TTFT** = 请求写出完成 → 首个含内容的 chunk
- **TPOT** = 首 token → 末 token 的平均每 token 间隔

## 2. 目录结构

```
benchmark/
├── bench.py                  # CLI 入口（压测 / 探针 / 基线定标 / 报告重建）
├── mock_server.py            # 本地模拟推理服务（工具自测用，不用于生产）
├── config.example.toml       # 完整配置示例（复制为 config.toml 修改）
├── config.mock-demo.toml     # 配合 mock_server 的自测配置（长度参数留空走自动推导）
├── requirements.txt          # 零依赖说明
├── bench/                    # 核心包
│   ├── config.py             #   配置加载、校验与长度参数自动推导
│   ├── prompts.py            #   按 token 预算合成 prompt（唯一前缀防缓存）
│   ├── httpclient.py         #   零依赖异步 HTTP/1.1 + SSE 客户端
│   ├── client.py             #   单请求执行、TTFT/TPOT 采集、失败分类
│   ├── stats.py              #   分位数（最近秩）与均值
│   ├── baseline.py           #   长度分桶、Phase 0 基线剖析、门限解析
│   ├── metrics.py            #   A1/A2/A3 分类、分位数、容量判定
│   ├── store.py              #   SQLite 存储（只存特征，不存原文）
│   ├── runner.py             #   Phase 0 + 递增并发编排（闭环/开环）
│   └── report.py             #   Markdown 评估报告生成
└── results/                  # 输出目录（db + report）
```

## 3. 快速开始

```bash
cd benchmark

# 1) 复制配置并按实际环境修改（至少改 base_url / model / max_context_tokens）
cp config.example.toml config.toml

# 2) 预检（几十秒）：验证连通性、鉴权、端点形态，并实测一次 TTFT/TPOT
python bench.py --config config.toml --probe

# 3) 正式压测：按 concurrency_steps 逐级施压
python bench.py --config config.toml

# 4) 查看报告
cat results/report_<时间戳>.md
```

命令行可覆盖关键参数而不改配置文件：

```bash
python bench.py --config config.toml \
    --base-url http://<router-svc>.<ns>.svc.cluster.local:8000 \
    --model <model-name> --api-key sk-xxx \
    --steps 8,16,32,64,128 --duration 90 --warmup 15 \
    --db results/run1.db --report results/run1.md
```

从已有数据库重建报告（改判定门限后复盘特别有用）：

```bash
python bench.py --config config.toml --report-only --db results/run1.db --report results/run1.md
```

更换推理引擎/模型后，可先快速重测基线（仅 Phase 0，不施压），确认各长度桶的
分位数与生效门限符合预期后再跑完整压测：

```bash
python bench.py --config config.toml --baseline-only
```

也可跳过 Phase 0 直接用 absolute 门限压测（配置为 baseline_relative 时会自动
降级为 absolute 并打印提示）：

```bash
python bench.py --config config.toml --no-baseline
```

## 4. 测试方法

0. **Phase 0 基线剖析**（`load.baseline_enabled=true` 时）：正式 ramp 之前用低压并发
   测量无压力时延，产出各长度桶分位数；仅 `baseline_relative` 模式下参与门限计算，
   absolute 模式下也执行（供报告的质量退化率诊断表使用）。
1. **递增并发**：按 `concurrency_steps` 从低压到高压逐级施压，每级分预热段与测量段
   （预热数据不计入统计）。闭环（closed）模式下每个 worker 串行发请求，
   并发升高时后端饱和会依次表现为：排队（TTFT/TPOT 抬升 → A3 增多）→ 拒绝（A1 增多），
   这正是试探系统容量边界的方式。
2. **请求特征合成**：按权重采样 prompt 长度与输出长度，附加唯一前缀规避前缀缓存；
   真实 token 数以服务端 `usage` 字段为准（落库 `token_source=usage`），
   `chars_per_token` 仅用于无 usage 时的兜底估算。
3. **容量判定（连续前缀规则）**：按并发升序扫描，分别定位 3 个 9（SLA 口径）、
   SLO 口径、以及两者同时达标的**质量容量**；遇到第一个样本充足且跌破目标的级别停止。
   样本不足 `min_samples_per_stage` 的级别不参与高置信度前缀，但会进入
   **低置信度参考容量**，防止低压优质级别被整体丢弃。
4. **低压自适应延时**：闭环模式下低压级别的样本产出速率受单请求 E2E 限制
   （并发 1、E2E≈3.5s 时 60s 仅约 17 个样本）。测量段样本不足时本级自动延长，
   上限 `load.max_stage_duration_s`。
5. **在途请求处理**：本级结束时仍未完成的请求记为"放弃"（`status=aborted`），
   单独计数、不计入 SLA/SLO 分母——否则最慢的请求被静默丢弃会造成存活者偏差
   （时延分位数与 A3 占比被系统性低估）。

## 5. 判定门限与基线定标

### 5.1 为什么 SLA 不动态化、SLO 门限可自适应

- **SLA（可用性目标）是业务契约**，一旦确定就不随测量结果变化——否则"目标随表现
  浮动"，验收失去意义。本工具中 `analysis.sla_target` 恒为固定配置值。
- **SLO 门限可以换尺子**，但**绝不能用本次运行观测到的分位数定标**（尤其是高压段）：
  那会让门限随压力同步抬升，SLO 恒约等于目标值——指标永真、无法验收；
  且高压段分位数本身已被排队污染（用病态定义正常）。

### 5.2 两种门限模式

| 模式 | 门限来源 | 适用场景 |
| --- | --- | --- |
| `absolute` | 固定配置值 `ttft_target_s` / `tpot_target_ms`，所有长度桶统一 | 业务验收契约，跨次运行可比 |
| `baseline_relative` | Phase 0 低压基线各长度桶 p99 × 系数（`ttft_factor` / `tpot_factor`） | 自动适配不同推理引擎/模型规模；换引擎重测基线即可，整条曲线自动平移 |

两种模式都不修改 SLA 目标；区别只在"用什么尺子量 A2/A3"。

### 5.3 Phase 0 基线剖析

正式 ramp 之前，工具先用 `load.baseline_concurrency`（必须低到不触发排队/拒绝）
测量一段无压力时延，产出各长度桶的 TTFT/TPOT 分位数。要点：

- 基线**只统计分位数与失败数，不对 A2/A3 判定**——基线是尺子的来源，再拿它判
  达标会形成循环论证；
- 样本不足 `analysis.min_samples_per_stage` 的桶，门限**回退为 absolute 配置值**
  并在日志与报告中显式标注，绝不静默使用不可信的分位数；
- 基线阶段出现失败请求会污染分位数，工具会在报告中提示降低基线并发或排查可用性；
- `--baseline-only` 可只跑 Phase 0：换引擎后先确认各桶分位数与生效门限，再跑完整压测。

### 5.4 长度分桶

TTFT 与 prompt 长度强相关（prefill 计算量近似线性），TPOT 与已生成上下文长度相关。
全局单一时延门限会让长 prompt 请求天然落入 A3，把"长度差异"混进"排队劣化"里。
`requests.length_buckets` 把请求按 prompt token 数分桶（如 `[1024, 4096]` →
≤1024 / 1025-4096 / >4096），每个桶有独立门限。报告中的 A3 按桶拆分与
**质量退化率**（各级 p99 ÷ 基线 p99，与门限无关）共同回答：
劣化是集中在长 prompt（指向 prefill 容量），还是各桶同步（全局排队/准入问题）。

## 6. 关键配置

| 参数 | 说明 | 建议 |
| --- | --- | --- |
| `service.base_url` | Router 入口地址 | Grove 模式下用 router serving Service DNS（hostNetwork 部署同样可达） |
| `service.timeout_s` | 单请求整体超时 | **必须大于最长合法请求时长**（分钟级请求建议 ≥ 600s），否则误判超时失败 |
| `requests.max_context_tokens` | 服务支持的上下文上限 | 从服务文档或 `--probe` 的 `/v1/models` 输出获取；prompt+output 超过会直接报错拒绝启动 |
| `slo.ttft_target_s` / `slo.tpot_target_ms` | 质量门限（absolute 口径） | 按业务约定；先用 `--probe` 实测单请求时延再定 |
| `slo.target_mode` | 门限来源 | `absolute` 验收用；`baseline_relative` 跨引擎适配用 |
| `slo.ttft_factor` / `slo.tpot_factor` | baseline_relative 的门限系数 | ≥ 1.0；系数越松达标越易，建议结合退化率表复核 |
| `requests.length_buckets` | prompt 长度分桶上界 | 与生产流量长度分布匹配；留空按 `max_context_tokens` 推导 `[ctx/32, ctx/8]` |
| `requests.prompt_token_targets` | prompt 长度分布 | **应镜像生产流量画像**；留空按 `max_context_tokens` 推导 `[ctx/64, ctx/16, ctx/4]` |
| `load.baseline_concurrency` / `baseline_duration_s` | Phase 0 基线参数 | 并发必须低于最小压力级；时长保证每桶样本 ≥ `min_samples_per_stage` |
| `analysis.sla_target` | 可用性目标 | 3 个 9 = `0.999`；注意它只是"接得住"，生产建议看质量容量 |
| `analysis.slo_target` | 质量目标 | 与 SLA 共同构成质量容量判定 |
| `load.concurrency_steps` | 并发梯度 | 覆盖预期拐点两侧 |
| `load.max_stage_duration_s` | 单级自适应延时上限 | 闭环低压级别样本不足时自动延长至此；默认 300s |
| `load.drain_timeout_s` | 在途请求等待上限 | **需大于最长 E2E**，否则最慢请求被记"放弃"导致时延低估 |
| `load.max_requests_per_stage` | 测量段请求数上限 | 防止高压级（秒级拒绝）样本爆炸 |
| `analysis.min_samples_per_stage` | 高置信度容量判定的样本门槛 | 结合 `max_stage_duration_s` 评估低压级可达性 |

完整字段见 `config.example.toml` 内注释。

## 7. 输出

**SQLite（`results/bench_<ts>.db`）**：

- `stages`：每级配置与聚合结果（A1/A2/A3、放弃数、SLA/SLO、分位数、RPS、输出 tok/s）。
  `kind='stage'` 为正式压力级，`kind='baseline'` 为 Phase 0 基线级（不参与容量判定）
- `requests`：逐请求**特征**（计划/实际 token 数、TTFT/TPOT/E2E、分类、失败类型），
  **不保存原始请求与响应内容**
- `stage_buckets`：每级按 prompt 长度桶的拆分（A2/A3 计数、超标归因、该桶生效门限）
- `baseline`：Phase 0 各长度桶的时延分位数，是 baseline_relative 门限的来源
- `is_warmup=1` 为预热数据，统计时排除
- `status='aborted'` 为本级结束时仍在途、超过 drain 上限被放弃的请求，不计入比率

**Markdown 报告（`results/report_<ts>.md`）**：结论（质量容量 + 3 个 9 口径 + SLO 口径）
→ 基线剖析与生效门限（含质量退化率表）→ 方法 → 分级结果（含 SLO 判定列）→ 延迟分布
→ A1 构成 → A3 归因（TTFT/TPOT/双超标 + 按长度桶拆分）→ 特征覆盖（含计划 vs 实测
长度偏差告警）→ 建议 → 附录。

## 8. 工具自测（无需真实服务）

```bash
# 终端 1：启动模拟服务（soft=超过后开始排队，hard=超过后直接 503 拒绝；
#         --ttft-per-1k-tokens 让 TTFT 随 prompt 长度增长，用于验证按桶门限）
python mock_server.py --port 8123 --ttft-ms 200 --tpot-ms 15 --ttft-per-1k-tokens 60 \
    --soft-capacity 8 --hard-capacity 12

# 终端 2：跑自测配置
python bench.py --config config.mock-demo.toml
```

自测已验证的行为：低并发 SLA/SLO=100%；超过 soft 容量后 TPOT 抬升落入 A3
（SLO 降而 SLA 不降）；超过 hard 容量后大量 503 落入 A1（SLA 断崖）；
报告以**质量容量**定位生产并发（SLA 100% 但 SLO 归零的级别不会被推荐）。

`config.mock-demo.toml` 中的 `requests` 段故意只配 `max_context_tokens` 与输出长度，
prompt 长度分布与长度分桶均**留空**——用于验证按上下文上限的自动推导
（8K 上下文 → prompt [128, 512, 2048]、桶 [256, 1024]），推导来源会在启动横幅与
报告"请求特征覆盖"一节标注。把 `[slo] target_mode` 改为 `baseline_relative`
并适当增大 `load.baseline_duration_s`，还可验证基线定标口径。

## 9. 使用注意与局限

- **超时口径**：`timeout_s` 是单请求整体超时。分钟级请求下若设小了，慢请求会被记为
  A1（timeout）而非 A3，直接污染 SLA——上线前务必用 `--probe` 实测最长请求时长。
- **样本量**：3 个 9 的判定需要足够样本（建议每级 ≥ 1000 条才能区分 99.9% 与 99.5%）。
  闭环低压级别样本天然少，工具会自动延长本级（上限 `max_stage_duration_s`）；
  仍不足的级别只作低置信度参考，报告中会明确标注。
- **排队型过载陷阱**：若服务过载时只排队不拒绝（A1=0、TTFT 持续抬升），
  SLA 会一直"达标"——此时请直接看 SLO 与质量容量结论，不要用 SLA 容量定生产并发。
- **前缀缓存**：`unique_prefix=true` 会加少量 token 换取无缓存干扰的 TTFT；
  若生产有关键前缀缓存优化，可关闭以测量"热"路径。
- **单机压力上限**：单进程闭环压测建议到数百并发；更高压力可多机运行、
  各自写独立 db 后合并 `requests` 表分析。
- **Router 行为依赖**：A1 中的 rejected 依赖 Router 返回 429/503（或报文含
  overload/queue full 字样）；若部署的 Router 用其他状态码表达过载，
  请在 `bench/client.py` 的失败分类处补充。
- **baseline_relative 的横向对比边界**：该口径下"达标"只对当次基线有意义——
  跨引擎/跨模型比较时应改用 absolute 口径（或并列两种口径），只对比报告中
  与门限无关的**质量退化率**与原始分位数。
- 本工具测量的是**客户端视角**的可用性与质量，不替代服务端监控
  （GPU 利用率、KV cache 占用、队列深度等仍需从集群侧采集）。
