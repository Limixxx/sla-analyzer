# PD 分离推理服务质量（SLO/SLA）压测评估工具

针对 **Grove 模式部署的 PD 分离大模型推理服务**，通过**递增并发压力**逐步试探出系统的
SLA 与 SLO，并评估达到 **3 个 9 可用性（99.9%）** 所能承受的最高并发。

零第三方依赖（Python ≥ 3.11 标准库），可直接拷贝进集群 Pod 内运行。

> **方法论文档**：`docs/` 目录记录本工具的理论基础（指标体系、容量模型、
> 门限定标、测量有效性、水位语义），回答"为什么这样设计"；本 README 只讲"怎么用"。

---

## 1. 指标定义

大模型推理请求处理时间达分钟级，传统"成功率"无法反映质量，因此将请求三分类：

| 类别 | 定义 |
| --- | --- |
| **A1** | 失败请求：HTTP 错误（含 Router 过载拒绝 429/503）、连接失败、整体超时、流内错误、空输出。Router 配无限队列时过载无 429/529、只表现为客户端超时——分类上仍是 A1，归因上按拒绝族/超时族区分（见 [docs/05](docs/05-watermark.md) 2.1） |
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

时间口径（流式采集，全部基于单调时钟）：

- **TTFT** = 客户端视角的首 token 等待：从建连前起算，含 TCP/TLS 建连与请求写出
  （用户实际感知的等待）。建连开销单独记录为 `connect_ms`，报告在延迟分布一节
  给出各级 p50/p99——跨机房/TLS 部署下占比可观时应予扣除
- **TPOT** = 首 token → 末 token 的平均每 token 间隔

## 2. 目录结构

```
benchmark/
├── bench.py                  # CLI 入口（压测 / 探针 / 基线定标 / 报告重建）
├── mock_server.py            # 本地模拟推理服务（工具自测用，不用于生产）
├── config.example.toml       # 完整配置示例（复制为 config.toml 修改）
├── config.mock-demo.toml     # 配合 mock_server 的自测配置（长度参数留空走自动推导）
├── config.mock-infinite-queue.toml  # 无限队列场景自测（无 429/529，过载投影为客户端超时）
├── requirements.txt          # 零依赖说明
├── bench/                    # 核心包
│   ├── config.py             #   配置加载、校验与长度参数自动推导
│   ├── prompts.py            #   按 token 预算合成 prompt（唯一前缀防缓存）
│   ├── httpclient.py         #   零依赖异步 HTTP/1.1 + SSE 客户端
│   ├── client.py             #   单请求执行、TTFT/TPOT 采集、失败分类
│   ├── stats.py              #   分位数（最近秩）与均值
│   ├── baseline.py           #   长度分桶、Phase 0 基线剖析、门限解析
│   ├── metrics.py            #   A1/A2/A3 分类、分位数、容量与水位判定
│   ├── qos.py                #   预测模型：数据集、分组 CV 训练、ECDF 校准、仿真
│   ├── mlp.py                #   纯 Python 多层感知机（Adam）
│   ├── npz.py                #   纯标准库 .npz/.npy 读写（numpy 互操作）
│   ├── store.py              #   SQLite 存储（只存特征，不存原文）
│   ├── runner.py             #   Phase 0 + 递增并发编排（闭环/开环）
│   └── report.py             #   Markdown 评估报告生成
├── docs/                     # 方法论文档（理论基础）
│   ├── 00-overview.md        #   总览、文档地图、方法论决策记录
│   ├── 01-metrics-slo-sla.md #   指标体系：A1/A2/A3、SLA/SLO、TTFT/TPOT
│   ├── 02-pd-capacity-model.md # 容量模型：PD 分离、两阶段过载、递增并发
│   ├── 03-threshold-calibration.md # 门限定标：基线定标、长度分桶、退化率
│   ├── 04-measurement-validity.md # 测量有效性：偏差识别与控制
│   ├── 05-watermark.md       #   水位语义：服务端水位的黑盒投影（软/硬/SLA）
│   └── 06-qos-model.md       #   预测模型：压测采样训练 MLP 预测 TTFT/TPOT
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

> `--duration` 会按 5 倍联动单级自适应延长上限（与默认配置 60s→300s 同比例），
> 避免快速验证场景（如 `--duration 6`）的单级仍按配置的 300s 上限延长；
> 需要其他上限时用 `--max-duration <秒>` 显式指定。

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

用训练好的模型做单点预测（无需 numpy；模型来自 `qos_model.save_path`
或 `--qos-model`）：

```bash
python bench.py --qos-model results/qos_model.npz \
    --qos-predict "concurrency=16,prompt_tokens=4096,completion_tokens=512"
# 输出：TTFT/TPOT 的 p50/p90/p99 预测（毫秒）+ 是否超出训练包络
```

关闭预测模型（仅评估、不训练）：`--no-qos`。

## 4. 测试方法

0. **Phase 0 基线剖析**（`load.baseline_enabled=true` 时）：正式 ramp 之前用低压并发
   测量无压力时延，产出各长度桶分位数；仅 `baseline_relative` 模式下参与门限计算，
   absolute 模式下也执行（供报告的质量退化率诊断表使用）。基线只统计、不判定。
1. **递增并发**：按 `concurrency_steps` 从低压到高压逐级施压，每级分预热段与测量段
   （预热数据不计入统计）。闭环（closed）模式下每个 worker 串行发请求，
   并发升高时后端饱和会依次表现为：排队（TTFT/TPOT 抬升 → A3 增多）→ 失败
   （A1 增多：拒绝型部署投影为 429/503，无限队列部署投影为客户端超时），
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
   上限 `load.max_stage_duration_s`。两种判据对应两种"够用"：
   **正式级按测量段总量**（`min_samples_per_stage`），
   **基线阶段按最短样本长度桶**（`baseline_min_samples`）——门限是按桶解析的，
   总量够而最短桶不够照样回退 absolute。延长会在日志中显式提示
   （"仍不足/已补齐" + 硬上限 + 含 drain 的实际运行时长），
   基线结束仍不足时给出"还差多少样本、约需多少秒、建议并发"的量化提示。
   实际时长（`actual_duration_s`）与配置时长分开记录，报告中同时展示。
5. **在途请求处理**：本级结束时仍未完成的请求记为"放弃"（`status=aborted`），
   单独计数、不计入 SLA/SLO 分母——否则最慢的请求被静默丢弃会造成存活者偏差
   （时延分位数与 A3 占比被系统性低估）。`sent_at` 记录真实发出时刻，
   保证时间戳可审计、吞吐窗口不被推晚。
6. **水位探测**：每级按可观测信号判定水位状态（见 5.5）：**NORMAL**（无信号）→
   **BUSY**（软水位投影：A3 占比或 TTFT p99 超软门限，服务在排队）→
   **OVERLOAD**（硬水位投影：出现拒绝或 A3 崩塌）。跨级汇总出软/硬/SLA
   三个水位边界，供 Router 配置排队预警线与熔断线。

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
- 样本不足 `analysis.baseline_min_samples` 的桶，门限**回退为 absolute 配置值**
  并在日志与报告中显式标注，绝不静默使用不可信的分位数；
- 样本不足时基线按"最短样本长度桶"自适应延长（判据与正式级不同，因为门限是
  按桶解析的），结束仍不足会给出"还差多少样本、约需多少秒、建议并发"的提示；
- 配置时长与实际时长分开记录（`duration_s` / `duration_actual_s`），
  报告中同时展示，避免"配置 30s"与"实际跑了 3 分钟"对不上；
- 基线阶段出现失败请求会污染分位数，工具会在报告中提示降低基线并发或排查可用性；
- `--baseline-only` 可只跑 Phase 0：换引擎后先确认各桶分位数与生效门限，再跑完整压测。

### 5.4 长度分桶

TTFT 与 prompt 长度强相关（prefill 计算量近似线性），TPOT 与已生成上下文长度相关。
全局单一时延门限会让长 prompt 请求天然落入 A3，把"长度差异"混进"排队劣化"里。
`requests.length_buckets` 把请求按 prompt token 数分桶（如 `[1024, 4096]` →
≤1024 / 1025-4096 / >4096），每个桶有独立门限。报告中的 A3 按桶拆分与
**质量退化率**（各级 p99 ÷ 基线 p99，与门限无关）共同回答：
劣化是集中在长 prompt（指向 prefill 容量），还是各桶同步（全局排队/准入问题）。

### 5.5 水位探测（服务端水位的黑盒投影）

推理水位（触发限流/排队/熔断/扩缩容的负载状态阈值集合，详见
[docs/05-watermark.md](docs/05-watermark.md)）分三层：**软水位**（BUSY，转入排队不拒绝）、
**硬水位**（OVERLOAD，直接 429 拒绝）、**SLA 水位**（业务承诺反向驱动）。

压测工具是黑盒客户端，看不到服务端的 KV Cache / 批 slot / GPU 等资源指标，
因此判定的是**服务端水位的可观测投影**——服务端水位一旦被触达，必然通过
排队与拒绝投影到客户端可见的信号上：

| 水位状态 | 判定（OR 逻辑，`[watermark]` 配置） | 服务端对应行为 |
| --- | --- | --- |
| NORMAL | 各项可观测指标均在软门限之内 | 资源有余量 |
| BUSY（软水位投影） | A3 占比 ≥ `soft_slo_violation_ratio`（默认 5%），或 TTFT p99 ≥ `soft_ttft_p99_ms`（0=关闭） | 转入排队/路由分流，不拒绝 |
| OVERLOAD（硬水位投影） | A1 占比 ≥ `hard_reject_ratio`（默认 1%），或 A3 占比 ≥ `hard_slo_violation_ratio`（默认 50%） | 拒绝接入（429/503）；无限队列部署下为排队耗尽、客户端超时 |
| unknown | 本级无完成样本 | 无法判定，不参与边界发现 |

报告给出三个水位边界：**软水位并发**（首个 BUSY 级，排队预警线）、
**硬水位并发**（首个 OVERLOAD 级，熔断线）、**SLA 水位并发**（= 质量容量，
生产并发建议）。BUSY/OVERLOAD 级附带 A3 归因，指向劣化平面：
TTFT 主导 → prefill 平面（算力/准入排队），TPOT 主导 → decode 平面
（KV Cache/批 slot），与 PD 分离的双水位语义对齐。

> 注意：这是水位的**投影**而非水位本身——软水位投影略滞后于服务端软水位触达
> （需积累足够 A3 样本才能分辨）。真正的资源水位计算在服务端
> （Prometheus 指标 + 监控分析器），两者互补而非替代。

### 5.6 服务质量预测模型（MLP）

评估的产物是**离散压力级上的阶跃观测**：只知 8 并发达标、16 并发崩塌，
中间与之外都是空白。`[qos_model]` 把压测采样（不同并发 × 不同请求长度）
同时用作训练数据，训练一个多层感知机，输入**请求数据特征 + 负载情况**
（并发、prompt 长度、输出长度），预测 TTFT/TPOT（方法论详见
[docs/06-qos-model.md](docs/06-qos-model.md)）：

- **入训样本**：非预热的 measured 请求，A3 必须保留（膝点区只有 A3 样本）；
  warmup / aborted / A1 失败样本剔除
- **训练协议**：目标 log 变换；**按 stage 分组 K 折**交叉验证（防同级样本
  泄漏）；折外残差汇成 ECDF，直接给出分位数与通过率
  `P(时延≤T) = F_resid(log T − ŷ)`，不假设分布族
- **包络守卫**：超出训练覆盖的（并发 × 长度）范围的预测标记"外推"，
  不参与任何结论；容量/水位边界在包络内细扫插值
- **模型文件**：`results/qos_model.npz`（ZIP + .npy，纯标准库读写），
  可被 numpy 生态直接 `np.load()`；`--qos-predict` 无需 numpy 即可单点预测

预测章节给出：预测 vs 实测对照（已测并发上偏差应在残差分位数内）、
参考流量下的 TTFT/TPOT 预测曲线、水位边界细扫（与实测边界互相印证）。
模型绑定当次服务配置，换引擎/模型必须重训；近膝点预测方差最大
（平滑网络涂抹膝点），容量结论仍以实测连续前缀规则为准。

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
| `load.baseline_concurrency` / `baseline_duration_s` | Phase 0 基线参数 | 并发必须低于最小压力级；时长保证每桶样本 ≥ `baseline_min_samples`（不足会自动延长并提示） |
| `analysis.sla_target` | 可用性目标 | 3 个 9 = `0.999`；注意它只是"接得住"，生产建议看质量容量 |
| `analysis.slo_target` | 质量目标 | 与 SLA 共同构成质量容量判定 |
| `load.concurrency_steps` | 并发梯度 | 覆盖预期拐点两侧 |
| `load.max_stage_duration_s` | 单级自适应延时上限 | 闭环低压级别样本不足时自动延长至此；默认 300s。正式级按总量、基线按最短桶判定 |
| `load.drain_timeout_s` | 在途请求等待上限 | **需大于最长 E2E**，否则最慢请求被记"放弃"导致时延低估 |
| `load.max_requests_per_stage` | 测量段请求数上限 | 防止高压级（秒级拒绝）样本爆炸 |
| `analysis.min_samples_per_stage` | 高置信度容量判定的样本门槛（级粒度） | 结合 `max_stage_duration_s` 评估低压级可达性 |
| `analysis.baseline_min_samples` | 基线各长度桶的样本门槛（桶粒度） | 按权重反推总量需求（默认 30 / 最短桶权重 0.2 ≈ 150 个总样本）；不足的桶回退 absolute |
| `watermark.enabled` | 是否判定水位状态 | 关闭后分级表无"水位"列、结论区无水位边界陈述（其余结论不受影响） |
| `watermark.soft_slo_violation_ratio` | 软水位（BUSY）门限：A3 占比 | 默认 5%；调小更早预警排队，调大更保守 |
| `watermark.hard_slo_violation_ratio` | 硬水位（OVERLOAD）门限：A3 占比 | 默认 50%（过半不达标 = 质量崩塌） |
| `watermark.hard_reject_ratio` | 硬水位（OVERLOAD）门限：A1 占比 | 默认 1%；服务端限流/排队耗尽不会零星出现（含客户端超时族） |
| `watermark.soft_ttft_p99_ms` | 软水位附加门限：TTFT p99 绝对值 | 默认 0（关闭）；开启后在 A3 尚少时提前报 BUSY，跨引擎对比时记得调整 |
| `qos_model.enabled` | 是否训练 MLP 预测模型 | 开启后压测结束自动训练并写 `.npz`；样本不足时跳过并在报告说明 |
| `qos_model.min_samples` | 最低训练样本量 | 默认 200；低于此值模型不可信，宁可不训 |
| `qos_model.hidden` | 隐藏层宽度 | 默认 `[12, 12]`；小数据下更大网络只会过拟合 |
| `qos_model.epochs` | 最大训练轮数（全批量 Adam） | 默认 120；受 `max_train_seconds` 硬预算截断 |
| `qos_model.max_train_seconds` | 训练时长硬上限 | 默认 150s；4000 样本全量验证约 95s，纯 Python 实现的兜底保护 |
| `qos_model.min_r2` | 验证 R² 下限 | 默认 0.5；低于此值报告只渲染数据口径、不出预测结论 |
| `qos_model.save_path` | 模型文件路径 | 默认 `results/qos_model.npz`；`--report-only` 时存在则直接加载 |

完整字段见 `config.example.toml` 内注释。

## 7. 输出

**SQLite（`results/bench_<ts>.db`）**：

- `stages`：每级配置与聚合结果（A1/A2/A3、放弃数、SLA/SLO、分位数、connect p50/p99、
  RPS、输出 tok/s、水位状态 `watermark_state`）。`kind='stage'` 为正式压力级，
  `kind='baseline'` 为 Phase 0 基线级（不参与容量判定）；`duration_s` 为配置时长、
  `actual_duration_s` 为实际时长（含自适应延长与 drain 收尾）
- `requests`：逐请求**特征**（计划/实际 token 数、connect/TTFT/TPOT/E2E、
  `latency_class` 分类、失败类型），**不保存原始请求与响应内容**。
  正式压力级的每条请求在落库时即回填 A1/A2/A3，可直接按类别检索
- `stage_buckets`：每级按 prompt 长度桶的拆分（A1/A2/A3 计数、超标归因、该桶生效门限）
- `baseline`：Phase 0 各长度桶的时延分位数（含实际时长），是 baseline_relative
  门限的来源
- `is_warmup=1` 为预热数据，统计时排除
- `status='aborted'` 为本级结束时仍在途、超过 drain 上限被放弃的请求，不计入比率
  （`sent_at` 仍为真实发出时刻）

**Markdown 报告（`results/report_<ts>.md`）**：结论（质量容量 + 3 个 9 口径 + SLO 口径
+ 水位边界 + 容量判定的补充说明）→ 基线剖析与生效门限（含按桶门限表"来源"列与质量退化率表）
→ 方法 → 分级结果（含 SLO 判定列与水位状态列）→ 延迟分布（含 connect 开销提示）
→ A1 构成 → A3 归因（TTFT/TPOT/双超标 + 按长度桶拆分）
→ 特征覆盖（含计划 vs 实测长度偏差告警）→ 服务质量预测模型（预测 vs 实测对照、
预测曲线、水位边界细扫、包络警告）→ 建议 → 附录。

**预测模型（`results/qos_model.npz`）**：ZIP + `.npy` 容器（权重、归一化参数、
残差 ECDF、包络、元数据），可被 numpy 生态 `np.load()` 直接消费；
纯标准库实现，`--qos-predict` 无 numpy 环境亦可推理。

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
还可验证基线定标口径：基线阶段按"最短样本长度桶"自适应延长
（门限 `analysis.baseline_min_samples`），样本仍不足的桶回退 absolute
并在日志中给出"还差多少/约需多久/建议并发"的量化提示。

**无限队列场景自测**（Router 无队列上限：无 429/529，过载投影为客户端超时）：

```bash
# 终端 1：无限队列模式（排队随超出量无上限增长，客户端先触达 timeout_s）
python mock_server.py --port 8123 --infinite-queue --queue-step-s 2.0 --soft-capacity 8

# 终端 2：客户端预算压到 10s，高压级全部客户端超时
python bench.py --config config.mock-infinite-queue.toml
```

预期：并发 ≤ 8 全部 NORMAL；并发 12 排队 ≈8s、成功但 TTFT 超标（BUSY）；
并发 ≥ 16 排队超 10s 预算，**全程无一条 429/503**，失败全部是 `timeout`
（A1 超时族）并触发 OVERLOAD。报告的过载形态、水位触发原因与治理建议
应按"排队型"而非"拒绝型"归因（见 [docs/05-watermark.md](docs/05-watermark.md) 2.1）。

## 9. 使用注意与局限

- **超时口径**：`timeout_s` 是单请求整体超时。分钟级请求下若设小了，慢请求会被记为
  A1（timeout）而非 A3，直接污染 SLA——上线前务必用 `--probe` 实测最长请求时长。
- **样本量**：3 个 9 的判定需要足够样本（建议每级 ≥ 1000 条才能区分 99.9% 与 99.5%）。
  闭环低压级别样本天然少，工具会自动延长本级（上限 `max_stage_duration_s`，
  正式级按总量、基线阶段按最短长度桶判定）；仍不足的级别只作低置信度参考，
  报告中会明确标注。
- **时延口径**：TTFT/E2E 为客户端视角（含建连与请求写出）。跨机房/TLS 部署下
  建连开销可能占 TTFT 相当比例——看报告延迟分布节的 connect p50/p99，
  必要时把压测客户端部署到靠近服务的网络位置。
- **排队型过载陷阱**：若服务过载时只排队不拒绝（A1=0、TTFT 持续抬升），
  SLA 会一直"达标"——此时请直接看 SLO 与质量容量结论，不要用 SLA 容量定生产并发。
  其极端变体是 **Router 无限队列**：服务永不拒绝，客户端先超时（A1 的 timeout 族
  堆积、全程无 429/529）。此时 SLA 同样断崖，但治理动作不是调高 `timeout_s`
  （那只会把可用性问题藏起来），而是给队列设上限/排队超时让过载可被观测
  （见 [docs/05-watermark.md](docs/05-watermark.md) 2.1）。
- **前缀缓存**：`unique_prefix=true` 会加少量 token 换取无缓存干扰的 TTFT；
  若生产有关键前缀缓存优化，可关闭以测量"热"路径。
- **单机压力上限**：单进程闭环压测建议到数百并发；更高压力可多机运行、
  各自写独立 db 后合并 `requests` 表分析。
- **Router 行为依赖**：A1 中的 rejected 依赖 Router 返回 429/503（或报文含
  overload/queue full 字样）；若部署的 Router 用其他状态码表达过载，
  请在 `bench/client.py` 的失败分类处补充。若 Router 配无限队列不过载拒绝，
  过载会表现为客户端超时（timeout 族），工具按 A1 正常统计并在报告中
  给出排队型归因，无需改代码。
- **baseline_relative 的横向对比边界**：该口径下"达标"只对当次基线有意义——
  跨引擎/跨模型比较时应改用 absolute 口径（或并列两种口径），只对比报告中
  与门限无关的**质量退化率**与原始分位数。
- 本工具测量的是**客户端视角**的可用性与质量，不替代服务端监控
  （GPU 利用率、KV cache 占用、队列深度等仍需从集群侧采集）。
