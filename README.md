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
| **A2** | 成功请求，且 **TTFT ≤ 目标** 且 **TPOT ≤ 目标** |
| **A3** | 成功请求，但不满足 TTFT / TPOT 门限（"慢但成功"） |

派生指标：

- **SLA（可用性）** = (A2 + A3) / 总请求数 —— 用户视角"请求有没有被正确服务"
- **SLO（质量）** = A2 / 总请求数 —— 用户视角"请求是否被快速服务"

**系统优化就是优化 A3**：A3 是成功但不达标的请求，压测的核心产出就是定位 A3 随并发增长的位置。

时间口径（流式采集）：

- **TTFT** = 请求写出完成 → 首个含内容的 chunk
- **TPOT** = 首 token → 末 token 的平均每 token 间隔

## 2. 目录结构

```
benchmark/
├── bench.py                  # CLI 入口（压测 / 探针 / 报告重建）
├── mock_server.py            # 本地模拟推理服务（工具自测用，不用于生产）
├── config.example.toml       # 完整配置示例（复制为 config.toml 修改）
├── config.mock-demo.toml     # 配合 mock_server 的自测配置
├── requirements.txt          # 零依赖说明
├── bench/                    # 核心包
│   ├── config.py             #   配置加载与校验（TOML/JSON）
│   ├── prompts.py            #   按 token 预算合成 prompt（唯一前缀防缓存）
│   ├── httpclient.py         #   零依赖异步 HTTP/1.1 + SSE 客户端
│   ├── client.py             #   单请求执行、TTFT/TPOT 采集、失败分类
│   ├── metrics.py            #   A1/A2/A3 分类、分位数、容量判定
│   ├── store.py              #   SQLite 存储（只存特征，不存原文）
│   ├── runner.py             #   递增并发编排（闭环/开环）
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

## 4. 测试方法

1. **递增并发**：按 `concurrency_steps` 从低压到高压逐级施压，每级分预热段与测量段
   （预热数据不计入统计）。闭环（closed）模式下每个 worker 串行发请求，
   并发升高时后端饱和会依次表现为：排队（TTFT/TPOT 抬升 → A3 增多）→ 拒绝（A1 增多），
   这正是试探系统容量边界的方式。
2. **请求特征合成**：按权重采样 prompt 长度与输出长度，附加唯一前缀规避前缀缓存；
   真实 token 数以服务端 `usage` 字段为准（落库 `token_source=usage`），
   `chars_per_token` 仅用于无 usage 时的兜底估算。
3. **容量判定（连续前缀规则）**：按并发升序扫描，记录最后一个 SLA 达标的级别；
   遇到第一个样本充足且跌破目标的级别停止。样本不足 `min_samples_per_stage` 的级别视为中立，
   不会错误地打断前缀。

## 5. 关键配置

| 参数 | 说明 | 建议 |
| --- | --- | --- |
| `service.base_url` | Router 入口地址 | Grove 模式下用 router serving Service DNS（hostNetwork 部署同样可达） |
| `service.timeout_s` | 单请求整体超时 | **必须大于最长合法请求时长**（分钟级请求建议 ≥ 600s），否则误判超时失败 |
| `requests.max_context_tokens` | 服务支持的上下文上限 | 从服务文档或 `--probe` 的 `/v1/models` 输出获取；prompt+output 超过会直接报错拒绝启动 |
| `slo.ttft_target_s` / `slo.tpot_target_ms` | 质量门限 | 按业务约定；先用 `--probe` 实测单请求时延再定 |
| `analysis.sla_target` | 可用性目标 | 3 个 9 = `0.999` |
| `load.concurrency_steps` | 并发梯度 | 覆盖预期拐点两侧；样本不足时调大 `stage_duration_s` |
| `load.max_requests_per_stage` | 测量段请求数上限 | 防止高压级（秒级拒绝）样本爆炸 |

完整字段见 `config.example.toml` 内注释。

## 6. 输出

**SQLite（`results/bench_<ts>.db`）**：

- `stages`：每级配置与聚合结果（A1/A2/A3、SLA/SLO、分位数、RPS、输出 tok/s）
- `requests`：逐请求**特征**（计划/实际 token 数、TTFT/TPOT/E2E、分类、失败类型），
  **不保存原始请求与响应内容**
- `is_warmup=1` 为预热数据，统计时排除

**Markdown 报告（`results/report_<ts>.md`）**：结论 → 方法 → 分级结果 → 延迟分布 →
A1 构成 → A3 归因（TTFT/TPOT/双超标）→ 特征覆盖 → 建议 → 附录。

## 7. 工具自测（无需真实服务）

```bash
# 终端 1：启动模拟服务（soft=超过后开始排队，hard=超过后直接 503 拒绝）
python mock_server.py --port 8123 --ttft-ms 200 --tpot-ms 15 --soft-capacity 8 --hard-capacity 12

# 终端 2：跑自测配置
python bench.py --config config.mock-demo.toml
```

自测已验证的行为：低并发 SLA/SLO=100%；超过 soft 容量后 TPOT 抬升落入 A3
（SLO 降而 SLA 不降）；超过 hard 容量后大量 503 落入 A1（SLA 断崖）；
报告能正确定位"3 个 9 对应的最高并发"。

## 8. 使用注意与局限

- **超时口径**：`timeout_s` 是单请求整体超时。分钟级请求下若设小了，慢请求会被记为
  A1（timeout）而非 A3，直接污染 SLA——上线前务必用 `--probe` 实测最长请求时长。
- **样本量**：3 个 9 的判定需要足够样本（建议每级 ≥ 1000 条才能区分 99.9% 与 99.5%），
  低压级样本天然少，报告中会对不足样本的级别给出置信度提示。
- **前缀缓存**：`unique_prefix=true` 会加少量 token 换取无缓存干扰的 TTFT；
  若生产有关键前缀缓存优化，可关闭以测量"热"路径。
- **单机压力上限**：单进程闭环压测建议到数百并发；更高压力可多机运行、
  各自写独立 db 后合并 `requests` 表分析。
- **Router 行为依赖**：A1 中的 rejected 依赖 Router 返回 429/503（或报文含
  overload/queue full 字样）；若部署的 Router 用其他状态码表达过载，
  请在 `bench/client.py` 的失败分类处补充。
- 本工具测量的是**客户端视角**的可用性与质量，不替代服务端监控
  （GPU 利用率、KV cache 占用、队列深度等仍需从集群侧采集）。
