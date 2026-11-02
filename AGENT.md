# sla-analyzer（PD 分离推理 QoS 压测工具）Agent 工作指引

本文件指导 Agent 在 `docs/issue/benchmark/` 这个**独立仓库**里工作。它只讲"怎么在这个项目里正确干活"，
不重复 `README.md`（怎么用）与 `docs/`（为什么这样设计）的内容；三者冲突时，以代码与 `docs/` 的决策记录为准。

## 0. 归属边界（先看这一条）

- 本目录有自己的 `.git`，remote 是 `https://github.com/Limixxx/sla-analyzer.git`，分支 `main`。
  它物理上位于 NanHuInfer 的 worktree（`.worktree/feat-dynamo-runtime-support/`）内，
  **但不是 NanHuInfer 仓库的一部分**：不要在这里执行主仓库的 `AGENT.md`/`CLAUDE.md` 逐字同步规则，
  也不要把这里的改动混进主仓库提交。git 命令必须在本目录内执行。
- 相邻的 `docs/issue/`（`TODO.md`、`PORTS.md`、IGD YAML 等）是 NanHuInfer 侧的调研笔记，不是本项目产物。
- 工具的运行环境是**集群 Pod 内**：零第三方依赖（Python ≥ 3.11 标准库），可直接拷贝运行。
  这是硬约束，任何新功能不得引入 `pip install`。

## 1. 不可违背的硬约束

1. **零依赖**：只用标准库（`asyncio`/`sqlite3`/`tomllib`/`zipfile`/`struct`）。`requirements.txt` 只做说明。
   ML 模型（`bench/mlp.py`）、`.npz` 容器（`bench/npz.py`）、HTTP/SSE 客户端（`bench/httpclient.py`）都是手写实现。
2. **统计诚实原则**（贯穿全仓，最高优先）：不可信的统计量比没有统计量更危险。
   样本不足、验证不达标、超出训练包络、基线被污染——一律**显式降级 + 喊出来**，绝不静默使用、绝不编造数字。
   典型实现：桶样本不足回退 absolute、`R² < min_r2` 不出预测结论、`extrapolated` 不参与结论、
   样本不足级只进"低置信度参考容量"。
3. **口径纪律**：SLA（接得住）与 SLO（接得好）永远分开陈述；**生产并发建议只认质量容量**。
   不得把 SLA 100% 的级别当容量结论（排队型过载陷阱）。
4. **禁止循环定标**：SLO 门限绝不使用本次运行（尤其高压段）观测到的分位数；
   `analysis.sla_target` 恒为固定配置值，不做动态化。
5. **时延口径**：一律 `time.monotonic()`；TTFT/E2E 为客户端视角（从建连前起算），
   `connect_ms` 单列可扣减；`sent_at` 取**发出时刻**，不取发现/取消时刻。
6. **隐私**：`requests` 表只存特征（长度、时延、分类、错误类型），**不存 prompt 与响应正文**。
7. **方法论决策 D1–D16**（`docs/00-overview.md`）：改动任何判定/分类/统计逻辑前必读对应章节；
   推翻某条决策要在 `docs/00-overview.md` 的决策表和修订记录里落地，不能只改代码。

## 2. 阅读顺序

1. `README.md` —— 指标定义、目录结构、快速开始、配置表、局限。
2. `docs/00-overview.md` —— 文档地图 + D1–D16 决策表 + 一页总览图。
3. 按需：`01 指标体系`（A1/A2/A3、SLA/SLO、TTFT/TPOT）、`02 容量模型`（两阶段过载、连续前缀）、
   `03 门限定标`（基线、长度分桶、退化率）、`04 测量有效性`（偏差 B1–B9）、
   `05 水位语义`（NORMAL/BUSY/OVERLOAD 投影）、`06 预测模型`（MLP、分组 CV、残差 ECDF、包络）。
4. 代码：`bench.py`（CLI 与四种运行模式）→ `bench/runner.py`（编排）→ `bench/metrics.py`（判定）→
   `bench/report.py`（结论渲染）。
5. 每篇文档末尾都有**实现映射表**——文档里的每个论断都应能在代码里指到具体函数；反之改代码也要回查映射表是否失效。

## 3. 架构与数据流

```
config.toml/.json → config.py(Config dataclass + 长度自动推导 + validate fail-fast)
      → runner.py(Phase 0 基线 + 递增并发 ramp：预热/测量/按桶或按量自适应延长/drain/aborted)
      → client.py + httpclient.py(单请求执行、SSE 流式采集、失败子类型 KIND_*)  → prompts.py(按 token 预算合成)
      → store.py(SQLite：meta/stages/stage_buckets/baseline/requests)
      → baseline.py(分桶 + 基线分位数 + 门限解析) / metrics.py(classify、StageStats、find_capacity、水位)
      → qos.py(+mlp.py+npz.py：采样→分组 CV 训练→ECDF 校准→包络守卫→边界细扫)
      → report.py(一~十节 + 附录 A/B) → results/report_<ts>.md
```

| 模块 | 职责 | 改动时的关键点 |
| --- | --- | --- |
| `bench.py` | CLI、覆盖参数、`--probe`/`--baseline-only`/`--report-only`/`--qos-predict` | 新参数要同步 README 第 3 章与 `--help`；`--duration` 与 `--max-duration` 的 5 倍联动 |
| `bench/config.py` | 加载、类型宽容策略（float 接受 int）、`validate()`、长度推导 | 任何校验都在启动期 fail-fast；`_field_default` 不能写成 `cls.name` |
| `bench/prompts.py` | 按 token 预算合成、唯一前缀防缓存 | `chars_per_token` 只是估算，真实值以 `usage` 为准 |
| `bench/httpclient.py` | 零依赖异步 HTTP/1.1 + SSE | 不引第三方；TLS 校验由 `verify_tls` 控制 |
| `bench/client.py` | 单请求执行、TTFT/TPOT/E2E/connect、失败分类 | 过载识别（429/503 或 overload/queue full 字样）在此，Router 行为变化要补这里 |
| `bench/stats.py` | 最近秩分位数、均值 | 不做插值（D9）；小样本 p99 = 最大值属已知局限 |
| `bench/baseline.py` | 分桶、Phase 0 统计、门限解析与回退 | 基线只统计不判定（D5）；桶粒度样本门限 ≠ 级粒度（D14） |
| `bench/metrics.py` | A1/A2/A3、`StageStats`、容量连续前缀、水位判定与边界 | 判定语义变更必须同步 `docs/01/02/05` + README 第 1、5 章 |
| `bench/runner.py` | 编排、延长判据、drain、aborted | 配置时长与实际时长分开记录（D11）；`sent_at` 发出时刻取样（D13） |
| `bench/store.py` | 建表、插入、按级/桶/基线读回 | 只存特征；`--report-only` 依赖这些列 |
| `bench/qos.py` `mlp.py` `npz.py` | 数据集、训练协议、校准、仿真、`.npz` 落盘 | A3 必须入训（D16）；按 stage 分组 CV；`.npz` 需与 `np.load()` 互通 |
| `bench/report.py` | 结论、建议、各级表格、预测章节 | 结论口径与 `metrics.find_capacity` 的 notes 全量渲染，不得吞掉提示 |
| `mock_server.py` | 本地模拟两阶段过载（soft 排队 / hard 拒绝） | 只用于自测，不进生产结论 |

## 4. 常见任务的改动清单

**新增一个配置项**：`config.py` dataclass 字段（带默认值）→ `load_config` 里 `_optional/_require` 读取
→ `validate()` 加校验与理由注释 → `config.example.toml` 注释 → README 第 6 章表格 → 若属判定语义再补 `docs/0x` 与实现映射。

**新增失败子类型**：`client.py` 的 `KIND_*` 常量 → `metrics.py` 的 A1 归因家族（`A1_TIMEOUT_KINDS` /
`A1_REJECT_KINDS`）→ `report.py` 第六节构成表与建议 → `docs/01`（A1 定义）+ `docs/05`（水位投影）。
注意：Router 若用别的状态码表达过载，过载会被误记为"慢"（A3）而非"拒绝"（A1），容量结论偏乐观。

**改判定/分类/水位/容量逻辑**：先读对应 `docs/` 章节 → 同时保证**实时压测路径与 `--report-only` 路径一致**
（`bench.py::_stats_from_store` 现场重算门限与水位，DB 里的 `watermark_state` 仅审计用）。
两条路径分叉是本工具最容易出的 bug。

**改 DB schema**：`store.py::SCHEMA` + 写入 SQL + 读回；旧库兼容沿用 `_stats_from_store` 里
`"列名" in keys` 的探测式回退写法，不要假设旧库有新列，也不要为兼容铺一套迁移框架。

**改报告**：章节编号是一~十 + 附录 A/B 的硬编码序列，增删章节要同时改编号、README 第 7 章的产出描述、
以及 `docs/` 里对该节的引用。

## 5. 验证方式（本项目无测试框架、无 CI）

没有 `tests/`、没有 pytest、没有 lint 配置。回归闸门就是下面这几条，改完必须至少跑前两条：

```bash
cd <本目录>
# 1) 编译 + CLI（秒级）
python -m py_compile bench.py mock_server.py bench/*.py && python bench.py --version

# 2) 端到端自测（约 1 分钟）：两阶段过载 + 基线口径 + 报告全链路
python mock_server.py --port 8123 --ttft-ms 200 --tpot-ms 15 --ttft-per-1k-tokens 60 \
    --soft-capacity 8 --hard-capacity 12 &
python bench.py --config config.mock-demo.toml --probe
python bench.py --config config.mock-demo.toml --steps 4,8,16,32 \
    --duration 4 --warmup 1 --max-duration 8 --target-mode baseline_relative \
    --db results/tmp_check.db --report results/tmp_check.md

# 3) 复盘幂等性（同库重建报告，结论应与 2) 一致）
python bench.py --config config.mock-demo.toml --report-only \
    --db results/tmp_check.db --report results/tmp_check2.md
```

预期行为（用于判断改动是否破坏了语义）：低压级 SLA/SLO=100%；超过 soft 容量后 TPOT 抬升落入 A3（SLO 降、SLA 不降）；
超过 hard 容量后大量 503 落入 A1（SLA 断崖）；结论以**质量容量**为准；短桶样本不足时门限回退 absolute 并打印
"还差多少样本/约需多久/建议并发"的量化提示；样本 < `qos_model.min_samples` 时明确"模型未生成 + 原因"而不是硬训。
文档级改动跑 `git diff --check` 并核对相对链接与实现映射表。

环境事实（本机已验证）：Windows + Python 3.14.4，工具要求 ≥ 3.11；`--probe` 对 mock 返回
TTFT≈230ms / TPOT≈21ms / E2E≈1.5s；上面的快速自测约 55 秒跑完。
**未执行的检查不得描述为通过**；真实集群压测需要用户提供 Router 地址，不可自行发起。

## 6. 代码与文档风格

- 注释、docstring、报告文案、日志一律**中文**；注释只写"为什么"（统计理由、偏差防控），不复述代码在做什么。
- 单位进变量名后缀：`_s` / `_ms`；配置项与报告表格保持同一单位口径。
- `from __future__ import annotations` + `dataclass` 承载配置与结果对象；跨模块用显式 `from .x import y`。
- 判定类函数返回 `(值, 理由列表)` 或带 `notes` 的数据结构，让**理由可以一路渲染到报告**，不要只返回布尔。
- 模块内不打印（`runner.py` 的阶段进度打印除外）；面向用户的输出集中在 `bench.py` 与 `report.py`。
- 改动范围克制：不顺手重构、不全仓格式化、不做与当前任务无关的"清理"。

## 7. Git 与产物

- `.gitignore` 忽略 `results/*.log|*.md|*.db` 和 `__pycache__/`，**但没忽略 `results/*.npz`**：
  训练产物 `qos_model.npz` 属可复现产物，不要提交（需要长期保留时先与用户确认或补忽略规则）。
- 根目录可能散落临时脚本（如 `_probe.py` 这类一次性验证脚本）与 `results/` 下的临时 db/report：
  收尾时清理自己创建的，**别人创建的先问再删**。
- Windows 下会打印 `LF will be replaced by CRLF` 警告，属正常噪音，不影响运行；提交前用 `git status`
  确认只包含预期文件。
- 本目录处于**活跃并发编辑**中：工作区同时有 `bench/client.py`、`bench/config.py`、`bench/metrics.py`、
  `bench/report.py`、`mock_server.py` 未提交改动，且改动范围在这次会话期间仍在扩大。
  动手前必须先 `git status --short` + `git diff` 重新核实现状，基于工作区内容做增量编辑，
  不要整体重写这些文件，也不要按本文件第 8 节的快照直接下结论。

## 8. 进行中的特性快照（2026-10-30，接手前重新核实）

主题：**无限队列部署下的过载识别**——Router 不返回 429/529 时，过载在客户端只投影为超时，
需要把 A1 拆成"超时族 / 拒绝族"才能分辨排队型与拒绝型过载。

- `bench/metrics.py`：新增 `A1_TIMEOUT_KINDS`（timeout、connect_timeout）、`A1_REJECT_KINDS`
  （rejected、http_4xx、http_5xx、stream_error）、`split_fail_kinds()` 与 `StageStats.a1_families()`；
  `classify_watermark` 在 A1 触发硬水位时附带家族归因（超时族主导 ⇒ 疑似无限队列/排队耗尽；
  拒绝族主导 ⇒ 拒绝型过载）。
- `bench/report.py`：结论、建议、预测章节口径说明处消费上述归因（"A1 拒绝型" 措辞改为 "A1 失败型"，
  并明确模型不预测失败概率）。
- `bench/config.py` / `bench/client.py`：`[watermark]` 与 `KIND_*` 的注释同步新语义。
- `mock_server.py`：新增 `--infinite-queue` / `--queue-step-s`，模拟不拒绝、排队时间随超出量无上限增长的部署。

该自测变体可用于验证超时族归因（其余参数同上）：

```bash
python mock_server.py --port 8123 --infinite-queue --queue-step-s 2.0
```

**尚未完成的配套**：`docs/05-watermark.md` 还没有代码注释所引用的"无限队列"一节；`README.md` 第 1/5/8 章
（A1 定义、水位表、自测说明）与 `docs/01` 的 A1 定义未提及家族拆分与 `--infinite-queue`；
`docs/00-overview.md` 的 D1–D16 决策表与修订记录未更新。按第 4 章清单补齐前先与用户确认。

已知文档引用缺口：`docs/05-watermark.md` 开头把术语来源写作 `results/watermark.md`，该文件不在仓库内
（且被 `.gitignore` 规则覆盖）——引用外部资料时应改指向仓库内可见文档或明确标注其在工作区外。

## 9. 与用户协作

- 用中文回复。用户会在"讲业务口径、不贴代码"与"结合代码、给 `文件:行号`"之间切换，按当前这条消息的要求来。
- 结论要有出处：机制论断指到具体代码/文档位置；涉及实测数字要给可复现命令或产物。
- 用户提出"这样理解是否正确"时，逐条给判定，不要默认采纳其前提。
- 容量/水位结论的表述纪律：质量容量与 3 个 9 口径分开陈述；样本不足、低置信度、外推、未触达等限定条件
  必须随行携带，不得简化成单个数字。

## 10. 明确不做

- 不引入第三方依赖，不改用 requests/httpx/numpy/torch，不做在线推理或进入请求路径。
- 不做服务端资源级水位（KV Cache / 批 slot / GPU）计算——那是服务端 Prometheus 与调度器的职责。
- 不做动态 SLA 目标、不用本次高压分位数定标门限、不把 aborted 计入任何比率。
- 不擅自对真实服务发起压测；需要集群地址、模型名、API Key 时向用户索取。
- 不把 `baseline_relative` 口径的"达标"当跨引擎可比的结论——横向比较只能用质量退化率与原始分位数。
