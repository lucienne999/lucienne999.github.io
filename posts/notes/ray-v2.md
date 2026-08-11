---
title: "从 2000 万条数据处理看 Ray：任务切分、背压与故障边界"
category: "ai-infra"
tags: ["ray", "分布式数据处理", "背压"]
date: 2026-08-08
---

数据处理通常从一个单机 `for loop` 开始：读取一条记录，完成计算，再把结果写回。数据量继续增长后，我们会先尝试线程池或进程池；只有当单机的 CPU、内存或 I/O 吞吐成为瓶颈，才需要把同一套计算拆到多台机器上。

Ray 解决的是这个阶段的问题。它让 Driver 可以把计算提交到远程 worker，也允许状态和中间结果脱离 Driver 的本地进程存在。但“能够分布式执行”和“能够稳定处理大规模数据”并不是一回事：前者由框架提供，后者取决于任务如何切分、数据在哪里展开、结果如何回收，以及失败后从哪里恢复。

## 先理解 Ray 的最小运行模型

阅读后面的设计前，只需要先区分五个概念：

- **Task**：一次无状态远程函数调用，适合执行可以独立重算的计算。
- **Actor**：常驻在某个 worker 进程中的有状态对象，适合持有模型、连接池、缓存或写入状态。
- **ObjectRef**：Task 或 Actor 方法的异步返回引用。提交方先拿到引用，结果完成后再通过 `ray.get` 读取。
- **`ray.wait`**：只等待部分 ObjectRef 完成，使 Driver 可以边提交、边回收，而不是等待所有任务结束。
- **Ray Data**：构建在 Task、Actor 和 Object Store 之上的 Dataset 执行层，负责 block 切分、流水调度和规则数据变换。

一个最小执行过程如下：

```text
Driver 提交 Task
  -> Ray 把 Task 调度到 worker 进程
  -> worker 读取或接收输入并完成计算
  -> 结果进入 Ray 管理的对象存储
  -> Driver 或下游 Task 通过 ObjectRef 消费结果
```

这里已经包含了大规模运行时最容易堆积的三类“存量”：Driver 持有的 ObjectRef、Object Store 中尚未消费的结果，以及 worker 正在处理的 batch。提交速率持续高于完成和消费速率时，这些存量就会不断增长。

因此，假设现在有 2000 万条记录需要完成读取、计算和写回，真正困难的不是把 `for loop` 改成 `.remote()`，而是让系统在数据规模扩大后仍然满足四个约束：

1. Driver 不持有全量输入和结果。
2. 调度器中同时存在的 Task 数量有上界。
3. Object Store 和 worker working set 不随总数据量增长。
4. 节点或单条数据失败后，只重算有限范围，并且不会重复产生不可逆副作用。

这四个约束分别对应分布式数据处理中的控制面压力、内存压力和恢复成本。Ray 提供了 Task、Actor、Object Store 与 Ray Data，但这些抽象不会自动保证系统有界。任务粒度、inflight window、数据所有权和 sink 语义仍然需要业务层设计。

本文只保留理解生产问题所需的最小概念，不逐一介绍 Ray API，而是从一个可量化的模型出发，回答三个问题：

- 为什么单条一个 Task、一次性提交全部任务会先撞到调度和内存墙？
- Task、Actor、`ray.wait` 和 Ray Data 分别应该放在数据路径的什么位置？
- 系统“还活着”和业务“仍在推进”为什么是两件事？

## 1. 先给数据处理建一个有量纲的模型

只说“2000 万条数据”没有意义。决定系统压力的是每条数据的大小、每个 Task 的工作集，以及中间结果在系统里停留多久。

定义：

- `N`：总记录数。
- `B`：每个 Task 处理的记录数，即 `batch_size`。
- `W`：最多同时在飞的 Task 数，即 `max_inflight`。
- `S_in`：单条记录在 worker 中展开后的平均输入大小。
- `S_out`：单条记录产生的平均输出大小。
- `R`：每个 Task 返回到 Object Store 的摘要大小。
- `T_batch`：处理一个 batch 的平均时间。

忽略框架内部常数后，可以得到几个直接影响设计的量：

```text
Task 数量                 = ceil(N / B)
在飞记录上界              ≈ W · B
worker 侧活跃工作集上界    ≈ W · B · (S_in + S_out)
返回值的 Object Store 上界 ≈ W · R
单次失败的重算上界         ≈ B 条记录
```

这里存在一个不能消失的 trade-off：

- 增大 `B`，Task 数减少，调度和序列化的固定成本被摊薄，但单 Task working set、长尾时间和失败重算范围增大。
- 增大 `W`，并行度和设备利用率可能上升，但 worker 内存、Object Store、外部存储并发和失败风暴都会变大。

因此，`batch_size` 和 `max_inflight` 不能只根据 CPU 核数填写。它们共同定义了系统中任意时刻允许存在多少数据。

以 `N=20_000_000`、`B=50_000` 为例，任务数是 400，而不是 2000 万。若每个 Task 只返回一个 2 KiB 的统计摘要，`W=64` 时返回值占用约 128 KiB；如果每个 Task 返回 50_000 条、每条 4 KiB 的完整结果，同样的窗口会产生约 12.5 GiB 的活跃结果。两种实现的 Task 数相同，内存行为却完全不同。

## 2. Naive 方案为什么会失效

最直接的写法是为每条记录提交一个 Task，然后统一 `ray.get`：

```python
refs = [process_one.remote(record) for record in records]
results = ray.get(refs)
```

它同时破坏了三个有界条件：

```text
全量输入在 Driver
  -> N 次参数序列化与 Task 提交
  -> Driver 持有 N 个 ObjectRef
  -> 完成结果在 Object Store 和 Driver 聚集
  -> 峰值内存、调度延迟和失败恢复成本随 N 增长
```

这里首先耗尽的不一定是 CPU。对于单条只运行几毫秒的轻任务，固定的提交、调度、进程间传输和结果管理成本可能比业务计算更大。继续增加 worker 只会让瓶颈更快转移到 Driver、raylet、Object Store 或下游 sink。

另一个常见误区是把大对象直接放进 Dataset 或 Task 参数：

```python
# 不推荐：图片、点云或视频已经在 Driver 中展开
ds = ray.data.from_items([{"image": image} for image in images])
```

当 payload 已经进入 Ray 的对象传输路径，系统需要承担序列化、共享内存占用、跨节点传输和 spill。更稳妥的方式是只传递数据定位信息：

```python
ds = ray.data.from_items([
    {"sample_id": "001", "uri": "s3://bucket/001.bin"},
    {"sample_id": "002", "uri": "s3://bucket/002.bin"},
])


def process_batch(batch):
    # 在被调度到的 worker 内按需读取和释放真实 payload。
    return load_and_process(batch["uri"])
```

这不是“URI 一定比对象快”，而是改变了数据所有权：Driver 和 Object Store 管理轻量 metadata，真实 payload 的生命周期限制在 worker 的单个 batch 内。

## 3. 控制路径与数据路径必须分开

前面的最小模型解释了 Task、Actor 和 ObjectRef 分别是什么。落实到生产系统时，还需要进一步明确每一层拥有什么数据、负责什么状态。一个有界的数据处理系统可以拆成四层：

```text
Driver / Control Path
  ├── 生成轻量 batch descriptor
  ├── 维护不超过 W 个 ObjectRef
  ├── 聚合小型 BatchSummary
  └── 处理 timeout、重试和最终状态

Task / Compute Path
  ├── 在 worker 内读取真实数据
  ├── 完成无状态计算
  └── 释放 batch working set

Actor / Stateful Sink
  ├── 持有连接池、模型或文件句柄
  ├── 执行有状态写入
  └── 暴露可观测进度

External Storage
  └── 通过唯一键、幂等 upsert 或事务提供最终一致性
```

Ray 的 worker 是独立 OS 进程。raylet 负责本节点资源管理、worker 生命周期和本地调度协作；GCS 维护集群成员、Actor 和资源等控制面元数据。真正的 payload 读取和业务计算应该发生在 worker，而不是集中经过 head 节点。

在这个模型里：

- Task 扩展无状态计算。
- Actor 持有无法在每个 Task 中重复创建的状态。
- `ray.wait` 给 Driver 的提交速率建立上界。
- Ray Data 管理规则 Dataset transformation 的 block 级执行。

这些组件是职责划分，不是必须全部使用的固定模板。

## 4. Task 粒度：同时约束调度开销与失败半径

批处理的最小实现只需要生成区间，不需要在 Driver 构造真实记录：

```python
def iter_batches(total: int, batch_size: int):
    for batch_id, start in enumerate(range(0, total, batch_size)):
        yield {
            "batch_id": batch_id,
            "start": start,
            "end": min(start + batch_size, total),
        }
```

选择 `B` 时，至少要同时检查：

1. `T_batch` 是否足以摊薄 Task 固定开销。
2. 单 Task 峰值内存是否明显低于 worker 可用内存。
3. 一个 straggler 是否会占住过多输入。
4. 重试一个 Task 时，重复读取和计算的成本是否可接受。
5. sink 能否承受 `W · B / T_batch` 对应的写入速率。

不存在普适的 `batch_size=50_000`。图片解码、文本 tokenize、点云解析和数据库回填的 `S_in`、CPU 时间与 I/O pattern 都不同。更可靠的调参顺序是：

```text
先用单 worker 测出 B -> (T_batch, peak RSS)
  -> 找到能够摊薄固定开销且不造成长尾的 B
  -> 再增加 W，直到 compute、存储或网络中的一个资源饱和
```

先调 `B` 再调 `W`，可以避免把“Task 太碎”和“并行度不够”混成同一个问题。

## 5. 背压：Driver 只维护一个滑动窗口

`ray.get(all_refs)` 的问题不是阻塞本身，而是必须先构造并持有全部 refs。大规模任务应该维持固定窗口：完成一个，再提交一个。

下面的实现只保留未完成任务和聚合统计，不保存所有 batch 的完整返回值：

```python
from dataclasses import dataclass
import time
import ray


@dataclass
class Aggregate:
    succeeded: int = 0
    failed: int = 0
    records: int = 0

    def add(self, summary: dict) -> None:
        self.succeeded += summary["status"] == "ok"
        self.failed += summary["status"] != "ok"
        self.records += summary["records"]


def run_bounded(batch_iter, max_inflight: int, idle_timeout_s: float):
    pending = {}
    aggregate = Aggregate()
    last_progress = time.monotonic()

    def drain_one() -> None:
        nonlocal last_progress
        ready, _ = ray.wait(
            list(pending),
            num_returns=1,
            timeout=min(5.0, idle_timeout_s),
        )
        if not ready:
            if time.monotonic() - last_progress >= idle_timeout_s:
                stalled = [pending[ref] for ref in pending]
                for ref in pending:
                    ray.cancel(ref, force=True)
                raise TimeoutError(f"stalled batches: {stalled}")
            return

        ref = ready[0]
        batch = pending.pop(ref)
        try:
            aggregate.add(ray.get(ref))
        except Exception as exc:
            raise RuntimeError(f"batch failed: {batch}") from exc
        last_progress = time.monotonic()

    for batch in batch_iter:
        while len(pending) >= max_inflight:
            drain_one()
        pending[process_batch.remote(batch)] = batch

    while pending:
        drain_one()

    return aggregate
```

这里的内存复杂度由总任务数 `ceil(N/B)` 降为 `O(W)`。异常信息保留 batch descriptor，失败后可以重放精确范围。

但 `ray.cancel(..., force=True)` 不是事务回滚。Task 可能已经对外部系统产生部分写入，取消也可能通过终止 worker 带来额外重建成本。因此，可靠恢复依赖的是幂等 sink 和 checkpoint，而不是“超时后 cancel”本身。

## 6. Actor 解决状态所有权，不自动解决一致性

数据库连接、模型实例和文件句柄不适合在每个 Task 中反复创建。Actor 可以让状态固定在一个 worker 进程内，其他进程只持有句柄。

```python
import ray


@ray.remote(max_concurrency=8)
class WriterActor:
    def __init__(self, config):
        self.client = DatabaseClient(config)

    def upsert(self, rows):
        # 一致性依赖数据库唯一键与幂等 upsert。
        return self.client.upsert_many(rows)
```

默认 Actor 串行执行方法。显式设置 `max_concurrency` 后，同一个 Actor 内部可能有多个并发调用，线程安全需要由 client 或业务代码保证。还需要分清三个不同的并发量：

```text
Actor 数量
  × 单 Actor max_concurrency
  × 单次方法可能占用的数据库连接数
  = 下游可能观察到的连接与请求规模
```

不能只调大 `max_concurrency` 而不检查数据库连接上限、QPS quota 和写入延迟。sink 饱和后，更多计算并行只会让中间结果更快堆积。

### 6.1 固定路由不等于可扩缩的 consistent hashing

使用下面的路由，只能保证 Actor 集合不变时同一个 key 落到同一 Actor：

```python
index = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % len(actors)
actor = actors[index]
```

当 Actor 数从 4 变成 5，大量 key 会重新映射。如果业务要求在线扩缩容时保持大部分 key 不迁移，应使用 rendezvous hashing 或 consistent hashing；如果只是一次离线 job 内固定 4 个 writer，取模已经足够，不需要引入更复杂的路由层。

无论使用哪种 hash，最终写一致性都不应该依赖 Actor 路由。Actor 可能重启、Task 可能重试，数据库仍需要唯一键、幂等 upsert 或事务。

### 6.2 Detached Actor 不是默认选项

普通 Actor 的生命周期受 owner 约束。只有当多个独立 job 确实需要共享同一组 writer 时，才需要：

```python
actor_cls.options(
    name="db_writer_0",
    namespace="document_writer",
    lifetime="detached",
    get_if_exists=True,
).remote(config)
```

Detached Actor 的代价是生命周期脱离当前 job：

- 新 job 可能复用旧代码版本或旧配置创建的 Actor。
- job 退出不会自动释放连接和资源。
- Actor 数量调整需要独立的部署与清理协议。

因此，单 job 的离线处理优先使用普通 Actor；跨 job 共享时再引入 detached lifetime，并把版本写入 name 或显式校验 Actor 配置。

## 7. Ray Data 与裸 Ray 解决的不是同一层问题

Ray Data 适合输入天然表达为 Dataset，并且处理过程主要是规则 transformation：

```text
read -> map_batches -> filter -> repartition/shuffle -> write
```

它在 block 层管理流水执行、资源声明和部分容错，避免业务代码直接维护大量 `ObjectRef`。典型场景包括 Parquet 转换、批量推理、embedding 生成、样本过滤和离线特征抽取。

裸 Ray Task + Actor 更适合：

- 单个业务对象已经是粗粒度任务。
- 需要提前结束、取消 straggler 或自定义成功率。
- 需要稳定路由到有状态资源。
- 每个任务有不同资源需求或非规则依赖关系。
- 需要精确记录 batch 级重放信息。

工程里经常混用两者：Ray Data 调度轻量 row，Actor 持有模型或连接池，worker 根据 URI 读取真实 payload。关键判断不是“哪个 API 更高级”，而是问题更像 Dataset transformation，还是一个带状态和失败语义的分布式调度系统。

## 8. 故障模型：节点活着不等于业务有进展

“任务结束不了”至少有四类不同原因：

| 故障 | Ray 可观察到什么 | 业务层还需要什么 |
|---|---|---|
| 进程或节点退出 | Task/Actor error、节点状态变化 | 幂等重试与 batch checkpoint |
| 外部 I/O 长时间阻塞 | worker 仍存活，Task 不返回 | 请求 timeout、业务进度时间戳 |
| native 扩展 hang | 进程可能存活但无进展 | task deadline、隔离与强制终止 |
| 单条坏数据 | block 或 batch 反复失败 | item 级失败记录与隔离队列 |

raylet/GCS 维护的是集群和进程层面的活性信息。它能判断节点或 worker 是否消失，但不能知道某个视频解码循环是否仍然产生有效进度。

长 batch 应该上报业务进度：

```python
@ray.remote
class ProgressTracker:
    def __init__(self):
        self.progress = {}

    def report(self, batch_id: int, processed: int, timestamp: float):
        self.progress[batch_id] = {
            "processed": processed,
            "timestamp": timestamp,
        }

    def snapshot(self):
        return dict(self.progress)
```

这里需要观察两种不同信号：

```text
节点 heartbeat：控制面还能否看到 node / raylet
业务 progress：processed 是否持续增加
```

前者异常时检查节点资源、raylet、GCS、网络和进程退出日志；后者停滞时检查具体 batch、外部依赖、锁、native stack 和坏数据。把两类信号混成一个“heartbeat 丢失”，会导致根因分析停留在猜测。

坏数据与环境错误也应该分开处理：

- 确定性解析错误：记录 item ID，进入隔离队列，不要让整个 batch 无限重试。
- 节点退出、连接重置等瞬时错误：抛出异常，由任务级重试处理。
- 已经产生外部副作用的错误：先通过幂等键读取最终状态，再决定是否重放。

## 9. 参数不是结论，实验数据才是

评价 Ray 方案至少需要一条串行或本机进程池基线，并记录以下指标：

| 方案 | 吞吐 records/s | Driver RSS | Worker peak RSS | Object Store peak | P50/P99 batch latency | 失败重算量 |
|---|---:|---:|---:|---:|---:|---:|
| 单进程 | 待测 | 待测 | - | - | 待测 | - |
| ProcessPool | 待测 | 待测 | 待测 | - | 待测 | 待测 |
| Ray Task，无窗口 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Ray Task，`W=...` | 待测 | 待测 | 待测 | 待测 | 待测 | `B` |
| Ray Data | 待测 | 待测 | 待测 | 待测 | 待测 | block size |

本文刻意不填未经测量的“加速倍数”。在具体业务里，至少应固定并公开：

- Ray 版本与 Python 版本。
- 节点数、每节点 CPU/内存、节点间网络。
- 输入存储及其读取带宽/QPS。
- `N`、`B`、`W`、单条展开大小与输出大小。
- 计算阶段和 sink 阶段各自耗时。
- spill 配置、重试次数和失败注入方法。

测试顺序建议是：

1. 单进程小样本，确定单核计算与 I/O 基线。
2. 固定 `W=1` 扫描 `B`，观察 Task 固定开销和 working set。
3. 固定合理的 `B` 增加 `W`，找到第一个饱和资源。
4. 注入 worker 退出、I/O timeout 和坏数据，测量恢复时间与重复写入。

如果增加 `W` 后 CPU 利用率不再提高、sink latency 持续上升，瓶颈已经在外部存储；如果 batch 计算很轻且 Driver CPU 很高，瓶颈更可能是调度；如果 Object Store spill 增长，应先检查是否返回了过大的结果，而不是继续增加内存。

## 10. 为什么不是其他方案

Ray 不是默认答案。

### 单机 `ProcessPoolExecutor`

当数据能在单机磁盘和内存范围内完成、没有跨节点资源调度需求时，进程池的控制面更简单，部署和调试成本也更低。只有单机吞吐或容量成为明确瓶颈时，分布式执行才开始产生收益。

### Spark

对于 TB/PB 级 Parquet ETL、大规模 shuffle、SQL 聚合和成熟 checkpoint 需求，Spark 的执行计划、shuffle 和存储生态通常更完整。Ray 更适合 Python-native 计算、异构资源调度、模型推理以及 Task/Actor 混合工作负载。

### 消息队列 + 独立 sink service

当数据库写入需要长期运行、跨多个 job 共享、严格限流和独立扩缩容时，把 writer 做成外部服务往往比 detached Actor 更清晰。Actor Pool 更适合生命周期受控的 job 内状态，不应被无条件扩展成基础设施服务。

### Ray Data

当流程是规则的数据集变换时，不要为了精细控制而手写完整调度器；当流程需要 early finish、自定义失败语义和有状态路由时，也不要为了使用 Dataset API 而隐藏这些控制需求。

## 11. 最终设计

对于 2000 万条记录的离线处理，一个可作为起点的结构是：

```text
轻量 batch descriptor
  -> Driver 维护 W 个 inflight Task
  -> worker 内读取真实 payload
  -> Task 处理 B 条记录并尽快释放 working set
  -> Actor 或外部 sink 执行幂等写入
  -> Driver 只聚合 BatchSummary 和 checkpoint
```

这个结构成立的前提是：

- `B` 已通过单 worker 的时间与内存测量确定。
- `W` 没有超过 compute、网络和 sink 中最窄的一环。
- Task 返回摘要而不是完整 payload。
- sink 支持幂等写入，重试不会制造重复结果。
- node health、task deadline 和业务 progress 分别可观测。

Ray 的核心价值不是把 Python 函数变成远程调用，而是让计算、状态、数据流和控制流可以采用不同的扩展方式。它的代价也来自同一个地方：一旦任务粒度、数据所有权和失败语义没有设计清楚，分布式执行只会把本地 `for loop` 的问题放大到 Driver、Object Store、网络和外部存储。

因此，判断一个 Ray 数据处理方案是否成立，不要先问“开了多少 worker”，而要问：

1. 系统中的活跃数据量是否由 `B · W` 明确限制？
2. 第一个饱和资源是什么，是否有指标证明？
3. 任意一个 Task 重试时，会重算多少数据、重复哪些副作用？

只有这三个问题都有量纲明确的答案，Ray 才是在解决规模问题，而不是把规模问题交给调度器。
