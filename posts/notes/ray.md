---
title: "数据处理中的 Ray 应用"
category: "ai-infra"
tags: ["ray", "并发"]
date: 2026-06-28
---

这篇笔记想回答一个很具体的问题：当数据处理规模从几千条涨到几百万、几千万，甚至 TB/PB 级别时，为什么不能只是“把 for loop 改成并发”？

真正麻烦的地方往往不是并行本身，而是这些工程问题：

- 数据应该怎么切分，才能避免调度器被小任务打爆？
- 任务应该怎么提交，才能避免 driver 和 Object Store 堆满？
- 结果应该怎么回收，才能既不无限等待，也不提前丢数据？
- 数据库连接、模型实例、进度计数器这类有状态资源应该放在哪里？
- 节点心跳丢失、worker 卡死、坏数据拖住整个 job 时，业务层应该如何兜底？

评价一个方案时，至少要同时看四个指标：端到端吞吐、单任务工作集内存、Object Store 峰值占用，以及失败后需要重算的数据量。只比较 worker 数或 wall time，通常看不出系统把瓶颈转移到了存储、调度还是 sink。

Ray 的价值在于，它不是只给 Python 加了一个并发库，而是把分布式执行里最常见的几个概念抽象成统一模型：

- **Task**：无状态远程函数，适合批量并行计算。
- **Actor**：有状态远程对象，适合封装连接池、模型、缓存、计数器。
- **ObjectRef**：远程结果引用，避免调用方立刻阻塞。
- **ray.wait**：等待部分任务完成，用来做背压和流式回收。
- **Ray Data**：建立在 Task / Actor 之上的数据集执行层，适合声明式批处理流水线。

一句话概括：**Task 负责扩展计算，Actor 负责持有状态，ray.wait 负责控制节奏，Ray Data 负责把规则的数据处理流程标准化。**

## 1. 先建立控制路径与数据路径

Ray 的 worker 是**独立 OS 进程**，每个节点还会运行 raylet 和本地 Object Store。GCS 维护节点、Actor、资源等集群元数据；raylet 管理本节点资源、worker 和任务调度。Task 不需要把每次业务计算都集中经过 head 执行，真正的数据读取和计算发生在各个 worker 内。

所以针对上面的现象：

- 普通 Task 借用 worker 进程执行一次无状态函数。
- Actor 长期占用一个 worker 进程，并把状态保存在这个进程内部。
- 调用远程函数或 Actor 方法时，调用方拿到的是 `ObjectRef`，不是立即拿到真实返回值。
- 想拿真实结果需要 `ray.get(ref)`；想等部分结果先完成可以用 `ray.wait(refs)`。

从这个模型出发，后面的状态隔离、Actor 串行语义、CPU 任务绕过 GIL、Object Store 压力、背压机制都会更容易理解。

> 这里的 **raylet** 可以理解成 Ray 在每台机器上的“本地节点管家”。它不是实际跑 Python 业务代码的 worker，而是负责管理这个节点上的资源和执行环境：接收调度过来的 Task / Actor，决定把任务交给哪个本地 worker，按需启动或复用 worker 进程，维护本节点的 CPU / GPU 等资源状态，和本地 Object Store 协作处理对象引用，同时定期向 head / GCS 上报 heartbeat。后面说的“节点 heartbeat 丢失”，本质上就是控制面长时间没有收到某个节点 raylet 的活性与资源状态汇报。


## 2. Task：最小可用的并行计算单元

一个最小 Ray 例子长这样：

```python
import time
import ray


ray.init()


@ray.remote
def process_record(record_id: int) -> dict:
    time.sleep(1)
    return {
        "record_id": record_id,
        "score": record_id * 10,
    }


if __name__ == "__main__":
    refs = [process_record.remote(i) for i in range(8)]
    results = ray.get(refs)
    print(results)
```

这里有三个关键点：

1. `@ray.remote` 会把普通函数转换成远程函数对象，调用时必须使用 `.remote(...)`。
2. `process_record.remote(i)` 不会立刻返回真实结果，而是返回一个 `ray.ObjectRef`。
3. `ray.get(refs)` 会等待所有任务完成；如果只想先拿已经完成的一部分，应使用 `ray.wait(refs)`。

如果不用 Ray，这 8 个任务串行执行大约需要 8 秒；使用 Ray 后，多个 worker 可以并行处理，整体耗时通常接近单个任务的耗时。

Task 适合做无状态计算，例如解析一批日志、处理一组图片、抽取一段数据的特征。它不适合直接保存共享计数器、数据库连接池、模型实例这类长期状态。

## 3. Actor：把共享状态封装进远程对象

如果任务之间需要共享状态，例如统计已经处理了多少条数据，可以使用 Ray Actor。Actor 是一个有状态的远程对象，适合封装计数器、模型实例、数据库连接、缓存等不能简单复制的资源。

```python
import ray


ray.init()


@ray.remote
class ProgressCounter:
    def __init__(self):
        self.count = 0

    def incr(self, n: int = 1) -> int:
        self.count += n
        return self.count

    def get(self) -> int:
        return self.count


@ray.remote
def process_batch(batch_id: int, counter) -> str:
    current = ray.get(counter.incr.remote())
    return f"batch {batch_id} done, processed={current}"


if __name__ == "__main__":
    counter = ProgressCounter.remote()
    refs = [process_batch.remote(i, counter) for i in range(5)]
    print(ray.get(refs))
    print("processed:", ray.get(counter.get.remote()))
```

Actor 避免并发冲突的核心不是“共享内存加锁”，而是**状态封装 + 消息传递**。

`self.count` 只存在于运行该 Actor 的 worker 进程里。其他 worker 手里只有 Actor 句柄，不能直接读写这个变量。多个任务调用 `counter.incr.remote()` 时，本质上是在给 Actor 发送远程方法调用请求。Ray 默认让同一个 Actor 一次只执行一个方法，因此多个请求不会同时修改 `self.count`。

需要注意，Actor 默认按消息到达队列的顺序执行，但多个发送方几乎同时提交时，到达顺序由调度和网络决定，不一定等于代码里肉眼看到的提交顺序。

## 4. Actor 并发：什么时候需要锁

默认 Actor 是串行执行的，所以内部状态天然安全。但如果显式开启 `max_concurrency`、使用 async actor 或 concurrency groups，同一个 Actor 内部就可能并发执行多个方法。

这时如果多个方法会读写同一份状态，就需要自己保护临界区：

- 普通并发 Actor 使用 `threading.Lock`。
- Async Actor 使用 `asyncio.Lock`。
- I/O 型 Actor 可以通过 `async def` 在等待网络、数据库、对象存储时让出执行权。
- CPU 密集型计算不要指望 async actor 提速，应该拆成 Task 或多个 Actor。

一个典型的并发 Actor 写法如下：

```python
import threading
import ray


@ray.remote(max_concurrency=8)
class LockedCounter:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()

    def incr(self, n: int = 1) -> int:
        with self.lock:
            self.count += n
            return self.count

    def get(self) -> int:
        with self.lock:
            return self.count
```

如果去掉锁，多个并发调用可能同时读到旧的 `count`，最后写回时覆盖彼此结果，最终计数小于预期。

## 5. 生产模式：Actor Pool + 稳定路由

单个 Actor 适合计数器、配置中心这类低吞吐共享状态。生产里更常见的是 **Actor Pool**：创建一组 Actor，通过 hash 把请求分片到不同 Actor，既扩展吞吐量，也让同一个 key 稳定落到同一个 writer 上。

一个数据库写入池可以简化成三层：

```text
DatabaseDocumentClient  # 真正访问数据库
_DatabaseWriterActor    # 把 client 包进 Ray actor
DocumentWritePool       # 创建 actor pool，负责路由和提交
```

核心代码大致如下：

```python
import hashlib
import ray


class _DatabaseWriterActor:
    def __init__(self, config):
        self.client = DatabaseDocumentClient(config)
        self.client.ensure_indexes()

    def ping(self) -> bool:
        self.client.ping()
        return True

    def upsert_document(self, document):
        return self.client.upsert_document(document)


class DocumentWritePool:
    def __init__(self, config, *, options):
        self.config = config
        self.options = options
        self._ray = ray
        self._actors = self._get_or_create_actors()

    def submit(self, *, document, timeout_s: float | None = None) -> None:
        actor = self._pick_actor(document)
        ref = actor.upsert_document.remote(document)
        self._ray.get(ref, timeout=timeout_s)

    def close(self, *, wait: bool = True) -> None:
        # detached actor 被整个 Ray job 共享，不能由单个 worker 关闭。
        return None

    def _get_or_create_actors(self):
        actor_cls = self._ray.remote(_DatabaseWriterActor)
        actors = []

        for idx in range(self.options.max_workers):
            actor = actor_cls.options(
                num_cpus=0,
                max_concurrency=self.config.max_pool_size,
                name=f"db_writer_{idx}",
                namespace="document_writer",
                lifetime="detached",
                get_if_exists=True,
            ).remote(self.config)
            actors.append(actor)

        self._ray.get([actor.ping.remote() for actor in actors])
        return actors

    def _pick_actor(self, document):
        key = f"{document.item_id}|{document.partition_id}|{document.version}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self._actors[int(digest[:8], 16) % len(self._actors)]
```

这里有几个工程判断：

- `actor_cls.options(...).remote(...)` 是实例级配置，适合在创建时动态传 `name`、`namespace`、`lifetime`。
- `num_cpus=0` 适合 I/O 密集型 Actor，因为主要时间花在等待网络或数据库上。
- `max_concurrency` 控制的是单个 Actor 内最多同时执行多少个方法调用，不是数据库连接数。
- 如果单 Actor 的数据库连接池大小也是 `max_pool_size`，整体连接上限大致是 `options.max_workers * config.max_pool_size`。
- `_pick_actor` 用稳定 hash 分片，同一个业务 key 会落到同一个 writer；真正写一致性仍依赖数据库唯一索引和原子 upsert。
- `ping()` 只是一个普通远程方法，用来提前暴露 Actor 初始化失败、数据库连接失败等问题。

## 6. Detached Actor 生命周期

默认 Actor 绑定创建它的 worker。创建者退出后，Actor 会被清理。

![默认非 detached Actor 生命周期](/images/notes/ray/ray-actor-default.png)

数据库 writer 这类共享基础设施通常不能跟随单个 worker 退出，否则先结束的 worker 可能把其他 worker 还在使用的 writer 带走。

![Detached Actor 共享机制](/images/notes/ray/ray-actor-detached.png)

生产里常见组合是：

- `name + namespace` 提供全局查找 key。
- `lifetime="detached"` 让 Actor 不跟创建者一起退出。
- `get_if_exists=True` 让后续 worker 复用已有 Actor。
- 业务层 `close()` 对 detached Actor 设为 no-op，避免单个 worker 误杀共享服务。

这也是我在使用 detached Actor 时最重要的经验：**保活不是靠 close，关键是不要让业务 worker 有权限顺手关掉共享 Actor。**

## 7. ray.get 与 ray.wait：背压从这里开始

小规模任务里，直接 `ray.get(refs)` 等全部完成没有问题。但当任务量达到几十万、几百万，甚至 2000 万条时，一次性提交所有 Task 再 `ray.get()` 会带来三个问题：

- driver 进程持有大量 `ObjectRef`。
- Ray 调度队列被海量小任务淹没。
- Object Store 堆积大量中间结果，触发 spill 或 OOM。

所以大规模任务需要用 `ray.wait` 做**背压（backpressure）**。

```python
# ray.get：等待所有任务完成
refs = [f.remote(i) for i in range(1000)]
results = ray.get(refs)
```

```python
# ray.wait：等待最先完成的一部分任务
refs = [f.remote(i) for i in range(1000)]
ready, remaining = ray.wait(refs, num_returns=1)
result = ray.get(ready[0])
```

典型写法是维护一个滑动窗口，最多保持 `max_inflight` 个 Task 在飞：

```python
pending = []
results = []

for batch_id, start, end in iter_batches(total_records, batch_size):
    ref = process_batch.remote(batch_id, start, end, writer)
    pending.append(ref)

    if len(pending) >= max_inflight:
        ready, pending = ray.wait(pending, num_returns=1)
        results.extend(ray.get(ready))

while pending:
    ready, pending = ray.wait(pending, num_returns=1)
    results.extend(ray.get(ready))
```

这个模式的核心是“完成一个、补一个”。任何时刻最多只有 `max_inflight` 个任务在跑，内存和调度压力都可控。

## 8. Ray Data：数据集执行层，而不是另一套 Ray

理解 Task、Actor 和 `ray.wait` 之后，再看 Ray Data 会清楚很多：Ray Data 不是另一套独立计算系统，而是 Ray 在 Task / Actor 之上提供的**分布式数据集执行层**。

如果裸 Ray API 让你手写：

```text
切 batch -> 提交 remote task -> 控制 inflight -> 收结果 -> 处理失败重试
```

那么 Ray Data 想提供的是：

```text
Dataset -> map / map_batches / filter / repartition -> 分布式执行 -> 流式消费结果
```

也就是说，Ray Data 把一批数据抽象成一个 `Dataset`，内部再把 Dataset 切成多个 block，由 Ray 调度到不同 worker 上执行。你不再直接维护一堆 `ObjectRef`，而是对数据集声明式地写 transformation：

```python
import ray


ds = ray.data.from_items([{"id": i} for i in range(1000)])


def process_batch(batch):
    batch["score"] = batch["id"] * 10
    return batch


result = (
    ds
    .map_batches(process_batch, batch_size=100, concurrency=8)
    .take_all()
)
```

这里的 `batch_size=100` 和手写 `iter_batches` 的思想一致：不要每条数据一个 Ray Task，而是让一次远程执行处理一批数据。区别在于，Ray Data 会帮你管理 block、调度、流水线执行、资源声明、结果迭代和部分容错。

## 9. Ray Data 适合什么，不适合什么

Ray Data 最适合的是**数据集形态明确的批处理或流式批处理**：

- 输入可以自然表达成一批 row，例如文件列表、对象存储路径、Parquet、CSV、图片 URI、Python list。
- 每条数据或每个 batch 的处理逻辑相对独立，适合 `map` / `map_batches`。
- 中间结果可以交给 Ray Data 管理，例如 shuffle、repartition、limit、materialize、write parquet。
- 你希望少写调度代码，不想自己维护 `pending refs`、`ray.wait`、worker 数量和 block 级流水线。
- 需要和 Ray Train、Ray Serve、批量推理等生态对接。

典型例子包括批量图片预处理、离线特征抽取、embedding 生成、Parquet 转换、模型批量推理、日志清洗、样本过滤。

但 Ray Data 不是万能默认选项。下面几类场景，裸 Ray Task / Actor 往往更合适。

| 问题特征 | 更适合 |
|----------|--------|
| `read -> map_batches -> filter -> write` | Ray Data |
| 批量推理、特征抽取、格式转换 | Ray Data |
| Dataset 里只放 URI / id，worker 内部读大对象 | Ray Data 可以用，但要避免传真实大对象 |
| 百万级轻量 metadata 需要过滤、join、重分区、shuffle | Ray Data |
| 单个业务对象已经是粗粒度任务 | 裸 Ray Task |
| 需要 early finish / force cancel straggler | 裸 Ray Task + `ray.wait` |
| summary 聚合需要完整扫描 | Ray Data 可考虑 |
| summary 聚合只要部分结果、需要提前收敛 | 裸 Ray Task |
| 需要数据库连接池、常驻模型、共享状态 | Actor / Actor Pool |
| 需要精细背压、稳定 hash 路由、自定义重试 | 裸 Ray Task + Actor |
| 任务量很小、脚本很短 | 单进程 / 线程池 / 少量 Ray Task |

一个实用判断标准是：

```text
我的问题更像“对一个数据集做一串变换”，还是更像“设计一个分布式服务/调度系统”？
```

如果是前者，优先考虑 Ray Data；如果是后者，优先考虑 Task + Actor。

Ray Data 的收益来自“轻量 row 的数据集调度”和“声明式数据变换”，不是来自把已有 `ray.remote` 代码机械换一层 API。工程里经常会混用：Ray Data 负责调度轻量 row，Actor 负责持有重资源，worker 内部按需读取真实数据。

## 10. 大对象不要直接塞进 Dataset

Ray Data 的标准心智是：Dataset 的 row / block 会被 Ray 管理，并可能经过 Object Store 在 worker 间流转。如果单条数据是图片、点云、视频片段、超大 numpy array，直接把真实数据放进 Dataset，可能带来额外序列化、网络传输、Object Store 内存和 spill 压力。

更好的方式通常是：**Dataset 只放轻量索引或 URI，真实数据在 worker 内部按需读取**。

```python
ds = ray.data.from_items([
    {"log_id": "xxx", "hdfs_path": "..."},
    {"log_id": "yyy", "hdfs_path": "..."},
])


def process_refs(batch):
    # worker 内部根据 log_id / path 读取真实图片、点云或视频
    # 不把大对象提前塞进 Ray Object Store
    ...
```

这个模式本质上是把 Ray Data 当成**带资源管理和容错能力的调度器**，而不是把它当成大对象搬运通道。

## 11. 生产坑：heartbeat 丢失与任务结束不了

Ray 集群里每个 worker node 上都有 raylet。raylet 会定期向 head / GCS 汇报 heartbeat，告诉控制面“这个节点还活着、资源状态是什么、任务执行是否还在推进”。如果 head / GCS 一段时间收不到某个节点的 heartbeat，就会把节点标记为 dead，相关 Task / Actor 会失败或被重调度。

坑在于：**heartbeat 丢失不一定等于机器真的死了**。更常见的是控制面被拖慢，或者节点处在“没死透、也不返回”的灰色状态。

常见触发链路是：

```text
单个 task / actor 处理过大 batch
  -> 内存暴涨、Object Store spill、磁盘 I/O 打满
  -> raylet 或 GCS 调度不到 CPU / 处理不过来
  -> heartbeat 延迟或丢失
  -> 节点被判 dead，或任务长时间无进展
  -> 上层看起来像“任务结束不了”
```

也可能是机器、容器或网络真的出了问题：pod 被驱逐、raylet OOM、head 节点压力过大、跨节点网络抖动、native 扩展卡死等。

Ray Data 对这类问题只能覆盖一部分。如果节点最终被判 dead，或者 Actor / Task 抛出明确异常，Ray Data 可以在 block 粒度重试、换 Actor 或换节点。但如果 worker 进程还活着，只是卡在 I/O、死循环、native hang、坏数据解析，既不抛异常也不返回，Ray Data 很难自动判断业务已经卡死。

生产里通常要补三层兜底：

1. **控制资源压力**：减小 batch，限制 `concurrency` / `max_inflight`，避免真实大对象进入 Object Store。
2. **给执行加活性检测**：driver 侧设置等待超时，Ray Data 结果迭代设置 idle timeout，业务层记录进度心跳。
3. **隔离坏数据**：单条数据失败要 catch 后标记 failed，不要让一条坏数据拖死整个 bucket；可重试错误再显式抛出，让 Ray 或 Ray Data 重调度。

### 11.1 裸 Ray：用 ray.wait(timeout) 发现长时间无结果

手写 Task Pool 时，不要无限等待所有 refs。可以给滑动窗口加 idle timeout：一段时间没有任何 Task 完成，就取消 pending，并把当前 batch 信息落日志或失败队列。

```python
import time
import ray


def submit_with_idle_timeout(batch_iter, max_inflight: int, idle_timeout_s: int):
    pending = []
    completed = []
    last_progress = time.monotonic()

    for batch in batch_iter:
        pending.append(process_batch.remote(batch))

        if len(pending) < max_inflight:
            continue

        ready, pending = ray.wait(pending, num_returns=1, timeout=5)
        if ready:
            completed.extend(ray.get(ready))
            last_progress = time.monotonic()
            continue

        if time.monotonic() - last_progress > idle_timeout_s:
            for ref in pending:
                ray.cancel(ref, force=True)
            raise TimeoutError(f"no task finished for {idle_timeout_s}s")

    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=5)
        if ready:
            completed.extend(ray.get(ready))
            last_progress = time.monotonic()
        elif time.monotonic() - last_progress > idle_timeout_s:
            for ref in pending:
                ray.cancel(ref, force=True)
            raise TimeoutError(f"no task finished for {idle_timeout_s}s")

    return completed
```

这个 timeout 不是为了替代 Ray 的节点死亡检测，而是为了处理“Ray 没判死，但业务已经没有进展”的情况。线上异常里最好带上 `batch_id`、输入范围、worker host、重试次数，方便定位是坏数据还是节点问题。

### 11.2 Ray Data：消费结果时加 idle timeout

Ray Data 的 `map_batches` 会管理 block 调度，但 driver 侧消费结果时仍然需要判断“是否长期没有任何 batch 产出”。可以把结果迭代包一层 idle timeout。

```python
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout


def iter_batches_with_idle_timeout(ds, idle_timeout_s: int):
    sentinel = object()
    iterator = ds.iter_batches(batch_size=None)

    def next_batch():
        try:
            return next(iterator)
        except StopIteration:
            return sentinel

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        while True:
            future = pool.submit(next_batch)
            try:
                batch = future.result(timeout=idle_timeout_s)
            except FutureTimeout as e:
                raise TimeoutError(
                    f"ray data made no progress for {idle_timeout_s}s"
                ) from e

            if batch is sentinel:
                break

            yield batch
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
```

更完整的写法通常会在每个 worker 返回的结果里带上 `bucket_id`、`status`、`failed_items` 和 `trace`。这样 driver 即使因为 idle timeout 中断，也知道最后成功推进到哪里，哪些 bucket 需要重跑。

```python
def run_one_bucket(row: dict) -> dict:
    bucket_id = row["bucket_id"]
    failed = []
    output = []

    for item in row["input_tasks"]:
        try:
            output.append(process_one_item(item))
        except BadDataError as e:
            failed.append({"item": item, "error": str(e)})
            continue

    return {
        "bucket_id": bucket_id,
        "status": "ok" if not failed else "partial_failed",
        "output": output,
        "failed_items": failed,
    }
```

坏数据不要默认抛到整个 Ray Data block 失败。只有环境问题、节点问题、外部依赖瞬时失败，才应该抛可重试异常，让 Ray Data 换 Actor 或换节点重试。

### 11.3 业务层：用进度 Actor 判断 worker 是否还在推进

如果一个 bucket 内部会处理很久，单靠 Ray 的 heartbeat 不够。raylet 活着不代表业务逻辑还在推进。可以让 worker 定期向一个轻量 Actor 上报进度。

```python
import time
import ray


@ray.remote
class ProgressTracker:
    def __init__(self):
        self.last_seen = {}

    def report(self, bucket_id: int, processed: int):
        self.last_seen[bucket_id] = {
            "processed": processed,
            "ts": time.time(),
        }

    def snapshot(self):
        return dict(self.last_seen)


@ray.remote
def process_bucket(bucket_id: int, items: list, progress):
    results = []
    for i, item in enumerate(items):
        results.append(process_one_item(item))
        if i % 100 == 0:
            progress.report.remote(bucket_id, i)
    progress.report.remote(bucket_id, len(items))
    return results
```

driver 侧定期检查 `snapshot()`，如果某个 bucket 的 `processed` 长时间不变，就把它视为业务卡死，而不是继续等 Ray 自己判断节点 dead。

```python
def check_stalled(progress, stale_s: int):
    now = time.time()
    snapshot = ray.get(progress.snapshot.remote())
    stalled = [
        bucket_id
        for bucket_id, info in snapshot.items()
        if now - info["ts"] > stale_s
    ]
    if stalled:
        raise TimeoutError(f"stalled buckets: {stalled}")
```

Ray heartbeat 判断节点 / raylet 是否活着；业务进度心跳判断你的数据处理是否还在向前推进。生产问题里，后者往往更能解释“为什么任务结束不了”。

## 12. 一个 2000w 数据处理系统应该怎么设计

把前面的概念串起来，可以得到一个完整的数据处理 pipeline：

```text
Driver
  ├── 按 batch 切分数据范围，不搬运真实数据
  ├── 创建 Writer Actor Pool
  ├── 提交 process_batch.remote(...)
  ├── 用 ray.wait 控制 max_inflight
  └── 汇总各 batch 与各 writer 的统计
```

为了方便对照实现，我整理了一个最小可运行 demo：[ray-demo](https://github.com/lucienne999/ray-demo)。它不依赖具体业务仓库，只模拟“2000w 条记录按 batch 处理、Ray Task 做计算、Writer Actor Pool 做写入、Driver 做背压和兜底”的通用形态。

![Ray 批处理 demo 职责分层与执行流](/images/notes/ray/ray-pipeline-flow.png)

代码可以按职责拆成四层：

```text
demo.py                  # 程序入口：解析参数、初始化 Ray、创建 writer pool、打印吞吐
src/ray_demo/driver.py   # 调度层：切 batch、提交 task、ray.wait 背压、timeout / partial success
src/ray_demo/tasks.py    # 计算层：worker 内生成/读取数据，处理一个 batch，返回 BatchSummary
src/ray_demo/actors.py   # 状态层：WriterActor 持有文件句柄、锁和全局统计
src/ray_demo/models.py   # 数据结构：ProcessingConfig、BatchSummary、ProcessingResult
```

本地快速跑一个小规模版本：

```bash
python3 demo.py --records 100000 --batch-size 5000 --max-inflight 8 --writers 2 --writer-concurrency 2
```

完整模拟 2000w 条记录：

```bash
python3 demo.py --records 20000000 --batch-size 50000 --max-inflight 64 --writers 4 --writer-concurrency 4 --work-factor 8
```

这个 demo 的关键点不是“造一个假的业务逻辑”，而是把后面几节的工程判断跑通：Driver 不持有 2000w 条真实数据，只提交 `(start_id, end_id)`；Task 在 worker 内部处理自己的 batch；WriterActor 统一写结果并维护统计；Driver 用 `ray.wait(timeout=...)` 判断是否长时间无进展，并根据 `min_success_rate` 决定是否允许提前收尾。

### 12.1 任务拆分：不要每条数据一个 Task

2000w 条数据如果每条一个 Task，就是 2000w 个 Task。Ray Task 虽然轻量，但每个 Task 都有调度、序列化、ObjectRef 管理的固定开销，千万级 Task 足以把调度器和 Object Store 压垮。

正确做法是**按 batch 切分**，让每个 Task 处理一段连续区间：

```python
def iter_batches(records: int, batch_size: int):
    batch_id = 0
    for start_id in range(0, records, batch_size):
        end_id = min(start_id + batch_size, records)
        yield batch_id, start_id, end_id
        batch_id += 1
```

如果 `batch_size=50_000`，2000w 条数据就是 400 个 Task。Task 数量从千万级降到数百级，调度开销才会进入可控区间。

注意传给 Task 的只是 `(start_id, end_id)` 区间，而不是真实数据。真实数据在 worker 内部按 id 读取或生成，避免 driver 把 2000w 条记录序列化后塞进 Object Store。

### 12.2 职责边界：Task 做计算，Actor 做写入

一个清晰的职责划分是：

- **Task**：无状态计算，负责造数据、读数据、算分、校验，返回 batch 统计摘要。
- **Actor**：有状态资源，负责持有输出文件句柄、数据库连接池、累计统计、写入状态。

```python
@ray.remote
def process_batch(batch_id, start_id, end_id, writer, work_factor):
    # worker 内部读取或生成真实数据
    # 执行 CPU 计算、校验、聚合
    summary = BatchSummary(...)
    ray.get(writer.write_batch.remote(summary))
    return summary
```

writer 可以是一个带并发和锁的 Actor：

```python
@ray.remote(max_concurrency=4)
class WriterActor:
    def __init__(self, writer_id: int, output_path: str):
        self.lock = threading.Lock()
        self.records = 0
        self.output_path = Path(output_path)

    def write_batch(self, summary) -> None:
        with self.lock:
            with self.output_path.open("a") as f:
                f.write(json.dumps(asdict(summary)) + "\n")
            self.records += summary.total
```

这里复用了前面的 Actor 规则：`max_concurrency > 1` 时，内部共享状态和文件写入必须自己加锁。

### 12.3 提交节奏：用 ray.wait 控制 max_inflight

即使只有 400 个 batch，也不一定要一次性提交。更稳妥的做法是用滑动窗口控制同时在飞的 Task 数。

```python
def submit_with_backpressure(config, writers):
    pending, completed = [], []

    for batch_id, start_id, end_id in iter_batches(config.records, config.batch_size):
        writer = pick_writer(writers, batch_id)
        ref = process_batch.remote(
            batch_id,
            start_id,
            end_id,
            writer,
            config.work_factor,
        )
        pending.append(ref)

        if len(pending) >= config.max_inflight:
            ready, pending = ray.wait(pending, num_returns=1)
            completed.extend(ray.get(ready))

    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        completed.extend(ray.get(ready))

    return completed
```

`pick_writer` 可以用 `sha1(batch_id)` 把 batch 稳定路由到某个 writer，让写入压力均摊到 Actor Pool。

### 12.4 串行基线：不要只看 Ray 跑得快不快

为了量化 Ray 的加速效果，最好保留一条串行基线：单进程跑一个小样本，测出单核吞吐后，投影到完整数据量需要多久，再和 Ray 的实际吞吐对比。

这样可以回答两个关键问题：

- Ray 是否真的带来了加速？
- 当前瓶颈是在 CPU 计算、I/O 写入、Object Store，还是调度开销？

如果任务本身几乎不耗 CPU，只是很轻的 Python 逻辑，那么并行收益很容易被调度和通信开销稀释。反过来，计算越重、batch 粒度越合理，Ray 的收益越明显。

## 13. 最后给一个选型速查

| 场景 | 推荐模式 | 关键设置 |
|------|----------|----------|
| 无状态批量计算 | Task | 粗粒度 batch，避免单条一个 Task |
| 全局计数器、配置中心 | 单个默认 Actor | 默认串行，不加 `max_concurrency` |
| 数据库连接池、批量写入 | Actor Pool + 稳定 hash | 控制 actor 数、连接池大小、并发数 |
| HTTP 爬虫、外部 API 调用 | Async Actor Pool | `async def` + 限流 |
| GPU 模型常驻推理 | Actor / Actor Pool | `num_gpus=1`，按模型和 GPU 分片 |
| 规则数据集变换 | Ray Data | `map_batches`、`concurrency`、轻量 row |
| 需要 early finish / cancel | 裸 Ray Task + `ray.wait` | idle timeout、显式 cancel |
| 大对象数据处理 | Ray Data + URI / Task 内读取 | 不把图片、点云、视频直接塞进 Object Store |

这篇笔记最终想沉淀的判断是：

**Ray 不是让你把所有代码都改成 `.remote()`，而是要求你先区分计算、状态、数据流和控制流。**

如果问题是可规则切分的数据变换，优先考虑 Ray Data；如果问题是强状态服务、复杂调度、提前终止、稳定路由，优先使用 Task + Actor。大规模数据处理里最稳的结构通常是：**轻量 metadata 驱动调度，worker 内部读取真实数据，Task 处理粗粒度 batch，Actor Pool 持有重资源，driver 用 ray.wait 控制背压，并在业务层补进度心跳和超时兜底。**
