---
title: "大模型训练之 DCP 代码解析和使用"
category: "ai-infra"
tags: ["pytorch", "llm"]
date: 2026-02-26
---

## DCP 深入浅出

本文档是大模型训练的模型保存相关的内容，主要介绍了 DCP 的原理和使用方法。

在DCP实操中，我们会开发一个工具，可以从 dcp 转换成原本 torch.load 以及 safetensor的格式。

本文的代码参考 <https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint> , commit 为 36bc2bf

## Intro

DCP（Distributed Checkpoint） 是 pytorch 中用于分布式保存和加载模型参数的工具，DCP 避免了保存过程的参数聚合，提升了训练效率，在加载过程中支持参数的重新分片（resharding），可以从任意设备数量重新开始训练；

DCP 采用分离式架构，核心组件为：

- Planner： 决定模型参数哪些部分需要被保存（`SavePlanner`）和加载（`LoadPlanner`），具体实现在 [`torch/distributed/checkpoint/planner.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/planner.py)
- Storage（Writer/Reader）： 根据planner生成的计划与物理存储交互，具体实现在 [`torch/distributed/checkpoint/storage.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/storage.py)
- Stateful: 状态协议，允许用户自定义对象的参数状态，任何实现了 `state_dict` 和 `load_state_dict` 方法的对象都可以被 DCP 自动处理

通俗来说，如果把 DCP 的保存过程比作“发快递”，那么：

- Planner (物流调度) : 负责决定怎么打包、怎么分车、走哪条路线（分片、去重）。
- Storage (快递员) : 负责把包裹（数据）搬运到目的地（磁盘）。
- Stateful (发货人) : 负责提供要寄的物品。有时候物品是一个复杂的机器（对象），发货人需要先把它拆解成零件（ state\_dict ），快递员和调度员才能处理。

## Details

本小节将 深入源码 介绍 DCP 各个模块的 具体职能 。为了方便理解后续内容，我们先介绍一些核心的基础数据结构。

### 基础数据结构 (Core Data Structures)

这些数据结构定义了 Checkpoint 系统中“数据如何被描述”以及“I/O 任务如何被表达”。

- TensorProperties : 它是 Tensor 的“身份证” ，用于描述一个 Tensor 的 非数据属性 （如数据类型、内存布局、梯度需求等）。它不直接描述切分状态，而是作为元数据的一部分，确保在加载时能还原出属性完全一致的 Tensor 对象。
  ```python
  @dataclass
  class TensorProperties:
      """Properties used to create :class:`Tensor`"""

      # Regular tensor fields
      dtype: torch.dtype = field(default_factory=torch.get_default_dtype)
      # This field is deprecated.
      layout: torch.layout = field(default=torch.strided)
      # This field is deprecated.
      requires_grad: bool = False
      # This field is deprecated.
      memory_format: torch.memory_format = field(default=torch.contiguous_format)
      # This field is deprecated.
      pin_memory: bool = False
  ```
- WriteItem & ReadItem : 它们本质上分别是“写入指令”和“读取指令”，本身并不承载实际的数据 Payload 。
  - WriteItem (发货单) : 告诉底层的 Storage Writer：“请把我内存里的这个对象（通过 Index 查找），按照这个规格（Size/Chunk Offset），写入到 Checkpoint 里的这里（Storage Index/Key）。”
  - 它包含了 辅助结构 WriteData 来详细描述待写入数据的物理属性：
    - BytesIOWriteData : 用于普通 Python 对象，记录写入的字节数 ( nbytes )。
    - TensorWriteData : 用于 Tensor 对象，记录 分片信息 ( chunk ) 、 Tensor 属性 ( properties ) 以及 全局大小 ( size ) 。
  - ReadItem (取货单) : 告诉底层的 Storage Reader：“请从 Checkpoint 里的这里（Storage Index/Offset），读取这么长一段数据（Lengths），然后放到我内存里的这里（Dest Index/Offset）。”
  ```python
  @dataclass(frozen=True)
  class TensorWriteData:
      chunk: ChunkStorageMetadata
      properties: TensorProperties
      size: torch.Size

  @dataclass(frozen=True)
  class BytesIOWriteData:
      nbytes: int

  @dataclass(frozen=True)
  class WriteItem:
      """Dataclass which holds information about what needs to be written to storage."""

      index: MetadataIndex
      type: WriteItemType

      # Size of bytesIO data to be written.
      bytes_io_data: BytesIOWriteData | None = None

      # Value present if it's a tensor write
      tensor_data: TensorWriteData | None = None

      def tensor_storage_size(self) -> int | None:
          """
          Calculates the storage size of the underlying tensor, or None if this is not a tensor write.

          Returns:
              Optional[int] storage size, in bytes of underlying tensor if any.
          """
          if self.tensor_data is None:
              return None

          numels = reduce(operator.mul, self.tensor_data.size, 1)
          dtype_size = torch._utils._element_size(self.tensor_data.properties.dtype)
          return numels * dtype_size

  @dataclass(frozen=True)
  class ReadItem:
      # Read Item
      type: LoadItemType

      # Index into the state_dict
      dest_index: MetadataIndex
      # Offsets into destination tensor
      dest_offsets: torch.Size

      # Index into the checkpoint
      storage_index: MetadataIndex
      # Offset into the checkpoint data
      storage_offsets: torch.Size

      # Size of the hypercube to copy
      lengths: torch.Size
  ```
- SavePlan & LoadPlan : 分别是保存计划和加载计划的容器 。它们是基于 WriteItem 和 ReadItem 的 顶层封装 ， 代表了当前 Rank 在本次 Checkpoint 操作中负责的所有 I/O 任务清单；
  ```python
  @dataclass(frozen=True)
  class SavePlan:
      items: list[WriteItem]
      storage_data: Any = None
      planner_data: Any = None
      # This is used to indicate that the ranks should
      # use the cached plans to write data instead.
      usable: bool = True # 这是一个优化选项，可以便于复用缓存，作用在之后会提到

  @dataclass
  class LoadPlan:
      items: list[ReadItem]
      storage_data: Any = None
      planner_data: Any = None
  ```

后续介绍的 Metadata 、 Planner 和 Storage 模块，正是基于上述基础结构，结合实际的 分布式 场景，实现了复杂的 元数据管理 、 任务规划 与 数据存储 功能。

### Metadata

`Metadata` 是 DCP 的核心描述对象，通常会序列化为磁盘上的 `.metadata` 文件。`.metadata` 本身不承载大块 tensor payload，它更像是 checkpoint 的逻辑目录和索引入口；只有将 `.metadata` 与同目录下的 `.distcp` 分片文件结合起来，系统才能完整还原一个 checkpoint。

从工程视角看，`Metadata` 的作用可以拆成两层：

- 描述逻辑结构：记录 state\_dict 中有哪些 key、每个 key 对应的是 Tensor 还是普通 Python 对象，以及它们的全局形状和属性。
- 提供恢复所需的全局视图：让 Loader 先看到 checkpoint 中“有哪些逻辑对象、每个对象被切成了哪些 chunk”。至于 chunk 最终落在哪个物理文件、如何定位到具体 payload，则还需要结合 storage backend 维护的寻址信息。

具体代码实现在：[`torch/distributed/checkpoint/metadata.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/metadata.py)。更严谨地说，`Metadata` 负责描述 checkpoint 的逻辑内容，而物理文件名、路径映射、offset 等 backend 细节，通常由 storage 层额外维护或序列化。下面按“全局目录 -> 存储对象 -> 基础构件”的顺序来看。

#### Global View

- `Metadata`：对应 `.metadata` 文件在内存中的对象，可以理解为整个 checkpoint 的全局目录。
  - `state_dict_metadata`：核心字典 `dict[str, STORAGE_TYPES]`。
  - Key：state\_dict 中数据的全限定名 FQN（如 `model.layer1.weight`）。
  - Value：具体的存储描述对象（`TensorStorageMetadata` 或 `BytesStorageMetadata`）。
  - 这张表回答的是 checkpoint 中“有哪些逻辑对象”以及“每个对象的全局形态是什么”。
  - `planner_data`：Planner 的私有补充信息，例如用于恢复嵌套结构、mapping 或其他规划阶段上下文。
  - `storage_data`：storage backend 的私有扩展字段。它不一定是 `Metadata` 核心语义的一部分，更适合理解为“与逻辑目录一同持久化的 backend 寻址信息”。
- `MetadataIndex`：用于唯一标识一个待保存/待加载对象的位置索引。最核心的组成通常是数据的 FQN，以及当对象是 Tensor 分片时对应的 offset 信息。
  - 对普通对象来说，`MetadataIndex` 基本可以理解为“这个 key 是谁”。
  - 对 Tensor 分片来说，`MetadataIndex` 还会进一步区分“这个 key 的哪一块数据”。
  - 在保存阶段，Planner 会使用它来聚合同一个逻辑对象的多个 `WriteItem`，并在必要时做去重或重写索引提示。

#### 存储对象描述 (Storage Object Description)

- `TensorStorageMetadata` (Tensor 存储元数据) :
  当 Checkpoint 中的某个 Key 对应的是 Tensor 数据时，使用此对象进行描述。它代表了 该 Tensor 在存储系统中的全局完整形态 ，是 Coordinator 将所有 Rank 上报的分片信息进行 聚合与整理后的结果 。
  - chunks (分片列表) : 类型为 list\[ChunkStorageMetadata] 。
  - 这是 核心拼图信息 。它详细记录了构成该全局 Tensor 的所有 物理切片 (Chunks) 。
  - 每个元素对应一个具体的切片，描述了该切片在全局 Tensor 中的位置（Offset）和大小（Size）。例如：列表中的第一个 Chunk 可能对应 Tensor 的 \[0:512, :] 区域，第二个 Chunk 对应 \[512:1024, :] 区域。
  - size (全局尺寸) : 类型为 torch.Size （如 \[4096, 4096] ）。
  - 它描述了 Tensor 未被切分前的原始完整形状 。无论该 Tensor 被物理切分成多少个碎片文件，这个 Size 始终提供统一的逻辑视图。
  - properties (元属性) : 类型为 TensorProperties 。
  - 记录了该 Tensor 的 非数据属性 （如 Dtype, Layout, RequiresGrad 等）。这些属性对于所有分片都是共享且一致的，确保加载时能还原出正确的 Tensor 对象。
- BytesStorageMetadata (字节流存储元数据) :
  当 Value 是普通 Python 对象（如 int, str, list）时，使用此对象。目前仅作为类型标识，不包含其他属性；

#### 基础构件 (Building Blocks)

- ChunkStorageMetadata (分片元数据) :
  这是 TensorStorageMetadata.chunks 列表中的元素，描述了一个大 Tensor 中的 一块具体切片 的位置和大小。
  - offsets : 该切片在全局 Tensor 中的起始坐标（如 \[0, 512] ）。
  - sizes : 该切片的大小（如 \[1024, 512] ）。
  - 有了它，系统就知道这块数据属于完整 Tensor 的哪个部分，从而能像拼图一样把它拼回去。
- TensorProperties (Tensor 属性) :
  这是 TensorStorageMetadata.properties 字段的内容，充当 Tensor 的“身份证”。
  - 包含字段 : dtype (数据类型), layout (内存布局), requires\_grad (梯度需求) 等。
  - 确保加载时能还原出属性完全一致的 Tensor 对象。

一个简化后的 `.metadata` 逻辑视图大概会如下所示：

```
Metadata(
    # 1. 核心数据字典
    state_dict_metadata={
        
        # --- Tensor 对象 ---
        "model.layer1.weight": TensorStorageMetadata(
            # 全局属性
            properties=TensorProperties(
                dtype=torch.float32,
                layout=torch.strided,
                requires_grad=True,
                pin_memory=False
            ),
            # 全局大小 (1024x1024)
            size=torch.Size([1024, 1024]),
            # 分片列表 (由两个 Rank 贡献)
            chunks=[
                # Chunk 0 (Rank 0 贡献的上半部分)
                ChunkStorageMetadata(
                    offsets=torch.Size([0, 0]),
                    sizes=torch.Size([512, 1024])
                ),
                # Chunk 1 (Rank 1 贡献的下半部分)
                ChunkStorageMetadata(
                    offsets=torch.Size([512, 0]),
                    sizes=torch.Size([512, 1024])
                )
            ]
        ),

        # --- 普通 Python 对象 ---
        "optim.step": BytesStorageMetadata()
    },

    # 2. Planner 辅助数据 (用于恢复嵌套结构)
    planner_data={
        "model.layer1.weight": ("model", "layer1", "weight"),
        "optim.step": ("optim", "step")
    },
    
)
```

对于物理文件名、路径映射、chunk 到具体 `.distcp` 文件的定位关系，更适合把它理解为 storage backend 侧的附加寻址信息。例如在文件系统实现里，可能存在类似下面这样的 backend 视图：

```python
storage_data = {
    "model.layer1.weight": {
        "chunk_0": "file_rank_0.pt",
        "chunk_1": "file_rank_1.pt"
    },
    "optim.step": "file_rank_0.pt"
}
```

也就是说：`Metadata` 负责描述“checkpoint 里有什么、逻辑上长什么样”，而 storage backend 的附加信息负责回答“这些 payload 物理上存在哪里、如何被精确读到”。

### Planner

`DefaultSavePlanner` 和 `DefaultLoadPlanner` 是 DCP 默认使用的保存规划器和加载规划器。它们实现了 `SavePlanner` 和 `LoadPlanner` 接口，负责将用户的 state\_dict 转换为**实际的读写**操作计划（Plan）。

#### DefaultSavePlanner

从工程视角看，`DefaultSavePlanner` 主要做两件事：

1. 决定当前 Rank 需要写哪些数据，并把本地 state\_dict 转成一组 `WriteItem`。
2. 在 Coordinator 侧将所有 Rank 的写计划聚合成全局 `Metadata`，同时对重复上报的数据做去重（deduplicate），避免无意义的重复落盘。

具体实现在 [`torch/distributed/checkpoint/default_planner.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/default_planner.py) 中的 `DefaultSavePlanner`，实现过程可以概括为以下几步：

1. 展开嵌套 ShardedTensor，将其转换为最内层的 LocalShard 列表。

- 什么是嵌套的 ShardedTensor ？

  这里的嵌套通常发生在\*\*混合并行（Hybrid Parallelism）\*\*的场景下，例如：
  - 外层 是 Pipeline Parallelism (PP) 或 Tensor Parallelism (TP) 导致的分片。
  - 内层 是 FSDP (Fully Sharded Data Parallel) 导致的分片。
  举个例子：
  - 外层 ShardedTensor : 比如模型被切分成了 2 部分（PP），当前 Rank 持有第 1 部分。
  - 内层 ShardedTensor : 这第 1 部分本身又因为 FSDP 被进一步切分成了 8 份，当前 Rank 持有其中的 1 份。
  在 Python 对象中，这表现为：`ShardedTensor` 对象里包裹的不是普通的 `torch.Tensor`，而是另一个 `ShardedTensor`。
- 展开会做什么？

  核心目标是消除递归结构，直接拿到最内层的 local\_shards，并将所有的相对坐标统一为全局绝对坐标。
  这一步涉及全局 Offset 的计算： 外层 Offset ( outer\_shard.metadata.shard\_offsets ) + 内层 Offset ( inner\_shard.metadata.shard\_offsets ) = 全局绝对 Offset 。例如：外层偏移 512，内层偏移 0 -> 全局偏移 512。这段代码的相关实现在 [`torch/distributed/checkpoint/_sharded_tensor_utils.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/_sharded_tensor_utils.py) 。

  转换前(Nested):
  ```python
  ShardedTensor(
      global_size=[1024],
      local_shards=[
          Shard(
              offset=[0], size=[512],
              tensor=ShardedTensor(  <-- 嵌套！
                  global_size=[512],
                  local_shards=[
                      Shard(offset=[0], size=[64], tensor=RealTensor([64]))
                  ]
              )
          )
      ]
  )
  ```
  转换后(Flattened):
  ```python
  ShardedTensor(
      global_size=[1024],
      local_shards=[
          Shard(
              offset=[0], size=[64],  <-- 直接指向最内层的数据位置
              tensor=RealTensor([64])
          )
      ]
  )
  ```

1. 生成全局 Metadata（Create Global Metadata）

上一步已经把所有 local shard 统一到了同一个坐标系。接下来，Coordinator 会收集所有 Rank 上报的 `SavePlan`，把这些局部写计划合成为一份全局 `Metadata`。这一阶段通常发生在 `create_global_plan` 中，可以理解为“把所有人手里的局部视图拼成一张全局目录表”。

在进入完整聚合之前，Coordinator 还会先看 `SavePlan.usable`，判断当前是否可以直接复用缓存的全局 plan / metadata；如果不能复用，才进入完整规划流程。对工程使用者来说，可以把它理解为一个“避免每次保存都重新做全量规划”的优化开关。

完整流程大致分成四步：

- 收集与合并（Aggregation）：
  Coordinator 收到各个 Rank 的 `SavePlan` 后，会按逻辑对象聚合这些 `WriteItem`。例如同一个 Tensor `model.layer1.weight`，可能同时收到来自 Rank 0 的 chunk A（`[0, 256]`）和来自 Rank 1 的 chunk B（`[256, 512]`）。
- 构建 `TensorStorageMetadata`：
  对于 Tensor，Planner 会把所有 chunk 聚合成一个 `TensorStorageMetadata`，其中包含：
  - 全局属性（`properties`）：如 dtype、layout 等；
  - 全局大小（`size`）：Tensor 的完整逻辑形状；
  - 分片列表（`chunks`）：所有 Rank 实际持有并上报的 `ChunkStorageMetadata`。
- 重写索引提示（Rewrite Index Hints）：
  聚合完成后，Planner 会更新 `WriteItem.index` 中的提示信息，让它更快地定位到 `Metadata` 中对应的 chunk 位置，减少后续查找成本。
- 验证完整性（Validation）：
  这是生成全局 Metadata 前最重要的一步。Planner 会检查所有 chunk 是否存在空洞、重叠或越界，确保这些分片最终能拼成一个完整、合法的全局 Tensor。

1. deduplicate（全局去重）

除了聚合之外，`DefaultSavePlanner` 在创建全局 Metadata 时还会顺手做一层“全局去重”。

这里更准确的理解不是“基于内容哈希的通用全局去重”，而是：如果多个 Rank 上报的是**同一个逻辑对象的同一逻辑位置**，那最终没有必要让每个 Rank 都往磁盘各写一遍。最典型的场景是：

- 某些参数在多个 Rank 上都保留了完整副本；
- 某些非分片对象（例如普通 Python 状态）在多个 Rank 上内容相同；
- 某个 Tensor chunk 因为并行策略的原因被多个 Rank 重复持有。

在这类情况下，Planner 会借助 `MetadataIndex` 来判断这些 `WriteItem` 是否指向同一个逻辑位置；如果答案是“是”，那它会只保留一份实际写入，其余重复项在全局计划里被折叠掉，或者改写为共享同一个 metadata / storage 定位信息。

换句话说，这一步主要解决的是“同一份逻辑数据被重复上报”带来的写放大，而不是对任意两个内容恰好相同、但逻辑身份不同的对象做内容寻址式 dedup。

可以把这个过程理解为：

- 聚合回答的是：“这些分片拼起来长什么样？”
- 去重回答的是：“哪些上报其实是重复的，不需要重复写？”

举个简化例子：

- Rank 0 和 Rank 1 都持有一个完全相同的 `optimizer.step`；
- 两边在本地都会生成一个 `WriteItem`；
- Coordinator 聚合时发现它们对应的是同一个 `MetadataIndex`，而且内容上属于同一个逻辑对象；
- 最终全局 Metadata 里只保留一个 `BytesStorageMetadata` 条目，物理层面也只需要一份写入。

这样做的收益很直接：

- 减少无意义的磁盘写放大；
- 缩小 checkpoint 体积；
- 避免加载时面对多个等价副本，简化寻址逻辑。

最后，如果当前保存流程启用了 plan cache，那么 Coordinator 还会把新生成的全局 plan 和 metadata 缓存起来，供下次 checkpoint 复用。

#### DefaultLoadPlanner

负责规划“如何加载”：它根据 Checkpoint 的元数据（Metadata）和当前的 state\_dict ，计算出每个 Rank 需要读取哪些数据分片，并支持 自动的 Resharding（重切分） 。

1. 展开：将当前模型嵌套的 state\_dict 展平，以便与保存时的结构对齐
2. create local plan: 生成本地加载计划： 每个rank根据自己分配的情况以及 metadata，读取对应的tensor内容
3. create global plan: 当前实现下直接返回所有 Rank 的本地计划，看起来留了个接口，便于后续一些功能接入，比如有超过1w个rank读取，可能会把存储打爆，在这一步中就可以规划一下读取的逻辑；或者一些跨机房的的读取，也可以放在这里做；

### Storage

Storage 模块是 DCP 与底层存储系统交互的接口层，它负责具体的 I/O 操作（读写字节流），并向 Planner 屏蔽了存储介质的差异（本地文件系统、S3、HDFS 等）。

默认提供了基于文件系统的实现： FileSystemWriter 和 FileSystemReader。

- StorageWriter & StorageReader: 是 OS 写入写出的抽象的接口；

```python
class StorageWriter(abc.ABC):
    """
    Interface used by ``save_state_dict`` to write to storage.

    One StorageWriter instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expect the following sequence of calls.

    0) (all ranks) set checkpoint_id if users pass a valid checkpoint_id.
    1) (all ranks) set_up_storage_writer()
    2) (all ranks) prepare_local_plan()
    3) (coordinator) prepare_global_plan()
    4) (all ranks) write_data()
    5) (coordinator) finish()
    """
    ...

class StorageReader(abc.ABC):
    """
    Interface used by ``load_state_dict`` to read from storage.

    One StorageReader instance acts as both the coordinator and the follower
    in a distributed checkpoint. As part of initialization, each instance
    is told its role.

    A subclass should expected the following sequence of calls by ``load_state_dict``:

    0) (all ranks) set checkpoint_id if users pass a valid checkpoint_id.
    1) (all ranks) read_metadata()
    2) (all ranks) set_up_storage_reader()
    3) (all ranks) prepare_local_plan()
    4) (coordinator) prepare_global_plan()
    5) (all ranks) read_data()
    """
    ...

```

- FileSystemWriter ：它通过多重继承 ( \_FileSystemWriter + BlockingAsyncStager ) 实现了 高效落盘 与 高效内存搬运 的双重能力。
  - \_FileSystemWriter (磁盘交互层): 负责将 Tensor 数据写入文件系统。
    - 文件格式 : 生成如 \_\_0\_0.distcp 的分片数据文件。
    - 并发写入 : 内部维护 queue.Queue 和线程池，提高磁盘吞吐量。
    - 原子性保证 : 采用 Write tmp -> fsync -> Rename 策略，杜绝“写了一半”的损坏状态。
    - 流水线搬运 ( \_OverlappingCpuLoader ) : 使用 Bucket 机制聚合小 Tensor，并实现流水线写入。在 dcp.save (同步) 时负责 GPU->Disk，在 dcp.async\_save (异步) 时负责 CPU->Disk。
  - BlockingAsyncStager (内存搬运层 - 仅用于 dcp.async\_save ):
    - 负责将 GPU 显存数据搬运到 CPU 内存 (Device -> Host)，这个过程叫 Staging 。
    - 特性 : "Blocking" 指的是在 Staging 阶段会 阻塞主线程 等待 D2H 拷贝完成，确保数据一致性（Snapshot）。
    - 优化手段 :
      1. 内存复用 : 维护 self.state\_dict\_cache ，避免反复 malloc/free。
      2. Pinned Memory : 启用 DMA 传输，加速 D2H 拷贝。
    - 注意： dcp.save (同步保存) 不经过此层，直接由 Writer 边读边写。
    - 关于异步保存可以查看：<https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html>
- FileSystemReader :
  - Writer 涉及 GPU -> CPU (或 Disk) 的搬运，但 Reader 只负责 Disk -> CPU 。
  - 原因 : 上层的 LoadPlanner 负责复杂的 Resharding (重切分) 操作，这在 CPU 上做更高效且安全（防 OOM）。
  - 实现 : 按需加载，精确切片，只把数据搬到 CPU 门口。

### Stateful

`Stateful` 协议是 DCP 系统中用于定义“可保存对象”的标准接口。任何实现了 `state_dict` 和 `load_state_dict` 方法的对象，都可以被 DCP 视为 Stateful 对象，从而被自动保存和加载。

代码位置：[`torch/distributed/checkpoint/stateful.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/stateful.py)

它的核心作用是将复杂的 Python 对象（如 `nn.Module`, `Optimizer`, 或自定义类）转换为 DCP 可以理解的 Key-Value 字典（即 State Dict），反之亦然。

在 DCP 的架构中，Stateful 对象充当了 **数据源（Source）** 和 **数据目的地（Destination）** 的角色：

- **保存时**：Planner 会调用 Stateful 对象的 `state_dict()` 方法，获取当前的状态数据（可能是 Tensor，也可能是普通 Python 对象）。
- **加载时**：Planner 会先从 Storage 读取数据，然后调用 Stateful 对象的 `load_state_dict()` 方法，将恢复后的状态注入回对象中。

PyTorch 原生的 `torch.nn.Module` 和 `torch.optim.Optimizer` 已经隐式或显式地支持了这一协议，因此可以直接作为 DCP 的输入。用户也可以通过实现这两个方法，让自定义的类支持分布式 Checkpoint。

## 总结

针对 DCP 里的 save & load 可以总结出以下的流程：

!\[DCP Flowchart]\(/images/dcp-flowchart.svg null)

为了更直观地理解 DCP 的 save 和 load 过程，我们可以将其分为以下几个层级和步骤：

- 保存流程 (Save Flow)

1. **\[用户空间]** `Stateful Object` (如 FSDP/DTensor) 调用 `state_dict()` 生成状态字典。
2. **\[API 层]** 触发保存操作，调用 `dcp.save` 或 `dcp.async_save`。
3. **\[Planner 层]**
   - **Local Planner**: `DefaultSavePlanner` 为当前 Rank 生成本地的写计划（`WriteItem`）。
   - **Global Planner**: Coordinator (Rank 0) 收集所有本地计划，进行去重和聚合，生成全局写计划 `Global SavePlan`。
4. **\[Storage 层]**
   - *(可选)* 如果是异步保存，会经过 `BlockingAsyncStager` 将数据卸载到 CPU 缓存。
   - `FileSystemWriter` 执行 `write_data`，将 Tensor 或 Bytes 数据向磁盘输出。
   - Coordinator 执行 `finish`，准备写入元数据对象。
5. **\[文件系统]**
   - 实际数据作为 Payload 写入具体的分布式分片文件（如 `__0_0.distcp` 等）。
   - 最终将全局结构和索引写入 `.metadata` 文件。

- 加载流程 (Load Flow)

1. **\[API 层]** 触发加载操作，调用 `dcp.load`。
2. **\[Storage 层]** `FileSystemReader` 调用 `read_metadata` 从文件系统读取 `.metadata` 索引文件，反序列化得到全局的 Metadata 对象。
3. **\[Planner 层]**
   - **Local Planner**: `DefaultLoadPlanner` 根据全局 Metadata 和用户当前传入的 `state_dict` 结构，生成本地的精确读计划（`ReadItem`）。
   - **Global Planner**: 生成最终的全局读计划，协调各个 Rank 的读取策略。
4. **\[Storage 层]** `FileSystemReader` 调用 `read_data`，根据读计划从文件系统中的各个分片文件 (`__0_0.distcp` 等) 中执行 Narrow（切片）和 Copy 操作，精确获取所需的数据块。
5. **\[用户空间]** 读取到的数据恢复到内存中的 `State Dict`，最后通过调用 `load_state_dict()` 注入回原来的 `Stateful Object` 中。

# DCP 实操

在实际应用中，当我们使用分布式训练（如 FSDP 或 Megatron）保存了 DCP 格式的检查点后，由于它被分散成多个文件（分片），在进行模型推理、导出 ONNX 或上传至 HuggingFace 时，往往需要将这些分片合并为一个完整的单一权重文件。

PyTorch 从 2.0+ 版本开始，提供了一个非常方便的工具函数：`torch.distributed.checkpoint.format_utils.dcp_to_torch_save`，它可以将分布式保存的 DCP 格式无缝转换为大家熟悉的 `torch.save` 格式（如 `.pt` 或 `.bin`）。结合 `safetensors` 库，我们也很容易将其转换为 `.safetensors` 格式。

下面是一个完整的实操脚本，展示了如何将一个目录下的 DCP 文件转换为合并后的 `torch.load` 格式以及 `safetensors` 格式。

为了演示整个端到端的流程，我们需要先生成一个 DCP 格式的 Checkpoint。下面是一个使用 FSDP (Fully Sharded Data Parallel) 训练并保存模型的完整示例。

1. 训练并保存 (FSDP + DCP)

将以下代码保存为 `train.py`：

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def setup():
    # 初始化进程组
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

def cleanup():
    dist.destroy_process_group()

def train_and_save():
    setup()
    rank = dist.get_rank()
    
    # 1. 定义模型
    # 简单的两层全连接网络
    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    ).to(rank)

    # 2. 使用 FSDP 包装模型
    # FSDP 会自动将参数分片到各个 GPU 上
    model = FSDP(model)
    
    # 3. 模拟训练
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 模拟一次前向反向传播，确保有梯度和优化器状态
    input_data = torch.randn(32, 1024).to(rank)
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    if rank == 0:
        print("[*] 训练完成，准备保存 Checkpoint...")

    # 4. 使用 DCP 保存分布式检查点
    # FSDP 能够直接提供 state_dict，DCP 会根据 FSDP 的分片情况自动规划保存
    checkpoint_dir = "checkpoints/fsdp_model"
    
    # 构建要保存的状态字典
    # 注意：这里直接传入 FSDP model，DCP 会调用其 state_dict() 方法
    # FSDP 的 state_dict() 返回的是 ShardedTensor，DCP 能够原生处理
    state_dict = {
        "model": model,
        "optimizer": optimizer
    }
    
    dcp.save(
        state_dict=state_dict,
        checkpoint_id=checkpoint_dir
    )
    
    if rank == 0:
        print(f"[+] Checkpoint 已保存至: {checkpoint_dir}")
        print(f"    包含文件: {os.listdir(checkpoint_dir)}")

    cleanup()

if __name__ == "__main__":
    # 使用 torchrun 启动: 
    # torchrun --nproc_per_node=2 train.py
    train_and_save()
```

1. 转换脚本 (DCP -> PyTorch/Safetensors)

生成了 DCP 检查点后，我们可以使用下面的脚本将其转换为单文件格式。

将以下代码保存为 `convert_dcp.py`：

```python
import os
import argparse
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from safetensors.torch import save_file

def convert_dcp_to_standard(dcp_dir: str, output_dir: str, save_safetensors: bool = True):
    """
    将分布式 DCP Checkpoint 转换为标准的合并格式
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[*] 开始读取 DCP Checkpoint: {dcp_dir}")
    
    # 1. 转换为 torch.save 格式 (.pt)
    # dcp_to_torch_save 会读取目录下的 .metadata 和分片文件，合并并在内存中生成一个完整的字典
    # 然后再将其写入到指定的文件中
    pt_output_path = os.path.join(output_dir, "model_consolidated.pt")
    
    print(f"[*] 正在合并分片并保存为: {pt_output_path}")
    # 注意：这个函数会在单个节点上加载所有分片，如果模型非常大，需要确保机器有足够的内存(RAM)
    dcp_to_torch_save(dcp_dir, pt_output_path)
    print("[+] 转换 .pt 成功！")

    # 2. (可选) 转换为 safetensors 格式
    if save_safetensors:
        safetensors_output_path = os.path.join(output_dir, "model.safetensors")
        print(f"[*] 正在转换为 safetensors 格式: {safetensors_output_path}")
        
        # 读取刚刚合并出来的单文件模型参数
        # 注意：这里我们使用 map_location="cpu" 确保不会意外占用显存
        state_dict = torch.load(pt_output_path, map_location="cpu")
        
        # 通常 DCP 保存的 state_dict 可能包含 'model', 'optimizer' 等外层 key
        # 如果是用于推理/发布的权重，通常我们只需要模型参数
        model_state_dict = state_dict.get("model", state_dict) 
        
        # 过滤掉非 Tensor 对象（safetensors 只能保存 Tensor）
        # 并且将所有的 Tensor 确保在 CPU 上并转为 contiguous
        clean_state_dict = {}
        for k, v in model_state_dict.items():
            if isinstance(v, torch.Tensor):
                clean_state_dict[k] = v.cpu().contiguous()
                
        # 保存为 safetensors
        save_file(clean_state_dict, safetensors_output_path)
        print("[+] 转换 .safetensors 成功！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCP to Standard Checkpoint Converter")
    parser.add_argument("--dcp_dir", type=str, required=True, help="包含 .metadata 和 .distcp 分片的原始 DCP 目录")
    parser.add_argument("--output_dir", type=str, required=True, help="转换后文件的输出目录")
    parser.add_argument("--no_safetensors", action="store_true", help="如果指定，则不生成 safetensors 文件")
    
    args = parser.parse_args()
    
    convert_dcp_to_standard(
        dcp_dir=args.dcp_dir, 
        output_dir=args.output_dir,
        save_safetensors=not args.no_safetensors
    )
```

### 使用方法

步骤 1：启动分布式训练并生成 DCP Checkpoint

你需要在一个有多张 GPU 的机器上运行（或者使用 `torchrun --nproc_per_node=1` 模拟单卡）：

```bash
# 使用 2 张 GPU 运行 FSDP 训练
torchrun --nproc_per_node=2 train.py
```

运行成功后，你会在当前目录下看到 `checkpoints/fsdp_model/` 文件夹，其中包含 `.metadata` 和多个 `.distcp` 分片文件。

步骤 2：转换 Checkpoint

现在我们将分散的检查点合并并转换格式：

```bash
python convert_dcp.py \
    --dcp_dir checkpoints/fsdp_model \
    --output_dir output/consolidated_model
```

执行完毕后，你将在 `output/consolidated_model` 目录下得到两个文件：

1. `model_consolidated.pt`: 这是一个标准的合并后的 PyTorch 字典，可以用 `torch.load()` 直接加载，适合继续单卡训练或恢复状态。
2. `model.safetensors`: 这是由 HuggingFace 推出的一种安全、零拷贝的张量存储格式，广泛用于 LLM 模型的发布和推理（如 vLLM，TGI 等）。

### 原理总结

为什么可以这么转？回顾我们前面介绍的 DCP 原理：

1. **Metadata 是寻址表**：`dcp_to_torch_save` 函数底层会首先读取 `.metadata` 文件。
2. **重组 (Consolidation)**：它会根据 `TensorStorageMetadata` 中的 `size` 在 CPU 内存中分配一个完整的空 Tensor，然后遍历 `chunks`，去对应的 `.distcp` 分片文件中读取数据，并写入到该空 Tensor 的对应 `offsets` 位置。
3. **最终输出**：当所有 Chunk 拼装完毕后，就得到了一份等价于单卡模式下的 `state_dict`，最后直接调用 `torch.save` 落盘。

# 小结

本文介绍了 PyTorch Distributed Checkpoint (DCP) 的核心设计与使用方法。相比于传统的 `torch.save`，DCP 专为分布式环境（尤其是 FSDP 和 Megatron-LM）设计，通过 Metadata 实现了逻辑视图与物理存储的解耦，支持了强大的 Resharding 能力。

- **核心价值**：支持在不同 GPU 数量（World Size）之间保存和加载模型，解决了大规模分布式训练中的扩缩容痛点。
- **关键组件**：
  - `Stateful`：定义需要保存的对象（如 Model, Optimizer）。
  - `Planner`：规划全局与本地的读写策略，处理分片逻辑。
  - `Storage`：执行实际的 I/O 操作，默认支持文件系统。
- **实战应用**：我们演示了如何使用 FSDP 训练并保存 DCP，以及如何利用 `dcp_to_torch_save` 将分布式权重合并为单文件（.pt/.safetensors），以便于推理部署或发布到 HuggingFace。

DCP 已经成为 PyTorch 大模型中的 Checkpoint 标准。

# 参考

1. <https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html>
2. <https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html>
