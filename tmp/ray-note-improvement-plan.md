# Ray 笔记改进计划

目标：把 `posts/notes/ray.md` 从 Ray API 使用笔记收敛成一篇面向 AI Infra / 具身数据工程师的生产设计复盘。主线是：

```text
规模与存储约束
  -> 控制面 / 数据面路径
  -> metadata-only 调度
  -> bucket、并发与背压
  -> 失败隔离、恢复与终止
  -> 适用边界
```

## 第一阶段：最优先优化

- [x] 压缩 Task / Actor 基础教程，保留后文依赖的执行语义，减少 README 级解释。
- [x] 补一节统一的数据路径：Driver、GCS、raylet、worker heap、Object Store、HDFS / HPStore。
- [x] 解释 Object Store 压力的来源，区分输入数据、Task 返回值和 `ObjectRef` 元数据。
- [x] 修正 heartbeat 表述：节点存活与资源上报不等于业务任务有进展。
- [x] 给出 `task_count`、`max_inflight`、batch size 的容量约束和调参顺序。
- [x] 解释 Task 内同步等待 Writer 的取舍：写入确认语义 vs sink latency 耦合。
- [ ] 检查最终章节衔接、术语一致性和 Markdown / lint。

## 第二阶段：吸收 RDP 生产经验

- [ ] 增加三类失败语义：单条不可重试、bucket 可重试、Actor crash / OOM。
- [ ] 展开“Ray Data 作为调度器而非大对象数据管道”的收益与代价。
- [ ] 增加确定性分桶、bucket-level resume、sink 幂等和重复写边界。
- [ ] 增加 `finish_threshold`、errored blocks、idle timeout 的多层终止策略。
- [ ] 增加 pack 两阶段 shuffle，解释 episode correlation、混合质量、内存和 I/O 代价。
- [ ] 说明 legacy Redis queue 与 Ray Data 双调度路径为何并存，以及迁移边界。

## 第三阶段：部署与可观测性

- [ ] 补 `runtime_env` 打包上限、exclude 和 external index 远程拉取。
- [ ] 补 Detached Actor 聚合 job-level metrics 的模式。
- [ ] 补 StreamingExecutor 预取与 driver 消费速度不一致的问题。
- [ ] 给 demo 增加串行 / Ray 实测数据，拆分 CPU、存储、sink 和调度瓶颈。

## 暂不扩展

- Placement Group、Ray Train、Ray Serve 属于 Ray 通用能力，但不是当前“数据处理”主线。除非文章扩展为完整 Ray 架构笔记，否则优先级低于 RDP 已验证的失败恢复、resume、shuffle 和数据局部性。

