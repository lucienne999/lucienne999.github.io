# Tasks
- [x] Task 1: 定义 Skill 目标与边界
  - [x] 明确该 Skill 只服务于技术文档评审，不扩展到代码审查或通用写作润色
  - [x] 明确输入文档范围与典型使用场景

- [x] Task 2: 设计 Skill 输出结构
  - [x] 定义“需要改进的地方”输出要求，要求按重要性排序
  - [x] 定义“做得好的地方”输出要求，要求具体且可定位
  - [x] 定义“详细改进方案”输出要求，要求给出可执行建议而非空泛评价

- [x] Task 3: 编写 Skill 文件
  - [x] 在 `.trae/skills/<skill-name>/SKILL.md` 中补充 frontmatter
  - [x] 编写 Skill 的详细说明、触发条件、评审原则和输出模板
  - [x] 提供至少一个技术文档评审示例

- [x] Task 4: 验证 Skill 规格与内容一致
  - [x] 检查 Skill 描述是否同时说明“做什么”和“何时调用”
  - [x] 检查评审输出是否同时覆盖批判、亮点和改进方案
  - [x] 检查语言风格是否保持严格且建设性

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 1, Task 2
- Task 4 depends on Task 3
