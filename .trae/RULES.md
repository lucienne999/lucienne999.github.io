# Workspace Rules

## 临时任务缓存约定

- 当用户明确表示任务是“暂时任务”“先别部署”“先存草稿”“仅本地暂存”时，默认不要把产物放入会参与站点构建或部署的目录，例如 `posts/`、`public/`、`src/`。
- 这类内容统一存放到 `.cache/<project_name>/`。
- `<project_name>` 使用清晰、稳定、便于识别的短名，优先采用 kebab-case，例如：
  - `.cache/kota-kinabalu/`
  - `.cache/japan-itinerary/`
  - `.cache/temporary-share-page/`

## 缓存目录使用规则

- 同一个临时任务的 markdown、HTML、素材说明等文件尽量放在同一个 `.cache/<project_name>/` 目录下。
- `.cache/` 目录默认视为本地暂存区，不应作为正式内容源。
- 除非用户明确要求“正式发布”“加入站点”“放进 lifes/posts/public”，否则不要把 `.cache/` 中的文件迁移到正式目录。

## 正式发布规则

- 只有在用户明确要求发布、展示到站点或参与部署时，才将内容从 `.cache/<project_name>/` 迁移到正式目录。
- 迁移前应保留原始缓存内容，避免覆盖用户仍在修改的临时稿。
