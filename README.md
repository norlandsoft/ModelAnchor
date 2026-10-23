# ModelAnchor

为 **DrawThings** 检索、下载与合并模型的本地 Web 服务。Python 技术栈（FastAPI + uvicorn），前端零构建。

当前能力：

- **两站检索**：HuggingFace 镜像（`hf-mirror.com`）与 ModelScope，统一结果，可按「单文件 / Diffusers 目录 / LoRA」过滤
- **两种下载**：单文件直下；diffusers 整目录按文件串行下载，进度按字节聚合
- **合并导出**：把 diffusers 目录（unet/transformer + vae + text_encoder…）按标准键前缀合成**单个 safetensors**
- **下载加速**：强制直连（不读任何代理环境变量）、同文件多源测速择优、大文件分段并行（16 MiB × 4 连接），实测比单连接快 3–9 倍
- **可靠性**：断点续传（Range + 分片元数据）、429/5xx 退避重试、磁盘空间预检、SHA256 校验、取消/重试
- **本地维护**：模型库扫描（文件与目录）、删除、磁盘余量

## 快速开始

```bash
cd /opt/Workspace/ModelAnchor

./scripts/start.sh               # 前台，Ctrl-C 停止
./scripts/start.sh --background  # 后台，日志在 .run/modelanchor.log
./scripts/start.sh --reload      # 开发模式，改代码自动重载
./scripts/stop.sh                # 停止后台实例
```

打开 <http://127.0.0.1:7990>，API 文档在 <http://127.0.0.1:7990/docs>。

首次启动会自动建 `.venv`（需要 [uv](https://docs.astral.sh/uv/)）并同步依赖。

## 模型目录

默认是项目内的 `models/`（不自动往 DrawThings 容器里塞东西，导入仍由你手动完成）。
要换位置就设环境变量：

```bash
MODELANCHOR_MODEL_DIR=/path/to/Models ./scripts/start.sh
```

目录里：

- 单文件下载 → `models/<文件名>`
- 整目录下载 → `models/<owner>_<repo>/`（保留仓库内的相对路径）
- 合并产物 → `models/<你填的名字>.safetensors`

下载先写 `<文件名>.part`，完成后原子重命名；中断后再次下载同一个仓库会接着续传。

## 在线检索

检索会同时打两个站点，任一站点失败不影响另一个（错误会显示在结果上方）。

### 布局判定

| 判定 | 依据 | 界面上的行为 |
| --- | --- | --- |
| `lora` | 站点标签含 `lora` / `base_model:adapter`，或仓库名带 lora，或**全部**权重文件名带 lora | 只列权重文件，逐个「下载」 |
| `diffusers` 目录 | 根目录有 `model_index.json`，或有 `unet/` / `transformer/` 目录 | 提供「下载整个目录」，下载完可合并 |
| 单文件 | 任意位置有 `.safetensors` / `.ckpt` / `.gguf`，且不是上面的结构 | 逐个权重文件提供「下载」 |
| 未知 | 没有权重文件（纯配置/受限仓库） | 只展示文件列表 |

ComfyUI 风格的仓库（权重放 `diffusion_models/`，旁边可能有 `vae/`）会被正确判为**单文件**，
不会被当成 diffusers 目录整包下载。

> 检索阶段 ModelScope 只探测根目录来判布局，列表里可能不显示总体积；点开「查看文件」才有精确值。

### LoRA 从哪里下载

LoRA 是「一个 `.safetensors` 适配器」而不是整套模型，所以走检索里的 **LoRA** 类型：

1. 搜索框输入 `lora`，或点快捷词：Flux LoRA / SDXL LoRA / Qwen-Image LoRA / Wan 2.2 LoRA；
2. 类型下拉选 **LoRA**，结果会只剩适配器仓库（标签或文件名里带 lora 的）；
3. 点「查看文件」→ 选具体版本 → 下载。文件落在 `models/<文件名>.safetensors`，也可以点「源站」看作者说明。

已知的 LoRA 集散地：

- **HuggingFace 镜像**：`black-forest-labs/FLUX.1-Depth-dev-lora`、`alexrzem/flux-loras`（几百个 Flux LoRA 合集）、`strangerzonehf/Flux-Ultimate-LoRA-Collection` 等；
- **ModelScope**：搜索 `lora` + 基座名，例如 `qwen-image lora`、`wan2.2 lora`；
- **Civitai**：LoRA 数量最多，但不提供免鉴权的检索/下载接口，本工具没有接入。可以拿到直链后用「从链接安装」下载（Civitai 的下载链接通常需要登录 Cookie 或 API Token）。

下载完在 DrawThings 里**手动导入**：Settings → Model → Manage → Import Model → Select from Files，
类型选 **LoRA**，并按需选择基座（参考[官方 Wiki](https://wiki.drawthings.ai/wiki/Install_a_Model_or_LoRA)）。

## 下载

- 单文件：直接 `POST /api/downloads`，可带 `sha256`，下完校验不过会删掉 `.part` 并报错。
- 整目录：`POST /api/downloads/repo`，服务端先拉文件列表再逐个下载，跳过 README/图片/`.gitattributes`/tests 等。
- 并发：默认同时 2 个任务（`MODELANCHOR_CONCURRENCY`），任务内部按文件串行。
- 重试：遇到 429/500/502/503/504 会按 2s、4s、8s 退避重试，仍失败则报「服务器限流或暂时不可用」。
- 取消：保留已下载的 `.part`，下次继续。

## 下载通道与速度

### 强制直连，不走代理

所有 HTTP 客户端都设置了 `trust_env=False`，并显式把连接池指向直连：

- `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` / `http_proxy` 等环境变量**不会生效**；
- 也不会读 `.netrc`、证书环境变量；
- 检索（HF 镜像 / ModelScope）与下载（含重定向后的 CDN）都走同一条规则。

需要走代理的场景本工具不支持——这是按需求刻意关掉的。

### 镜像超时怎么处理

`hf-mirror.com` 本身会慢：实测同一接口有时 3–5 秒，有时 40 秒以上。所以检索侧做了这些：

- 搜索请求单独用 **12 秒**超时（仓库详情用 `MODELANCHOR_HUB_TIMEOUT`，默认 30 秒）；
- 超时/连接错误自动重试 1 次；
- 检索结果缓存 **10 分钟**，同样的关键词第二次是毫秒级；
- 一个站点超时不影响另一个：ModelScope 的结果照常显示，错误栏写清原因并提供「重试」按钮；
- 失败信息会明确写「响应超时，镜像此刻可能不稳定」，而不是原始的 `The read operation timed out`。

### 中断与续传

任务列表存在内存里，进程重启就没了；但「这个文件名当初从哪个 URL 下的」会记在
`.run/downloads.json`，所以重启后仍能续：

- 本地模型库里，`.part` 行显示的是**真实进度**（读 `.part.json` 分片账本），不是文件大小；
  分片 `.part` 是稀疏文件，`ls -l` 看到的是总大小，别被它骗了；
- 有链接记录的 `.part` 直接点 **继续**；
- 没有记录（例如这个功能上线前就开始的任务）点继续会跳到安装表单，文件名已填好，
  粘贴一次原下载链接即可，提交后从断点续；
- 分片任务按块续，只补没下完的 16 MiB 分片；
- 没有分片元数据的老 `.part`：连续写满的按字节续，稀疏文件则重下（无法判断哪些块真写了，
  宁可重来也不产生坏文件）。

### 多源择优

同一个仓库/文件若在 HF 镜像和 ModelScope 都存在，会：

1. 先各拉 512 KiB 测速（2 秒上限）；
2. 只有备用源比首选**快 1.2 倍以上**才切换，避免测速噪声导致来回跳；
3. 下载过程中某个源持续失败，会自动换下一个源，并从已下载的字节继续。

实测（同一份 Qwen-Image VAE，242 MiB）：

| 通道 | 吞吐 |
| --- | --- |
| HF 镜像单连接 | 2.5 – 6.7 MB/s（字节最终来自境外 CDN） |
| ModelScope 单连接 | 8.5 MB/s |
| ModelScope 4 连接 | 13.2 MiB/s 聚合 |
| **ModelAnchor 实际下载** | **~23 MiB/s，242 MiB 用 15.4 秒** |

### 分段并行 + 自适应

- ≥ `MODELANCHOR_SEGMENT_MIN_MB`（默认 32 MiB）且服务器支持 `Range` 的文件才分段；
- 分片大小默认 16 MiB，连接数默认 4；
- 每个分片用 `pwrite` 落在 `.part` 文件的固定偏移，完成一片就在 `<文件名>.part.json` 里记账，中断后只补没下完的分片；
- `MODELANCHOR_SEGMENT_MODE=auto`（默认）会先测「单连接 vs 4 并发」的实际带宽：并行没有 1.15 倍收益就退回单连接（HF 的 CDN 有时并行反而更慢），`always` / `never` 可手动锁定；
- 目录任务（diffusers 仓库）里的小文件用 4 个 worker 并发，大文件各自分段，进度按文件累加。

代价：大文件开始下载前有 1–3 秒测速开销；分片任务会多一个 `.part.json` 小文件（完成后自动删除）。

## 合并导出

「合并导出」把 diffusers 目录合成一个标准 safetensors：

- 每个组件的张量按架构加前缀，例如 Flux：`transformer/*` → `model.diffusion_model.*`、`vae/*` → `first_stage_model.*`、`text_encoder/*` → `cond_stage_model.transformer.*`
- 支持分片组件（`model.safetensors.index.json` 的 `weight_map`）
- 直接重写 safetensors 头并原样搬运数据区，**不经过 torch/numpy**，所以 bf16/fp8 也不会被改精度
- 合并前会显示识别结果（组件、张量数、字节数），并检查磁盘空间

支持的架构预设：

| 值 | 键前缀映射 |
| --- | --- |
| `auto` | 按目录自动判断（unet → sd15/sdxl，transformer → flux） |
| `sd15` | `model.diffusion_model.` / `first_stage_model.` / `cond_stage_model.transformer.` |
| `sdxl` | 同上，文本编码器用 `conditioner.embedders.0.transformer.`、`conditioner.embedders.1.model.` |
| `flux` / `sd3` | `model.diffusion_model.` / `first_stage_model.` / `cond_stage_model.(2|3).transformer.` |
| `generic` | 用组件目录名当前缀（`unet.`、`vae.`…），给非标准结构兜底 |

### 为什么只有 .safetensors，没有 .ckpt

DrawThings 的 `.ckpt` 不是 safetensors 改扩展名，而是 **SQLite 数据库**——文件头就是
`SQLite format 3`，内部是一张 `tensors(name, type, format, datatype, dim, data)` 表（实测
`qwen_image_2.1_vae_f16.ckpt`：226 行、`user_version=327680`）。伪造它的 `type` 标志位和 `dim`
编码需要逐位对齐 DrawThings 的加载器，风险远大于收益，所以这里只导出标准 safetensors。

需要 `.ckpt` 时，请用 DrawThings 自带的转换/导入流程，或直接在检索里找官方已经打包好的 `.ckpt` 下载。

### 合并的限制

- 只处理 `.safetensors`；仓库若只有 `pytorch_model.bin`，会明确报错（读 pickle 需要 torch）
- 输出是**全量权重**：文本编码器也一起打包，所以体积会明显大于单个主模型
- 合并需要约等于产物大小的临时空间；产物写入 `<name>.safetensors.tmp` 后再重命名

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 存活、版本、端口、模型目录 |
| GET | `/api/library` | 本地模型/目录列表、占用与磁盘剩余 |
| GET | `/api/catalog` | DrawThings 官方模型目录（仅 API 保留，界面已不展示） |
| GET | `/api/search` | `q`、`source=all\|hf\|modelscope`、`layout=all\|single\|diffusers\|lora`、`limit` |
| GET | `/api/repos` | `source` + `repo_id`，返回完整文件列表、布局、每个文件的直链 |
| POST | `/api/downloads` | 单文件：`catalog_id` 或 `url`（可选 `filename`、`sha256`） |
| POST | `/api/downloads/repo` | 整目录：`source` + `repo_id` |
| GET | `/api/tasks` | 全部任务（下载 + 合并），含进度、当前文件 |
| POST | `/api/tasks/{id}/cancel` | 取消任务 |
| DELETE | `/api/tasks/{id}` | 移除已结束的任务记录 |
| GET | `/api/merges/plan` | `source_dir` + `arch`，预览组件与张量数 |
| POST | `/api/merges` | `source_dir`、`output_name`、`arch` |
| DELETE | `/api/library/{name}` | 删除文件或整个目录 |

## 配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODELANCHOR_HOST` | `127.0.0.1` | 监听地址（默认只对本机开放） |
| `MODELANCHOR_PORT` | `7990` | 监听端口 |
| `MODELANCHOR_MODEL_DIR` | `./models` | 模型目录 |
| `MODELANCHOR_HF_BASE` | `https://hf-mirror.com` | HuggingFace 源前缀；能直连就改成 `https://huggingface.co` |
| `MODELANCHOR_HUB_TIMEOUT` | `30` | 站点 API 超时（秒）；搜索固定用 12 秒 |
| `MODELANCHOR_CONCURRENCY` | `2` | 同时下载的任务数 |
| `MODELANCHOR_CONNECTIONS` | `4` | 单个文件的分片连接数 |
| `MODELANCHOR_STATE_DIR` | `.run` | 下载记录（`.run/downloads.json`）的存放目录 |
| `MODELANCHOR_CHUNK_MB` | `16` | 分片大小（MiB） |
| `MODELANCHOR_SEGMENT_MIN_MB` | `32` | 多大的文件才启用分段 |
| `MODELANCHOR_SEGMENT_MODE` | `auto` | `auto` 先测速再决定；`always` 强制分段；`never` 单连接 |
| `MODELANCHOR_ALLOW_ANY_URL` | `1` | 是否允许任意链接下载 |
| `MODELANCHOR_PYTHON` | `3.11` | 建 venv 用的 Python 版本 |

## 目录结构

```
ModelAnchor/
├── app/
│   ├── main.py            # 应用装配、页面路由、生命周期
│   ├── sources.py         # 跨站点备用源查找与直链解析
│   ├── api.py             # /api/* 路由
│   ├── downloader.py      # 下载引擎（多源测速、分段并行、续传、重试、取消）
│   ├── merger.py          # diffusers → 单文件：组件发现、架构前缀、任务
│   ├── safetensors_io.py  # safetensors 原始读写（无 torch 依赖）
│   ├── hubs/              # 两个站点的检索与文件列表
│   │   ├── hf.py
│   │   ├── modelscope.py
│   │   └── base.py        # 数据模型、布局判定、缓存
│   ├── library.py         # 本地库扫描、路径防逃逸
│   ├── catalog.py         # DrawThings 官方目录
│   └── config.py          # 环境变量
├── data/catalog.json      # 官方模型目录（仅 /api/catalog 使用，界面不展示）
├── web/                   # 零构建前端
├── scripts/               # start.sh / stop.sh
└── tests/                 # 45 个用例，全部离线
```

## 开发

```bash
uv sync --extra dev          # 装 ruff / pytest（start.sh 的 uv sync 会清掉它们，属正常）
.venv/bin/python -m pytest -q
.venv/bin/ruff check app tests
./scripts/start.sh --reload

# 检索结果可以用 URL 带上关键词直接打开
open "http://127.0.0.1:7990/?q=flux"
```

## 已知限制 / 下一步

- 整目录下载按文件串行，小文件多时每个文件都有一次重定向开销；可以加小文件并发
- ModelScope 上部分 `black-forest-labs/*` 仓库需要登录，会显示「没有读到文件列表」
- 任务状态在内存里，进程重启后不保留（`.part` 仍可续传）
- 「DrawThings 官方模型」面板已移除；对应的 `/api/catalog` 仍保留给脚本调用
- 超大 LoRA 合集（例如 499 个文件）目前只列前 200 条，还没有分页/仓库内搜索
- 分段下载依赖服务器支持 `Range`；不支持时自动退回单连接
- 开始下载前有 1–3 秒测速开销（大文件可忽略，小文件不走测速）
- Civitai 未接入（无免鉴权 API），需要直链 + Cookie/Token 时用「从链接安装」
- 合并产物没有做权重正确性校验（例如与官方 ckpt 对比），只保证结构与字节搬运正确
- 还没有做「一个模型 = 主模型 + VAE + 编码器」的成套安装

## License

Apache License 2.0，见 [LICENSE](LICENSE)。
