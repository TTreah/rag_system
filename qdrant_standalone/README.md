# qdrant_standalone — Qdrant 向量检索线独立部署包

从 `rag_system/` 中抽出的 Qdrant 线完整副本：文档解析 → Markdown 分块 → 嵌入 → Qdrant 入库 → 检索。
不包含 LightRAG 图谱线，也不依赖 Ollama。

## 一、需要部署的服务

| 端口 | 服务 | 是否必需 | 用途 |
|---|---|---|---|
| **6333** | Qdrant | ✅ **必需** | 向量库，所有数据存这里 |
| **8000** | MinerU API | ✅ 强烈建议 | PDF 解析/OCR（逐页文本层/扫描页自动切换）；不可用时自动回退 pymupdf，但扫描件会没有文本 |
| 11434 | Ollama | ❌ 不需要 | 代码已移除 Ollama/BGE 后端 |
| 1234 | — | ❌ 不需要 | 无依赖 |
| 18001 / 9621 / 9622 | 嵌入中介 / LightRAG | ❌ 不需要 | 属于图谱线，与本副本无关 |

外部 API：**DeepSeek**（查询改写、QA 增强、表格/图片描述）。设置环境变量：

```bash
export DEEPSEEK_API_KEY=sk-...
```

## 二、需要部署的模型

模型缓存在 `~/.cache/modelscope/models/`，可由 `core/embedder.py` 自动发现。

| 模型 | 大小 | 用途 | 何时必需 |
|---|---|---|---|
| `Qwen/Qwen3-VL-Embedding-2B` | 4.0 GB | 文本+图片向量化（2048 维） | ✅ 检索/入库必需 |
| `Qwen/Qwen3-VL-Reranker-2B` | 4.0 GB | 重排 + 相关性闸门 | ✅ 检索必需 |
| `Qwen/Qwen3-VL-2B-Instruct` | 4.0 GB | 生成图片内容描述 | 入库含图片时必需 |
| `OpenDataLab/PDF-Extract-Kit-1.0` | 2.5 GB | MinerU pipeline 后端（布局/OCR/表格） | 启用 MinerU 时必需 |
| `OpenDataLab/MinerU2.5-Pro-2605-1.2B` | 2.2 GB | MinerU VLM/hybrid 后端 | 使用 hybrid/vlm 后端时需要 |

合计约 **16.7 GB**。

```bash
pip install modelscope
python -c "from modelscope import snapshot_download as d; \
d('Qwen/Qwen3-VL-Embedding-2B'); d('Qwen/Qwen3-VL-Reranker-2B'); d('Qwen/Qwen3-VL-2B-Instruct')"
# MinerU 模型通常由 MinerU 首次运行自动下载；也可手动：
python -c "from modelscope import snapshot_download as d; \
d('OpenDataLab/PDF-Extract-Kit-1.0'); d('OpenDataLab/MinerU2.5-Pro-2605-1.2B')"
```

离线部署可直接拷贝整个 `~/.cache/modelscope/models/` 目录。

## 三、安装与启动

```bash
pip install -r requirements.txt

# Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# MinerU（可选但建议；独立 Python 环境）
#   参考 MinerU 官方文档安装后启动：
#   mineru-api --host 0.0.0.0 --port 8000
```

## 四、常用入口

所有命令在本目录下执行。

### 入库

```bash
# 批量入库（--local 跳过 Qdrant；--recreate 先删库重建）
python qdrant_line/run_news.py --recreate

# 单文档运维
python qdrant_line/kb_add.py /path/file.pdf --collection CEO-files
python qdrant_line/kb_update.py --name 文件名关键词 --collection CEO-files
python qdrant_line/kb_delete.py --name 文件名关键词 --collection CEO-files --yes
```

### 检索

```bash
python qdrant_line/rag_search.py "查询内容"
```

检索流程：查询改写（原查询+改写双路）→ 混合召回 top-10 → 去重 → reranker → 相关性闸门 → **top-3 输出**。

### 图片资产升级（入库含图片时）

```bash
# 图片本体向量 + Qwen3-VL-Instruct 真实描述（可断点续跑）
python qdrant_line/update_image_assets.py
# 为图片补描述文本点（细节检索）
python qdrant_line/add_caption_points.py
```

### 其他

```bash
# 补齐点 payload 中缺失的顶层字段
python qdrant_line/backfill_payload.py
# 查看差异规划工具（被 kb_* 与外部同步脚本复用）
#   qdrant_line/sync_index.py 提供 scan_disk / plan_sync / _file_hash
```

## 五、入库与分块策略

- **PDF**：MinerU 逐页判断文本层/扫描页，输出 Markdown；四类清洗：
  Unicode 归一化、HTML 表格转 Markdown、pymupdf 提取图片、过滤页眉页脚。
- **分块**：Markdown 按标题层级切分；`chunk_size=768`、`overlap=100`、`min_chunk_size=200`。
- **上下文头**：每个块前置 `【文件名｜章节标题】`。
- **资产**：表格转 Markdown 全文入向量并生成描述；图片存图片向量 + 描述文本点。
- **QA 增强**（可选 `qa_augment=True`）：每块生成 3 个假设问题参与向量化。

## 六、代码结构

```
core/           解析（含 MinerU）、分块、嵌入、重排、图片描述
augment/        查询改写、QA 增强、资产描述
qdrant_line/    入库流水线、检索、同步工具、kb_* 运维脚本
requirements.txt
```

## 七、注意事项

- **DeepSeek key**：优先读 `DEEPSEEK_API_KEY`；未设置时会读 `augment/query_rewriter.py` 中的 `LIGHTRAG_ENV`（默认指向本机 LightRAG 部署的 `.env`，换机器请设置环境变量或修改该常量）。
- **MinerU 不可达**时 PDF 自动回退原生解析，不会阻断入库。
- 首次运行会下载/加载模型，约需 30 秒；之后检索 2-4 秒/次。
- 本副本与 `rag_system/` 正本保持同步；修改时请两边一起改，或只维护正本后重新拷贝。
