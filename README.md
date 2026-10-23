# rag_system 脚本功能说明

企业知识库双检索系统：Qdrant 向量检索线 + LightRAG 知识图谱线。

## 目录结构

```
rag_system/
├── core/           基础组件（解析 / 分块 / 嵌入 / 重排）
├── qdrant_line/    Qdrant 向量检索线（入库流水线 + 增量同步 + 检索）
├── kg_line/        LightRAG 知识图谱线（同步 + 查询）
├── augment/        查询增强（改写 / QA 假设问题）
├── eval/           用例评测（读 docx → 批量执行 → 生成报告）
└── tests/          回归测试
```

入口脚本均可直接运行，如：`python qdrant_line/run_sync.py`、`python kg_line/run_kg_sync.py`、
`python eval/eval_qdrant_cases.py`、`python -m pytest tests/`（或逐个 `python tests/test_xxx.py`）。

## core/ — 基础组件

| 文件 | 功能 |
|---|---|
| document_parser.py | 多格式文档解析（PDF/DOCX/PPT/TXT），`.doc` 老格式经 textutil/LibreOffice 自动转换，输出统一 ParsedDocument |
| text_chunker.py | 多策略文本分块（fixed/recursive/sentence/markdown/section），含小碎块合并与尾部 overlap 预告 |
| embedder.py | 嵌入模型封装：Qwen3-VL 多模态（当前默认，2048维）、FlagEmbedding/BGE-M3（dense+sparse）、Ollama、哈希降级，自动解析本地缓存 |
| reranker.py | 重排序器：Qwen3-VL-Reranker-2B（默认，yes/no logit 打分）或 BGE-Reranker-v2-m3，含相关性闸门阈值（0.30，按后端区分） |

## qdrant_line/ — Qdrant 向量线

| 文件 | 功能 |
|---|---|
| vectorize_pipeline.py | 向量化主流水线：解析→分块（上下文头/QA增强）→嵌入→入库 Qdrant；图片/表格资产落盘 kb-assets/ 并生成描述入库关联 |
| sync_index.py | 扫盘与差异比对纯函数工具库（同步编排与文件清单已下线，由外部扫描+DB脚本接管） |
| run_news.py | Qdrant 线批量入库/检索 CLI 入口 |
| kb_add.py | 单文档运维：新增入库（已入库自动跳过，查主库 payload 判断） |
| kb_update.py | 单文档运维：修改已入库文档（先删整份旧块再重入，防幽灵块） |
| kb_delete.py | 单文档运维：按路径/文件名/哈希删除整份文档的所有块（默认需 --yes 确认） |
| rag_search.py | 检索工具：查询改写 + dense/sparse 混合召回 + RRF 融合 + 重排 + 相关性闸门，top_k=3 |

## kg_line/ — LightRAG 图谱线

| 文件 | 功能 |
|---|---|
| kg_docs_client.py | LightRAG 文档管理 API 客户端（上传/文本直传/删除/状态轮询），纯标准库 |
| kg_manifest.py | 图谱线本地 JSON 清单（原子写入），记录路径→doc_id 映射 |
| kg_preprocess.py | 入库前预处理：标记关键数值条款、前置文档名头，提高 LLM 抽取的数字保留率 |
| kg_sync.py | 图谱增量同步：复用差异规划，本地解析→预处理→直传，异步抽取状态轮询 |
| run_kg_sync.py | 图谱线同步 CLI 入口（--file 部分同步/--recreate/--detect-renames） |
| lightrag_query.py | 图谱查询 CLI 工具（mix/local/global/naive/hybrid 模式，交互式） |
| serve_qwen3vl_embedding.py | 本机嵌入中介服务：把 Qwen3-VL-Embedding 包装成 OpenAI 兼容 /v1/embeddings 供 LightRAG 容器调用（MPS+串行锁） |

## augment/ — 查询增强

| 文件 | 功能 |
|---|---|
| query_rewriter.py | 口语查询→书面语改写（DeepSeek），fail-open + 进程内缓存 |
| qa_augment.py | 入库时为每 chunk 生成口语化假设问题，拼接向量化（QA 对增强索引） |
| asset_describer.py | 为提取的图片/表格资产生成检索用描述文字（DeepSeek），检索命中时返回资产文件路径 |

## eval/ — 用例评测

| 文件 | 功能 |
|---|---|
| eval_case_utils.py | 评测公共工具：读 docx 用例、要点覆盖率判定（bigram+数字）、生成报告 |
| eval_qdrant_cases.py | Qdrant 100 条用例批量评测：命中率/MRR/覆盖率/无关题闸门守住率 |
| eval_lightrag_cases.py | LightRAG 100 条用例批量评测：引用命中率/要点覆盖率/无关题拒答率 |

## tests/ — 回归测试

| 文件 | 功能 |
|---|---|
| test_sync_index.py | 增量同步差异规划回归测试（纯逻辑，秒跑） |
| test_text_chunker.py | 分块器回归测试（边界与 offset 精度） |
| test_kg_sync.py | 图谱同步回归测试（内存 FakeClient，无需真实服务） |
| test_embedder_scenarios.py | 嵌入器离线/在线两种加载模式验证 |
