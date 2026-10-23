---
name: kb-search
description: 千里达企业知识库检索。当需要查询产品开发流程、用户画像（M985/枭雄系列）、新品定价规则、经销商传播规范等企业制度内容时使用。提供两条互补通道：向量检索（快、免费、返回原文片段）和知识图谱问答（慢、走 LLM、返回综合答案）。
---

# 知识库检索 Skill

## 两条检索通道

### 1. 向量检索（首选，日常查询）

```bash
cd /Users/trinxai/pyprofiles/rag_system
/opt/anaconda3/bin/python3 qdrant_line/rag_search.py "你的问题"
```

- 返回 top-3 相关文档片段（含文件名、相关度分数、原文）
- 口语化提问也可以（内置查询改写）
- **与知识库无关的问题会返回空**（相关性闸门拒答），这是正常行为，直接告知用户"知识库中没有相关内容"
- 首次调用需加载模型，约 30 秒；之后每次 2-4 秒

### 2. 知识图谱问答（需要综合归纳/实体关系时用）

```bash
cd /Users/trinxai/pyprofiles/rag_system
/opt/anaconda3/bin/python3 kg_line/lightrag_query.py "你的问题"
```

- 返回 LLM 综合生成的答案 + 引用来源文件
- 适合：职责关系（"谁负责什么"）、流程归纳（"分几个阶段"）、跨文档综合
- 每条约 5-15 秒
- 查询模式可选：`--mode local`（实体聚焦）/ `global`（宏观主题）/ `mix`（默认综合）

## 如何选择

- 找具体条款、数字、原文 → 通道 1
- 问"谁负责/什么关系/哪些阶段/对比" → 通道 2
- 拿不准就两个都跑，交叉验证

## 前置条件（异常时排查）

三个服务必须在线：
- Qdrant：`curl -s localhost:6333/collections`（Docker 容器 `qdrant`）
- LightRAG：`curl -s localhost:9621/health`（Docker 容器 `lightrag-lightrag-1`）
- 嵌入中介：`curl -s localhost:18001/health`（launchd 服务 com.trinxai.qwen3vl-embedding，开机自启）

## 知识库维护（仅在用户明确要求时执行）

文档变更后同步：
```bash
# 向量库增量同步（先 dry-run 看差异）
/opt/anaconda3/bin/python3 qdrant_line/run_sync.py --dry-run
/opt/anaconda3/bin/python3 qdrant_line/run_sync.py

# 图谱增量同步（LLM 抽取，有 token 成本，先 dry-run）
/opt/anaconda3/bin/python3 kg_line/run_kg_sync.py --dry-run
/opt/anaconda3/bin/python3 kg_line/run_kg_sync.py
```

**绝对不要**主动执行 `--recreate`（清空重建）或删除类操作，除非用户明确说出"重建/清空"。
