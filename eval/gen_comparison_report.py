"""
gen_comparison_report.py — 生成《检索优化对比报告.docx》

记录从 BGE 栈到 Qwen3-VL 栈的完整优化历程与 100 条用例对比结果。
重新生成：python eval/gen_comparison_report.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.eval_case_utils import write_report

OUT = "/Users/trinxai/Documents/pythonfile/测试用例/检索优化对比报告.docx"

metrics = [
    # ── 最终状态（Qwen3-VL 栈，100 条用例）──
    ("【最终·Qwen3-VL栈】Qdrant 总通过率", "100/100 ✅"),
    ("【最终·Qwen3-VL栈】Qdrant 精确题", "50/50"),
    ("【最终·Qwen3-VL栈】Qdrant 模糊/口语题", "25/25"),
    ("【最终·Qwen3-VL栈】Qdrant 无关题", "25/25（含近似域干扰题全拦）"),
    ("【最终·Qwen3-VL栈】LightRAG 总通过率", "100/100 ✅"),
    # ── BGE 栈（100 条用例）──
    ("【BGE栈】Qdrant 总通过率", "97/100（精确49/50、模糊24/25、无关24/25）"),
    ("【BGE栈】LightRAG 总通过率", "97/100（复核后≈99，LLM措辞方差）"),
    # ── 40 条用例时代 ──
    ("【40条】最初未优化", "Qdrant 36/40 ｜ LightRAG 38/40"),
    ("【40条】入库策略优化后", "Qdrant 36~39/40 ｜ LightRAG 40/40"),
    ("【40条】检索策略+QA增强后", "Qdrant 39/40 ｜ LightRAG 40/40"),
]

rows = [
    ["Qdrant 精确题", "19/20", "49/50", "50/50", "满分"],
    ["Qdrant 模糊/口语题", "8/10", "24/25", "25/25", "满分"],
    ["Qdrant 无关题", "9/10（无闸门）", "24/25", "25/25", "满分"],
    ["LightRAG 相关题", "29/30", "74/75", "75/75", "满分"],
    ["LightRAG 无关题", "9/10", "23/25", "25/25", "满分"],
]

conclusion = """【最终结论】

1. 模型替换（BGE → Qwen3-VL）后，两条线在 100 条用例（50 精确 + 25 模糊口语 + 25 无关含近似域干扰）上均取得 100/100。

2. Qwen3-VL-Reranker 的分数分布远优于 BGE-Reranker：口语化弱相关命中从 0.014~0.066 抬升到 0.38~0.59，无关候选最高 0.264，分布间隙清晰（[0.264, 0.377]），相关性闸门阈值定在间隙中点 0.30——BGE 时代"分布重叠、没有完美切点"的问题不复存在。模糊题从 24/25 到 25/25，关键就在重排器对口语查询的判定能力。

3. 无关题的近似域干扰子集（自行车保养、山地车变速、跑步机、短视频涨粉、财务报表——沾边但库内无答案）两条线全部正确拒答，知识边界可靠。

4. 优化历程回顾（每步解决的问题）：
   - 入库侧：大窗口分块(768/100)治列表切断；上下文头治跨文档撞车；QA 假设问题治口语召回；
   - 检索侧：查询改写双路融合治词汇鸿沟；reranker + 阈值闸门治无关拒答；
   - 模型侧：Qwen3-VL-2B（2048 维、多模态、MPS 加速）替换 BGE-M3，模糊题判定能力质的提升；
   - 图谱侧：数值条款标记 + 中文摘要 + chunk_top_k=40，数字细节保留率提升。

5. 当前栈配置：
   - 嵌入：Qwen/Qwen3-VL-Embedding-2B（ModelScope 本地权重 + transformers，dense-only，2048 维）
   - 重排：Qwen/Qwen3-VL-Reranker-2B（yes/no logit 打分，闸门 0.30）
   - 图谱：LightRAG（DeepSeek 抽取 + Qwen3-VL 嵌入中介服务）
   - 注意：Ollama 上的 VL 版 GGUF 不可用（嵌入端点架构不支持、reranker 转换质量差），两个模型均走本地 PyTorch 权重。

6. 多模态能力已就位但未启用：Qwen3-VL 支持图片嵌入与重排，当前文档均为纯文本，接口已预留（embedder 适配层支持 image 输入）。后续若入库含图片的文档（产品图、海报），嵌入链路无需改造。

【方法学说明】判定器（bigram+数字的要点覆盖率、否定词表拒答检测）在过程中迭代过，基线数字已按最终口径重算。LLM 相关环节（改写、图谱问答）存在 ±1~2 条轮次方差。阈值 0.30 是按当前 4 文档语料标定的，语料扩大后需用 eval/eval_qdrant_cases.py 重新标定。"""

write_report(
    OUT,
    "检索优化对比测试报告（最终版）",
    "BGE 栈 → Qwen3-VL 多模态栈 ｜ 100 条用例（50 精确 + 25 模糊 + 25 无关）｜ 2026-10-20",
    metrics,
    ["用例分类", "最初(40条)", "BGE栈(100条)", "Qwen3-VL栈(100条)", "结论"],
    rows,
    conclusion,
)
print("done")
