"""闸门阈值标定：60 条抽样（20 精确 + 30 模糊 + 10 无关）× 阈值 {0.4, 0.45, 0.5}

阶段 1：每条用例检索一次（闸门=0），候选的重排分 + 要点覆盖所需文本全部落盘
阶段 2：离线套三个阈值，对比相关题通过率 / 无关题守住率
"""
import sys, os, json, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.eval_case_utils import load_cases, point_coverage, file_stem
from qdrant_line.rag_search import RAGSearcher

from config import TEST_CASE_DIR
CASES = os.path.join(TEST_CASE_DIR, "CEO-files检索测试用例300条.docx")
CACHE = "/tmp/calib60.json"
THRESHOLDS = (0.4, 0.45, 0.5)

cases = load_cases(CASES)
relevant = [c for c in cases if '无关' not in c['expect_file']]
irr_all = [c for c in cases if '无关' in c['expect_file']]
precise, fuzzy = relevant[:75], relevant[75:]
random.seed(42)
sample = random.sample(precise, 20) + random.sample(fuzzy, 30) + random.sample(irr_all, 10)

# ── 阶段 1：检索（有缓存就跳过）──
if os.path.exists(CACHE):
    data = json.load(open(CACHE))
    print(f"载入缓存 {len(data)} 条")
else:
    s = RAGSearcher(collection='CEO-files', rerank_threshold=0.0)
    data = []
    for i, c in enumerate(sample, 1):
        rs = s.search(c['query'], top_k=12)
        data.append({
            "id": c["id"], "query": c["query"],
            "irr": '无关' in c['expect_file'],
            "expect_stem": file_stem(c["expect_file"]) if '无关' not in c['expect_file'] else None,
            "expect_points": c["expect_points"],
            "cands": [{"score": r["rerank_score"],
                       "stem": file_stem(r["source_file"]),
                       "text": r["text"]} for r in rs],
        })
        print(f"  检索 {i}/60", flush=True)
    json.dump(data, open(CACHE, "w"))

# ── 阶段 2：离线套阈值 ──
print(f"\n{'阈值':>6} | {'相关通过':>10} | {'无关守住':>10}")
print("-" * 36)
for T in THRESHOLDS:
    rel_pass = rel_n = irr_hold = irr_n = 0
    for d in data:
        gated = [c for c in d["cands"] if c["score"] >= T][:3]
        if d["irr"]:
            irr_n += 1
            irr_hold += (len(gated) == 0)
        else:
            rel_n += 1
            hit_texts = [c["text"] for c in gated if c["stem"] == d["expect_stem"]]
            if hit_texts:
                cov = point_coverage(d["expect_points"], "\n".join(hit_texts))[0]
                rel_pass += (cov >= 0.5)
    print(f"{T:>6} | {rel_pass}/{rel_n} = {rel_pass/rel_n:.0%}   | "
          f"{irr_hold}/{irr_n} = {irr_hold/irr_n:.0%}")

# 参考：分数分布间隙
irr_max = max((c["score"] for d in data if d["irr"] for c in d["cands"]), default=0)
rel_min_top = min((max(c["score"] for c in d["cands"]) for d in data
                   if not d["irr"] and any(c["stem"] == d["expect_stem"] for c in d["cands"])),
                  default=0)
print(f"\n无关候选最高分: {irr_max:.4f}")
print(f"相关题（正确文件在候选中）的最低 top 分: {rel_min_top:.4f}")
