"""
eval_qdrant_cases.py — Qdrant 检索测试用例执行器

读取《Qdrant检索测试用例.docx》逐条执行检索，按维度统计指标，
结果报告写回测试用例文件夹（docx）。

评估维度:
    Top-1 / Top-3 / Top-5 命中率   期望文件出现在前 k 条的比例
    MRR                            第一个正确命中的倒数排名均值
    要点覆盖率                     期望内容要点在命中 chunk 文本中的出现比例
    通过率                         Top-5 命中 且 要点覆盖率 ≥ 0.5

运行:
    python eval_qdrant_cases.py
    python eval_qdrant_cases.py --top-k 5 --threshold 0.5
"""

import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
del _sys


import os
import sys
import time
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_line.rag_search import RAGSearcher
from eval.eval_case_utils import (
    load_cases, point_coverage, file_stem, write_report,
)

CASES_DOCX = "/Users/trinxai/Documents/pythonfile/测试用例/Qdrant检索测试用例.docx"
REPORT_DOCX = "/Users/trinxai/Documents/pythonfile/测试用例/Qdrant检索测试结果.docx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--collection", default="news_collection",
                    help="目标 collection（如 CEO-files）")
    ap.add_argument("--coverage-threshold", type=float, default=0.5)
    ap.add_argument("--rewrite", action="store_true",
                    help="检索前先做口语→书面语查询改写（HyDE 同义扩展）")
    ap.add_argument("--no-rerank", action="store_true",
                    help="关闭 reranker 重排与相关性闸门（对照用）")
    ap.add_argument("--cases", default=CASES_DOCX)
    ap.add_argument("--out", default=None,
                    help="报告输出路径（默认：用例同目录的 <collection>检索测试结果.docx）")
    args = ap.parse_args()

    out_path = args.out
    if not out_path:
        out_path = os.path.join(os.path.dirname(args.cases),
                                f"{args.collection}检索测试结果.docx")

    cases = load_cases(args.cases)
    print(f"加载 {len(cases)} 条用例，开始检索评测 (collection={args.collection}, "
          f"top_k={args.top_k}, rewrite={args.rewrite}) ...")

    searcher = RAGSearcher(collection=args.collection,
                           use_rerank=not args.no_rerank)
    rows = []
    hits = {1: 0, 3: 0}
    rr_sum = 0.0
    cov_sum = 0.0
    passed = 0
    per_file = {}
    irr = {"n": 0, "pass": 0}
    relevant_n = 0

    for case in cases:
        t0 = time.time()
        results = searcher.search(case["query"], top_k=args.top_k,
                                  use_rewrite=args.rewrite)
        elapsed = time.time() - t0

        irrelevant = case["expect_file"].startswith("（无关")
        if irrelevant:
            # 无关问题的通过标准：相关性闸门把结果整单拒掉（一条都不返回）。
            # reranker 的分是真实相关度，能拦住；--no-rerank 对照模式下
            # 退回 RRF 分数判据（0.7），那个分数按排名给、区分度差
            top1 = results[0].get("rerank_score", results[0]["score"]) \
                if results else 0.0
            if args.no_rerank:
                ok = (not results) or top1 < 0.7
            else:
                ok = not results
            passed += ok
            irr["n"] += 1
            irr["pass"] += ok
            rows.append([
                case["id"], case["query"][:30], "-", "-",
                f"{top1:.3f}" if results else "已拒答",
                "✅" if ok else "❌",
            ])
            print(f"  [{case['id']:>2}] 无关 top1={top1:.3f} "
                  f"{'PASS' if ok else 'FAIL'} ({elapsed:.1f}s) {case['query'][:24]}")
            continue

        expect_stem = file_stem(case["expect_file"])
        rank = None
        hit_texts = []
        for i, r in enumerate(results, 1):
            if file_stem(r.get("source_file", "")) == expect_stem:
                if rank is None:
                    rank = i
                # 同一文件在 top-k 里可能命中多个 chunk，答案可能在任何一块里
                hit_texts.append(r.get("text", ""))

        # 要点覆盖：在该文件所有命中 chunk 的合集上算；
        # 文件没命中则在全部结果上算（兜底，通常为 0）
        eval_text = "\n".join(hit_texts) if hit_texts else \
            "\n".join(r.get("text", "") for r in results)
        cov, hit_n, total_n = point_coverage(case["expect_points"], eval_text)

        relevant_n += 1
        if rank:
            for k in hits:
                if rank <= k:
                    hits[k] += 1
            rr_sum += 1.0 / rank
        cov_sum += cov
        ok = bool(rank and cov >= args.coverage_threshold)
        passed += ok

        fname = file_stem(case["expect_file"])[:18]
        st = per_file.setdefault(fname, {"n": 0, "pass": 0})
        st["n"] += 1
        st["pass"] += ok

        actual = results[0]["source_file"] if results else "(无结果)"
        top1_score = results[0].get("rerank_score", results[0]["score"]) \
            if results else None
        rows.append([
            case["id"], case["query"][:30],
            f"#{rank}" if rank else "未命中",
            f"{hit_n}/{total_n}",
            f"{top1_score:.3f}" if top1_score is not None else "-",
            "✅" if ok else "❌",
        ])
        print(f"  [{case['id']:>2}] rank={rank or '-':>2} cov={cov:.2f} "
              f"{'PASS' if ok else 'FAIL'} ({elapsed:.1f}s) {case['query'][:24]}")

    n = len(cases)
    rn = max(relevant_n, 1)
    mrr = rr_sum / rn
    avg_cov = cov_sum / rn
    metrics = [
        ("用例总数", f"{n}（相关 {relevant_n} + 无关 {irr['n']}）"),
        ("Top-1 命中率（相关题）", f"{hits[1]}/{relevant_n} = {hits[1]/rn:.0%}"),
        ("Top-3 命中率（相关题）", f"{hits[3]}/{relevant_n} = {hits[3]/rn:.0%}"),
        ("MRR（平均倒数排名）", f"{mrr:.3f}"),
        ("平均要点覆盖率（相关题）", f"{avg_cov:.0%}"),
        ("相关题通过率（命中 且 覆盖率≥"
         f"{args.coverage_threshold}）", f"{passed - irr['pass']}/{relevant_n} = {(passed - irr['pass'])/rn:.0%}"),
        ("无关题边界守住率（闸门拒答）",
         f"{irr['pass']}/{irr['n']} = {irr['pass']/max(irr['n'],1):.0%}"),
        ("重排与闸门", "关闭（对照）" if args.no_rerank else "开启"),
    ]
    for f, st in per_file.items():
        metrics.append((f"  └ {f}", f"{st['pass']}/{st['n']} 通过"))

    conclusion_lines = [
        f"本次共执行 {n} 条检索用例（相关 {relevant_n} + 无关 {irr['n']}），"
        f"相关题通过 {passed - irr['pass']} 条（{(passed - irr['pass'])/rn:.0%}），"
        f"无关题守住边界 {irr['pass']}/{irr['n']}。",
        f"相关题 Top-3 命中率 {hits[3]/rn:.0%}，MRR {mrr:.3f}，"
        f"平均要点覆盖率 {avg_cov:.0%}。",
    ]
    if args.no_rerank:
        conclusion_lines.append(
            "注意：本轮为 --no-rerank 对照组，RRF 分数按排名计算，"
            "相关与无关查询分数区间重叠，无关题判据仅供参考。"
        )
    fails = [r for r in rows if r[5] == "❌"]
    if fails:
        conclusion_lines.append(
            "未通过用例编号：" + "、".join(r[0] for r in fails)
            + "。建议检查：对应 chunk 是否被切断关键数字、查询与文档用词差异是否过大。"
        )
    else:
        conclusion_lines.append("全部通过，检索质量符合预期。")

    write_report(
        out_path,
        f"Qdrant 检索测试报告（{args.collection}）",
        f"执行时间：{datetime.datetime.now():%Y-%m-%d %H:%M} ｜ top_k={args.top_k} ｜ "
        f"用例来源：{os.path.basename(args.cases)}",
        metrics,
        ["编号", "测试查询", "命中排名", "要点命中", "Top1得分", "结论"],
        rows,
        "\n".join(conclusion_lines),
    )
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
