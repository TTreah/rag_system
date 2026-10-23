"""
eval_lightrag_cases.py — LightRAG 图谱问答测试用例执行器

读取《LightRAG图谱检索测试用例.docx》逐条查询，按维度统计指标，
结果报告写回测试用例文件夹（docx）。

评估维度:
    引用命中率    回答的引用来源包含期望文件（跨文档题要求 ≥2 个引用文件）
    要点覆盖率    期望答案要点在回答文本中的出现比例
    通过率        引用命中 且 要点覆盖率 ≥ 0.6
    平均耗时      每条问答的端到端耗时（含 LLM 生成）

运行（每条都要走一次 LLM，20 条约十几分钟，耐心等）:
    python eval_lightrag_cases.py
    python eval_lightrag_cases.py --mode local --threshold 0.6
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

from kg_line.lightrag_query import LightRAGClient
from eval.eval_case_utils import (
    load_cases, point_coverage, file_stem, write_report,
)

CASES_DOCX = "/Users/trinxai/Documents/pythonfile/测试用例/LightRAG图谱检索测试用例.docx"
REPORT_DOCX = "/Users/trinxai/Documents/pythonfile/测试用例/LightRAG图谱检索测试结果.docx"

CROSS_DOC_MARKER = "跨文档"
IRRELEVANT_MARKER = "（无关"

# 拒答信号：回答里出现这类短语才算"守住了边界"
# 拒答信号：回答里出现这类短语才算"守住了边界"。
# 取宽口径的否定词（无法/没有/抱歉）——LLM 拒答措辞千变万化，
# 枚举具体短语追不上（实测同一问题多次生成措辞都不同）
NEG_PHRASES = ["无法", "没有", "不包含", "不涉及", "不知道", "未见",
               "超出", "抱歉", "暂无", "并未"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mix",
                    choices=["mix", "local", "global", "naive", "hybrid"])
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--cases", default=CASES_DOCX)
    ap.add_argument("--out", default=REPORT_DOCX)
    args = ap.parse_args()

    cases = load_cases(args.cases)
    print(f"加载 {len(cases)} 条用例，开始图谱问答评测 (mode={args.mode}) ...")

    client = LightRAGClient()
    client.health()  # 前置连通检查

    rows = []
    ref_hit_n = 0
    cov_sum = 0.0
    passed = 0
    time_sum = 0.0
    irr = {"n": 0, "pass": 0}
    relevant_n = 0

    for case in cases:
        t0 = time.time()
        try:
            data = client.query(case["query"], mode=args.mode)
            answer = data.get("response", "")
            refs = [r.get("file_path", "") for r in (data.get("references") or [])]
        except Exception as e:
            answer, refs = f"(查询失败: {e})", []
        elapsed = time.time() - t0
        time_sum += elapsed

        # 引用判定：跨文档综合题要求引用 ≥2 个不同文件；其余要求包含期望文件
        irrelevant = case["expect_file"].startswith(IRRELEVANT_MARKER)
        if irrelevant:
            # 无关问题考的是拒答：回答必须明确表示"没有相关信息"，
            # 而不是硬凑一个看似合理的答案（幻觉防线）
            ok = any(p in answer for p in NEG_PHRASES)
            passed += ok
            irr["n"] += 1
            irr["pass"] += ok
            rows.append([
                case["id"], case["query"][:26], "-", "-",
                f"{elapsed:.0f}s",
                "✅" if ok else "❌",
            ])
            print(f"  [{case['id']:>2}] 无关 拒答={'Y' if ok else 'N'} "
                  f"{'PASS' if ok else 'FAIL'} ({elapsed:.0f}s) {case['query'][:22]}")
            continue

        relevant_n += 1
        if CROSS_DOC_MARKER in case["expect_file"]:
            distinct = {os.path.basename(r) for r in refs if r}
            ref_hit = len(distinct) >= 2
            ref_desc = f"{len(distinct)}个文件"
        else:
            expect_stem = file_stem(case["expect_file"])
            ref_hit = any(expect_stem in file_stem(r) for r in refs)
            ref_desc = os.path.basename(refs[0])[:18] if refs else "(无引用)"

        cov, hit_n, total_n = point_coverage(case["expect_points"], answer)
        cov_sum += cov
        ref_hit_n += ref_hit
        ok = bool(ref_hit and cov >= args.threshold)
        passed += ok

        rows.append([
            case["id"], case["query"][:26],
            "✓" if ref_hit else "✗",
            f"{hit_n}/{total_n}",
            f"{elapsed:.0f}s",
            "✅" if ok else "❌",
        ])
        print(f"  [{case['id']:>2}] ref={'Y' if ref_hit else 'N'} cov={cov:.2f} "
              f"{'PASS' if ok else 'FAIL'} ({elapsed:.0f}s) {case['query'][:22]}")

    n = len(cases)
    rn = max(relevant_n, 1)
    avg_cov = cov_sum / rn
    metrics = [
        ("用例总数", f"{n}（相关 {relevant_n} + 无关 {irr['n']}）"),
        ("查询模式", args.mode),
        ("引用命中率（相关题）", f"{ref_hit_n}/{relevant_n} = {ref_hit_n/rn:.0%}"),
        ("平均要点覆盖率（相关题）", f"{avg_cov:.0%}"),
        ("相关题通过率（引用命中 且 覆盖率≥"
         f"{args.threshold}）", f"{passed - irr['pass']}/{relevant_n} = {(passed - irr['pass'])/rn:.0%}"),
        ("无关题拒答率（明确回答无相关信息）",
         f"{irr['pass']}/{irr['n']} = {irr['pass']/max(irr['n'],1):.0%}"),
        ("平均单条耗时", f"{time_sum/n:.0f}s"),
        ("总耗时", f"{time_sum/60:.1f} 分钟"),
    ]

    conclusion_lines = [
        f"本次共执行 {n} 条图谱问答案例（{args.mode} 模式，相关 {relevant_n} + "
        f"无关 {irr['n']}），相关题通过 {passed - irr['pass']} 条"
        f"（{(passed - irr['pass'])/rn:.0%}），无关题拒答 {irr['pass']}/{irr['n']}。",
        f"相关题引用命中率 {ref_hit_n/rn:.0%}，平均要点覆盖率 {avg_cov:.0%}。",
    ]
    fails = [r for r in rows if r[5] == "❌"]
    if fails:
        conclusion_lines.append(
            "未通过用例编号：" + "、".join(r[0] for r in fails)
            + "。建议：对照原文人工复核是否为幻觉或归纳遗漏；"
              "实体关系类问题可尝试 local 模式，宏观归纳类可尝试 global 模式对比。"
        )
    else:
        conclusion_lines.append("全部通过，图谱问答质量符合预期。")

    write_report(
        args.out,
        "LightRAG 图谱问答测试报告",
        f"执行时间：{datetime.datetime.now():%Y-%m-%d %H:%M} ｜ mode={args.mode} ｜ "
        f"用例来源：{os.path.basename(args.cases)}",
        metrics,
        ["编号", "测试问题", "引用命中", "要点命中", "耗时", "结论"],
        rows,
        "\n".join(conclusion_lines),
    )
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
