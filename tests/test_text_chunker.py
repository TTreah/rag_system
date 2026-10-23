"""
test_text_chunker.py — 分块器回归测试

重点覆盖曾经出错的地方：
  1. 递归分块必须保留分隔符（段落分隔符曾被整个丢掉，导致段落粘连）
  2. char_start/char_end 必须精确指向原文切片
  3. chunk_size 是上界，_merge_small 不得突破它
  4. 各策略在边界输入下不崩

运行: python test_text_chunker.py
"""

import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
del _sys


import sys

from core.text_chunker import TextChunker

FAILURES = []


def check(cond, label, detail=""):
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}  {detail}")
        FAILURES.append(label)


def hr(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


# ─── 1. 分隔符保留 ──────────────────────────────────────────

def test_separators_preserved():
    hr("1. 递归分块保留分隔符")

    c = TextChunker(strategy="recursive", chunk_size=512, overlap=0, min_chunk_size=1)

    text = "AAA\n\nBBB"
    out = c.split(text)
    check(len(out) == 1 and out[0].text == text,
          "段落分隔符 \\n\\n 未被丢弃", f"got {out[0].text!r}")

    text = "\n\n".join(f"para{i}" for i in range(5))
    out = c.split(text)
    check("\n\n" in "".join(ch.text for ch in out),
          "多段落文档仍含段落分隔符")

    # 换行分隔符（第二级）
    text = "\n".join(f"line{i}" for i in range(6))
    c2 = TextChunker(strategy="recursive", chunk_size=12, overlap=0, min_chunk_size=1)
    out = c2.split(text)
    check(len(out) > 1, "内容超长时确实发生切分", f"chunks={len(out)}")
    check(all("\n\n" not in ch.text for ch in out),
          "同级分隔符不串级引入")


# ─── 2. 偏移精确性 ──────────────────────────────────────────

def test_offsets_exact():
    hr("2. char_start/char_end 精确指向原文切片")

    text = "\n\n".join(f"段落{i}：这是一段用于测试分块边界的文本内容。" for i in range(8))
    c = TextChunker(strategy="recursive", chunk_size=60, overlap=0, min_chunk_size=1)
    out = c.split(text)

    check(len(out) > 1, "确实发生了多次切分", f"chunks={len(out)}")
    bad = [ch for ch in out if ch.text != text[ch.char_start:ch.char_end]]
    check(not bad, "每个 chunk 的 offset 都等于 text[char_start:char_end]",
          f"{len(bad)}/{len(out)} 个不匹配")

    # 分段 + 无 overlap 时，chunk 应当能拼回原文（只差被吃掉的分隔符）
    check("段落0" in out[0].text and "段落7" in out[-1].text,
          "首尾 chunk 覆盖正确范围")


# ─── 3. chunk_size 上界 ─────────────────────────────────────

def test_chunk_size_is_upper_bound():
    hr("3. chunk_size 是上界")

    # 大量短段落 → _merge_small 会尝试合并，但不得突破 chunk_size
    text = "。".join("短句" * 3 for _ in range(40))
    for strategy in ("recursive", "sentence", "markdown", "fixed"):
        c = TextChunker(strategy=strategy, chunk_size=100, overlap=0, min_chunk_size=80)
        out = c.split(text)
        over = [len(ch.text) for ch in out if len(ch.text) > 100]
        check(not over, f"[{strategy}] overlap=0 时无 chunk 超过 chunk_size",
              f"超长: {over}")

    # 关键回归：overlap > 0 时，正文 + 重叠后的总长仍须 <= chunk_size
    # （正文若按 chunk_size 累积，_add_overlap 会把它顶到 chunk_size + overlap）
    long_text = "。".join(f"这是第{i}个用于验证重叠余量的句子内容" for i in range(60))
    for strategy in ("recursive", "sentence", "markdown", "section"):
        c = TextChunker(strategy=strategy, chunk_size=120, overlap=40, min_chunk_size=20)
        out = c.split(long_text)
        over = [(i, len(ch.text)) for i, ch in enumerate(out) if len(ch.text) > 120]
        check(not over, f"[{strategy}] overlap=40 时无 chunk 超过 chunk_size",
              f"超长: {over}")

    # min_chunk_size > chunk_size 的退化配置不应死循环或抛错
    c = TextChunker(strategy="recursive", chunk_size=50, overlap=0, min_chunk_size=500)
    out = c.split("。".join(f"句子{i}" for i in range(30)))
    check(bool(out), "min_chunk_size > chunk_size 时仍产出结果",
          f"chunks={len(out)}")

    # overlap >= chunk_size 是无意义配置，不应把 chunk 撑成两倍
    c = TextChunker(strategy="recursive", chunk_size=50, overlap=50, min_chunk_size=1)
    out = c.split("。".join(f"句子{i}" for i in range(30)))
    over = [len(ch.text) for ch in out if len(ch.text) > 50]
    check(not over, "overlap >= chunk_size 时被忽略", f"超长: {over}")


# ─── 4. 各策略健壮性 ────────────────────────────────────────

def test_all_strategies_robust():
    hr("4. 各策略在边界输入下不崩")

    sample = "# 标题一\n\n正文段落一。\n\n## 标题二\n\n正文段落二。\n\n- 列表项\n"

    for strategy in ("fixed", "recursive", "sentence", "markdown", "section"):
        try:
            c = TextChunker(strategy=strategy, chunk_size=64, overlap=8, min_chunk_size=10)
            out = c.split(sample, metadata={"src": "t"})
            ok = bool(out) and all(ch.text.strip() for ch in out)
            check(ok, f"[{strategy}] 有非空输出且无空 chunk", f"chunks={len(out)}")
        except Exception as e:
            check(False, f"[{strategy}] 未抛异常", f"{type(e).__name__}: {e}")

    # 空 / 纯空白输入
    for strategy in ("fixed", "recursive", "sentence", "markdown", "section"):
        try:
            out = TextChunker(strategy=strategy).split("")
            out2 = TextChunker(strategy=strategy).split("   \n\n  ")
            check(out == [] and out2 == [], f"[{strategy}] 空输入返回空列表")
        except Exception as e:
            check(False, f"[{strategy}] 空输入未抛异常", f"{type(e).__name__}: {e}")

    # 未知策略应当报错而不是静默返回空
    try:
        TextChunker(strategy="nope").split("abc")
        check(False, "未知策略抛 ValueError")
    except ValueError:
        check(True, "未知策略抛 ValueError")


# ─── 5. markdown 元信息 ─────────────────────────────────────

def test_markdown_metadata():
    hr("5. markdown 策略的标题元信息")

    text = "# 一级标题\n\n内容A\n\n## 二级标题\n\n内容B"
    c = TextChunker(strategy="markdown", chunk_size=200, overlap=0, min_chunk_size=1)
    out = c.split(text)

    headings = [ch.metadata.get("heading") for ch in out]
    check("一级标题" in headings, "一级标题写入 metadata.heading", f"{headings}")
    check("二级标题" in headings, "二级标题写入 metadata.heading", f"{headings}")
    levels = [ch.metadata.get("heading_level") for ch in out if ch.metadata.get("heading_level")]
    check(levels == [1, 2], "heading_level 正确", f"{levels}")


# ─── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("text_chunker 回归测试")
    print(f"Python: {sys.executable}")

    test_separators_preserved()
    test_offsets_exact()
    test_chunk_size_is_upper_bound()
    test_all_strategies_robust()
    test_markdown_metadata()

    hr("SUMMARY")
    if FAILURES:
        print(f"  {len(FAILURES)} 项失败:")
        for f in FAILURES:
            print(f"    - {f}")
        print("\n  Overall: FAILED")
        sys.exit(1)
    print("  Overall: ALL PASS")
    sys.exit(0)
