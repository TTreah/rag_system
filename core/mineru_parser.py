"""
mineru_parser.py — PDF → Markdown（MinerU），含四项清洗

处理的问题：
  1. Unicode 归一化（康熙部首、全角、孤立代理字符）
  2. HTML 表格 → Markdown（正文去标签，表格作为独立资产）
  3. 图片提取（改用 pymupdf 取图片本体，MinerU pipeline 后端不可靠）
  4. 页眉/页脚过滤（按 content_list 的 header/footer/page_footnote 类型剔除）

设计：不做页级溯源，只保留文件级；正文按 Markdown 标题结构交给
TextChunker(strategy="markdown") 分块。
"""

import os
import re
import json
import uuid
import html as _html
import urllib.request
import urllib.error
from html.parser import HTMLParser
from typing import Dict, List, Tuple

DEFAULT_MINERU_URL = "http://localhost:8000/file_parse"

_SKIP_TYPES = {"header", "footer", "page_footnote", "page_number"}
_MIN_IMAGE_BYTES = 10 * 1024


# ═══════════════════════════════════════════════════════════════
# MinerU 调用
# ═══════════════════════════════════════════════════════════════

def _post_pdf(path: str, api_url: str, timeout: int = 1800) -> dict:
    boundary = uuid.uuid4().hex

    def field(name, value):
        return (f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="{name}"\r\n\r\n{value}\r\n').encode("utf-8")

    with open(path, "rb") as f:
        data = f.read()
    name = os.path.basename(path)
    parts = [
        f"--{boundary}\r\n".encode(),
        (f'Content-Disposition: form-data; name="files"; filename="{name}"\r\n'
         f"Content-Type: application/pdf\r\n\r\n").encode("utf-8"),
        data, b"\r\n",
        field("backend", "pipeline"),
        field("parse_method", "auto"),
        field("lang_list", "ch"),
        field("table_enable", "true"),
        field("return_md", "true"),
        field("return_content_list", "true"),
        f"--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)
    req = urllib.request.Request(
        api_url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    results = data.get("results") or {}
    if not results:
        raise RuntimeError(f"MinerU 无结果: {data.get('error')}")
    return list(results.values())[0]


# ═══════════════════════════════════════════════════════════════
# HTML 表格 → Markdown
# ═══════════════════════════════════════════════════════════════

class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_markdown(table_html: str) -> str:
    """把 MinerU 的 <table> HTML 转成 Markdown 管道表（不做 span 合并）。"""
    if not table_html or "<table" not in table_html.lower():
        return ""
    p = _TableParser()
    p.feed(table_html)
    rows = [[c.replace("|", "\\|") for c in r] for r in p.rows if any(c for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════
# 正文重建（过滤页眉页脚、表格转资产）
# ═══════════════════════════════════════════════════════════════

def rebuild_markdown(content_list) -> Tuple[str, List[dict]]:
    """从 content_list 重建正文 Markdown，返回 (md, 表格资产列表)。"""
    if isinstance(content_list, str):
        content_list = json.loads(content_list)
    blocks = []
    tables = []
    for item in content_list or []:
        t = item.get("type")
        if t in _SKIP_TYPES:
            continue
        page = item.get("page_idx", "")
        if t == "table":
            md_table = html_table_to_markdown(item.get("table_body", ""))
            if not md_table:
                continue
            cap = item.get("table_caption") or []
            if isinstance(cap, str):
                try:
                    cap = json.loads(cap)
                except Exception:
                    cap = [cap]
            caption = " ".join(cap).strip() or f"表格（第 {int(page) + 1} 页）"
            tables.append({"data": md_table, "label": f"Table {len(tables) + 1}",
                           "page": page, "caption": caption})
            blocks.append(f"【表格】{caption}")
        elif t == "image":
            # 图片由 pymupdf 单独提取，正文不放占位，避免重复
            continue
        elif t in ("text", "title", "equation", "list"):
            text = (item.get("text") or "").strip()
            if not text:
                continue
            level = item.get("text_level")
            if level and str(level).isdigit():
                blocks.append("#" * min(int(level), 6) + " " + text)
            else:
                blocks.append(text)
    return "\n\n".join(blocks), tables


# ═══════════════════════════════════════════════════════════════
# 图片提取（pymupdf）
# ═══════════════════════════════════════════════════════════════

def extract_images(pdf_path: str, content_list) -> List[dict]:
    import pymupdf
    if isinstance(content_list, str):
        content_list = json.loads(content_list)
    page_texts: Dict[int, str] = {}
    for item in content_list or []:
        if item.get("type") in ("text", "title") and item.get("text"):
            pg = int(item.get("page_idx", 0))
            page_texts.setdefault(pg, "")
            page_texts[pg] += item["text"] + " "

    assets = []
    doc = pymupdf.open(pdf_path)
    try:
        for pi in range(doc.page_count):
            for img in doc[pi].get_images(full=True):
                try:
                    info = doc.extract_image(img[0])
                    if len(info.get("image", b"")) < _MIN_IMAGE_BYTES:
                        continue
                    assets.append({
                        "kind": "image", "data": info["image"],
                        "ext": info.get("ext", "png"),
                        "label": f"Page {pi + 1}",
                        "context": page_texts.get(pi, "")[:300],
                    })
                except Exception:
                    continue
    finally:
        doc.close()
    return assets


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

_CACHE_DIR = os.path.expanduser("~/.cache/mineru_parse")


def _cache_path(pdf_path: str) -> str:
    st = os.stat(pdf_path)
    key = f"{os.path.abspath(pdf_path)}|{st.st_size}|{int(st.st_mtime)}"
    import hashlib
    return os.path.join(_CACHE_DIR, hashlib.md5(key.encode()).hexdigest() + ".json")


def parse_pdf(pdf_path: str, api_url: str = DEFAULT_MINERU_URL):
    """返回 ParsedDocument（content 为清洗后的 Markdown，assets 含表格+图片）。

    MinerU 结果按「路径+大小+mtime」缓存到 ~/.cache/mineru_parse，
    参数扫描/重建复用，避免重复 OCR。
    """
    from core.document_parser import DocumentParser, ParsedDocument

    cache_file = _cache_path(pdf_path)
    content_list = None
    if os.path.exists(cache_file):
        try:
            content_list = json.load(open(cache_file, encoding="utf-8"))
        except Exception:
            content_list = None
    if content_list is None:
        result = _post_pdf(pdf_path, api_url)
        content_list = result.get("content_list") or []
        if isinstance(content_list, str):
            content_list = json.loads(content_list)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        json.dump(content_list, open(cache_file, "w", encoding="utf-8"),
                  ensure_ascii=False)
    md, tables = rebuild_markdown(content_list)

    # 1) Unicode 归一化（NFKC + 部首映射 + 代理字符清理）
    md = DocumentParser._normalize_text(md)

    # 2) 表格资产
    assets = []
    for t in tables:
        assets.append({
            "kind": "table", "data": t["data"], "ext": "md",
            "label": t["label"],
            "context": t["caption"],
        })

    # 3) 图片资产（pymupdf 提取）
    assets.extend(extract_images(pdf_path, content_list))

    title = ""
    for line in md.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    title = title or os.path.splitext(os.path.basename(pdf_path))[0]

    return ParsedDocument(
        metadata={"title": title, "parser": "mineru"},
        content=md,
        sections=[],          # 空 sections → 流水线按 markdown 策略分块
        assets=assets,
    )
