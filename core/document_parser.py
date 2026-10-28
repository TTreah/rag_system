"""
document_parser.py — 多类型文档解析器
根据文件后缀自动选择解析方式，提取文本内容，输出统一结构供分块和向量化使用。

支持的格式:
    PDF (.pdf)         → PyMuPDF
    Word (.docx, .doc) → python-docx
    纯文本 (.txt, .md, .yaml, .yml) → 内置
    PowerPoint (.pptx, .ppt) → python-pptx

输出结构:
    {
        "file_path": str,
        "file_type": str,
        "metadata": {"title": str, "pages": int, ...},
        "content": str,                # 全文纯文本
        "sections": [                  # 分段列表，供 chunk 使用
            {"index": 0, "text": str, "label": str},
            ...
        ]
    }
"""

import os
import sys
import codecs
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class Section:
    """文档中的一个逻辑段落/页面"""
    index: int
    text: str
    label: str = ""          # 页面、sheet、slide 等标签


@dataclass
class ParsedDocument:
    """解析结果"""
    file_path: str = ""
    file_type: str = ""      # 后缀，如 ".pdf"
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: str = ""        # 全量纯文本
    sections: List[Section] = field(default_factory=list)
    # 非文本资产：图片(bytes) 与表格(markdown text)，由流水线落盘并生成描述
    # 结构: [{"kind": "image"|"table", "data": bytes|str, "ext": "png",
    #         "label": "Page 3", "context": "周边文本摘要"}]
    assets: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "metadata": self.metadata,
            "content": self.content,
            "sections": [{"index": s.index, "text": s.text, "label": s.label}
                         for s in self.sections],
        }


# ═══════════════════════════════════════════════════════════════
# 文档解析器
# ═══════════════════════════════════════════════════════════════

class DocumentParser:
    """
    多类型文档解析器。

    根据文件后缀自动选择解析策略，输出统一的 ParsedDocument。
    使用方式:
        parser = DocumentParser()
        doc = parser.parse("/path/to/file.pdf")
        print(doc.content)          # 全文
        for sec in doc.sections:    # 逐段/逐页
            ...
    """

    # ── 后缀 → (方法名, 描述) ────────────────────────────────
    HANDLERS: Dict[str, Tuple[str, str]] = {
        # PDF
        ".pdf":  ("_parse_pdf",   "PDF Document"),

        # Word
        ".docx": ("_parse_docx",  "Word Document"),
        ".doc":  ("_parse_docx",  "Legacy Word (limited)"),

        # 纯文本
        ".txt":  ("_parse_text",  "Plain Text"),
        ".md":   ("_parse_text",  "Markdown"),
        ".yaml": ("_parse_text",  "YAML"),
        ".yml":  ("_parse_text",  "YAML"),

        # 演示文稿
        ".pptx": ("_parse_pptx",  "PowerPoint"),
        ".ppt":  ("_parse_pptx",  "PowerPoint (limited)"),

        # 独立图片文件（无正文，作为资产入库，文件名即正文）
        ".png":  ("_parse_image", "Image"),
        ".jpg":  ("_parse_image", "Image"),
        ".jpeg": ("_parse_image", "Image"),
        ".webp": ("_parse_image", "Image"),
    }

    def __init__(self, use_mineru: bool = False, mineru_url: str = None):
        from config import MINERU_URL
        mineru_url = mineru_url or MINERU_URL
        """use_mineru=True 时，PDF 走 MinerU（逐页混合 OCR + Markdown 重建）。"""
        self.use_mineru = use_mineru
        self.mineru_url = mineru_url

    # ── 公共 API ──────────────────────────────────────────

    # CJK 部首补充区（U+2E80–U+2EFF）→ 统一汉字。NFKC 不覆盖这一区，
    # 只收录本库实测出现的（扫描 CEO-files 3500 chunks 得出）
    _RADICAL_MAP = str.maketrans({
        "⻋": "车", "⻔": "门", "⻛": "风", "⻓": "长",
        "⻄": "西", "⻰": "龙", "⻜": "飞", "⻚": "页",
        "⻘": "青", "⻆": "角", "⻢": "马", "⻨": "麦",
        "⻬": "齐", "⻅": "见", "⻩": "黄", "⻦": "鸟",
        "⺠": "民", "⻉": "贝", "⻥": "鱼", "⻁": "虎",
        "⺟": "母", "⻣": "骨", "⻧": "盐", "⻙": "韦",
        "⻝": "食",
    })

    @staticmethod
    def _normalize_text(text: str) -> str:
        """文本归一化（入库前必须过这道，否则非标字符让嵌入和检索双通道失效）：
        1. 剥离孤立代理字符
        2. NFKC：康熙部首区(U+2F00段)→标准汉字、全角→半角、半角标点→中文标点
        3. 部首补充区(U+2E80段)映射表
        """
        import re
        import unicodedata
        if not text:
            return text
        text = re.sub(r"[\ud800-\udfff]", "", text)
        text = text.replace("\ufffd", "")   # OCR 无法识别的替换符，直接剔除
        text = unicodedata.normalize("NFKC", text)
        return text.translate(DocumentParser._RADICAL_MAP)

    @staticmethod
    def _sanitize(text: str) -> str:
        """归一化入口（保留旧名）。实际逻辑见 _normalize_text。"""
        return DocumentParser._normalize_text(text)

    def parse(self, file_path: str) -> ParsedDocument:
        """
        解析单份文件，自动选择策略。

        Raises:
            ValueError: 不支持的文件类型
        """
        ext = Path(file_path).suffix.lower()
        if ext not in self.HANDLERS:
            raise ValueError(
                f"Unsupported file type: '{ext}'. "
                f"Supported: {sorted(self.HANDLERS.keys())}"
            )

        if ext == ".pdf" and self.use_mineru:
            try:
                from core.mineru_parser import parse_pdf
                doc = parse_pdf(file_path, self.mineru_url)
                if not doc.content.strip():
                    # MinerU 可能返回"非空 content_list 但正文全空"（如整页表格
                    # 识别成空单元格）；此时回退原生解析，避免整份文档丢失
                    raise RuntimeError("MinerU 正文为空")
                doc.file_path = file_path
                doc.file_type = ext
                doc.metadata.setdefault("description", "PDF (MinerU)")
                doc.content = self._sanitize(doc.content)
                return doc
            except Exception as e:
                # MinerU 不可用不应阻断入库：回退到原生文本解析
                import logging
                logging.getLogger(__name__).warning(
                    "MinerU 解析失败，回退原生解析: %s (%s)", file_path, e)

        method_name, description = self.HANDLERS[ext]
        parser = getattr(self, method_name)
        doc = parser(file_path)
        doc.file_path = file_path
        doc.file_type = ext
        doc.metadata.setdefault("description", description)

        # 统一清洗代理字符（正文 / sections / 资产文本）
        doc.content = self._sanitize(doc.content)
        for s in doc.sections:
            s.text = self._sanitize(s.text)
        for a in doc.assets:
            if isinstance(a.get("data"), str):
                a["data"] = self._sanitize(a["data"])
            if isinstance(a.get("context"), str):
                a["context"] = self._sanitize(a["context"])
        return doc

    def parse_batch(self, file_paths: List[str]) -> List[ParsedDocument]:
        """批量解析文件"""
        results = []
        for fp in file_paths:
            try:
                results.append(self.parse(fp))
            except Exception as e:
                print(f"[WARN] Failed to parse {fp}: {e}", file=sys.stderr)
        return results

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """返回所有支持的后缀列表"""
        return sorted(cls.HANDLERS.keys())

    # ── PDF 解析 ─────────────────────────────────────────

    # 句末标点（这些字符后面应该断开）
    _SENTENCE_ENDS = {".", "!", "?", "。", "！", "？", ":", "：", ";", "；",
                      ")", "）", "\"", "“", "”", "’"}

    def _parse_pdf(self, path: str) -> ParsedDocument:
        import pymupdf
        doc_obj = pymupdf.open(path)
        metadata = {
            "title": doc_obj.metadata.get("title", ""),
            "author": doc_obj.metadata.get("author", ""),
            "pages": doc_obj.page_count,
            "format": doc_obj.metadata.get("format", ""),
        }
        sections = []
        full_parts = []
        assets = []

        for i in range(doc_obj.page_count):
            page = doc_obj[i]

            # ── 步骤 1: 用 blocks 模式提取，块内行已合并 ──
            blocks = page.get_text("blocks")
            # (x0, y0, x1, y1, text, block_no, block_type)

            # ── 步骤 2: 过滤 + 块内清洗（修复块内仍存在的逐词换行）──
            cleaned_blocks = []
            for b in blocks:
                raw = b[4].strip()
                if not raw or b[6] != 0:  # type 0 = text
                    continue
                # 块内：合并非句末的换行为空格
                cleaned = self._clean_block_text(raw)
                if cleaned:
                    cleaned_blocks.append((b[1], cleaned))  # (y, text)

            cleaned_blocks.sort(key=lambda t: t[0])

            # ── 步骤 3: 合并相邻块（跨 block 的段落）──
            merged = self._merge_blocks(cleaned_blocks)
            page_text = "\n\n".join(merged)

            # ── 步骤 4: 提取表格 ──
            tables = page.find_tables()
            table_texts = []
            for tbl in tables:
                try:
                    tbl_data = tbl.extract()
                    fmt = self._format_table(tbl_data)
                    table_texts.append(fmt)
                    if fmt.strip():
                        assets.append({
                            "kind": "table", "data": fmt, "ext": "md",
                            "label": f"Page {i + 1}",
                            "context": page_text[:300],
                        })
                except Exception:
                    pass

            # ── 步骤 5: 提取图片（过小的 logo/图标除外）──
            for img in page.get_images(full=True):
                try:
                    info = doc_obj.extract_image(img[0])
                    if len(info["image"]) < 10 * 1024:  # <10KB 多为装饰图标
                        continue
                    assets.append({
                        "kind": "image", "data": info["image"],
                        "ext": info.get("ext", "png"),
                        "label": f"Page {i + 1}",
                        "context": page_text[:300],
                    })
                except Exception:
                    continue

            if table_texts:
                page_text += "\n\n" + "\n\n".join(table_texts)

            if page_text.strip():
                sections.append(Section(index=i, text=page_text.strip(),
                                        label=f"Page {i + 1}"))
                full_parts.append(page_text.strip())

        doc_obj.close()
        return ParsedDocument(
            metadata=metadata,
            content="\n\n".join(full_parts),
            sections=sections,
            assets=assets,
        )

    @classmethod
    def _clean_block_text(cls, text: str) -> str:
        """
        清洗单个 block 内的文本：
        - 若非句末标点结尾 → 合并到下一行（用空格替代换行）
        - 若行首是小写字母或中文 → 也很可能是连续文本
        - 最后规范化多余空白
        """
        import re

        lines = text.split("\n")
        if len(lines) <= 1:
            return re.sub(r"[ \t]{2,}", " ", text)

        result = [lines[0]]
        for k in range(1, len(lines)):
            prev_line = result[-1]
            curr_line = lines[k]
            if not prev_line or not curr_line:
                result.append(curr_line)
                continue

            prev_last = prev_line.rstrip()[-1] if prev_line.rstrip() else ""
            curr_first = curr_line.lstrip()[0] if curr_line.lstrip() else ""

            # 判断是否应该合并：
            # 1. 上一行不以句末标点结尾
            # 2. 当前行以小写字母开头（英文）→ 肯定是接上一行
            # 3. 上一行以逗号结尾 → 还在同一句
            should_merge = (
                prev_last not in cls._SENTENCE_ENDS
                or curr_first.islower()
                or prev_last in {",", "，", ";", "；"}
            )

            if should_merge:
                result[-1] = prev_line + " " + curr_line.lstrip()
            else:
                result.append(curr_line)

        # 规范化空白：多空格→单空格，但保留段落换行
        cleaned = "\n".join(result)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned

    @staticmethod
    def _merge_blocks(text_blocks: List[tuple]) -> List[str]:
        """
        合并属于同一段落的相邻文本块。
        """
        if not text_blocks:
            return []

        SENTENCE_ENDS = {".", "!", "?", "。", "！", "？"}

        merged = []
        buf = text_blocks[0][1]
        prev_y = text_blocks[0][0]

        for j in range(1, len(text_blocks)):
            y, text = text_blocks[j]
            prev_last = buf.rstrip()[-1] if buf.rstrip() else ""
            gap = y - prev_y

            # 前块未结束 或 间距小 → 合并
            if prev_last not in SENTENCE_ENDS or gap < 14:
                buf += " " + text
            else:
                merged.append(buf)
                buf = text
            prev_y = y

        merged.append(buf)
        return merged

    # ── Word 解析 ────────────────────────────────────────

    def _parse_docx(self, path: str) -> ParsedDocument:
        from docx import Document

        # 真 .doc（二进制格式）python-docx 打不开，先转成 .docx 再解析。
        # 只转换内容，file_path 仍指向原文件——清单追踪的是源文件本身。
        actual_path = path
        tmp_converted = None
        if path.lower().endswith(".doc"):
            try:
                Document(path)
            except Exception:
                actual_path = self._convert_doc_to_docx(path)
                tmp_converted = actual_path

        doc_obj = Document(actual_path)

        # 读取文档属性
        props = doc_obj.core_properties
        metadata = {
            "title": props.title or "",
            "author": props.author or "",
            "paragraphs": len(doc_obj.paragraphs),
        }

        sections = []
        current_section_idx = 0
        current_lines = []
        section_label = "Body"

        for para in doc_obj.paragraphs:
            text = para.text
            if not text.strip():
                if current_lines:
                    sections.append(Section(
                        index=current_section_idx,
                        text="\n".join(current_lines),
                        label=section_label,
                    ))
                    current_section_idx += 1
                    current_lines = []
                continue

            style = para.style.name if para.style else ""
            # 标题 → 新 section
            if style.startswith("Heading"):
                if current_lines:
                    sections.append(Section(
                        index=current_section_idx,
                        text="\n".join(current_lines),
                        label=section_label,
                    ))
                    current_section_idx += 1
                    current_lines = []
                section_label = f"{style}: {text[:60]}"
                current_lines.append(text)
            else:
                current_lines.append(text)

        # 收尾
        if current_lines:
            sections.append(Section(
                index=current_section_idx,
                text="\n".join(current_lines),
                label=section_label,
            ))

        # 提取表格
        assets = []
        for ti, table in enumerate(doc_obj.tables):
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append("\t".join(cells))
            tbl_text = "\n".join(rows)
            if tbl_text.strip():
                sections.append(Section(
                    index=len(sections),
                    text=tbl_text,
                    label=f"Table {ti + 1}",
                ))
                assets.append({
                    "kind": "table", "data": tbl_text, "ext": "md",
                    "label": f"Table {ti + 1}",
                    "context": "\n".join(s.text for s in sections[-3:])[:300],
                })

        # 提取图片（文档关系里的嵌入图，过小的 logo/图标除外）
        for ri, rel in enumerate(doc_obj.part.rels.values()):
            if "image" not in rel.reltype:
                continue
            try:
                blob = rel.target_part.blob
            except Exception:
                continue
            if len(blob) < 10 * 1024:
                continue
            ext = rel.target_part.partname.ext if hasattr(rel.target_part.partname, 'ext') else "png"
            assets.append({
                "kind": "image", "data": blob,
                "ext": str(ext).lstrip(".") or "png",
                "label": f"InlineImage {ri + 1}",
                "context": "\n".join(s.text for s in sections[:3])[:300],
            })

        full_text = "\n\n".join(s.text for s in sections)
        if tmp_converted:
            try:
                os.unlink(tmp_converted)
            except OSError:
                pass
        return ParsedDocument(
            metadata=metadata,
            content=full_text,
            sections=sections,
            assets=assets,
        )

    @staticmethod
    def _convert_doc_to_docx(path: str) -> str:
        """把真 .doc 转成临时 .docx，返回临时文件路径。

        优先 macOS 自带的 textutil（零依赖），其次 LibreOffice 的 soffice。
        都没有就给出明确报错，而不是让 python-docx 抛一个看不懂的包格式异常。
        """
        converters = ["textutil", "soffice"]
        for tool in converters:
            if not shutil.which(tool):
                continue
            fd, tmp = tempfile.mkstemp(suffix=".docx")
            os.close(fd)
            try:
                if tool == "textutil":
                    subprocess.run(
                        ["textutil", "-convert", "docx", path, "-output", tmp],
                        check=True, capture_output=True, timeout=120,
                    )
                else:
                    outdir = os.path.dirname(tmp)
                    subprocess.run(
                        ["soffice", "--headless", "--convert-to", "docx",
                         "--outdir", outdir, path],
                        check=True, capture_output=True, timeout=180,
                    )
                    produced = os.path.join(
                        outdir, os.path.splitext(os.path.basename(path))[0] + ".docx"
                    )
                    if produced != tmp:
                        os.replace(produced, tmp)
                return tmp
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                continue
        raise ValueError(
            f"无法解析旧版 .doc 文件: {path}\n"
            f"需要 textutil（macOS 自带）或 LibreOffice（soffice）做格式转换，"
            f"当前系统两者都没有。请手工把文件另存为 .docx 后重试。"
        )

    # ── 纯文本解析 ───────────────────────────────────────

    def _parse_text(self, path: str) -> ParsedDocument:
        # 先看真实字节里有没有 UTF-8 BOM，决定解码后是否要剥掉首字符
        with open(path, "rb") as fb:
            has_bom = fb.read(3) == codecs.BOM_UTF8

        # 按序尝试编码，记录真正生效的那个（原来无论走哪条分支都报最后一个候选值）
        used_encoding = "utf-8"
        for encoding in ["utf-8", "gbk", "gb2312", "latin-1"]:
            try:
                with open(path, "r", encoding=encoding) as f:
                    text = f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
            used_encoding = encoding
            break
        else:
            # latin-1 对任意字节都能解码，实际到不了这里，留作兜底
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            used_encoding = "utf-8 (errors=replace)"

        # utf-8 解码会把 BOM 还原成 U+FEFF，不剥掉的话会污染首个 chunk
        if has_bom:
            text = text[1:]

        # 按段落分 section（连续非空行为一段）
        lines = text.splitlines()
        sections = []
        buf = []
        sec_idx = 0

        for line in lines:
            if line.strip():
                buf.append(line)
            else:
                if buf:
                    sections.append(Section(
                        index=sec_idx, text="\n".join(buf),
                        label=f"Para {sec_idx + 1}",
                    ))
                    sec_idx += 1
                    buf = []
        if buf:
            sections.append(Section(
                index=sec_idx, text="\n".join(buf),
                label=f"Para {sec_idx + 1}",
            ))

        return ParsedDocument(
            metadata={"lines": len(lines), "encoding": used_encoding},
            content=text.strip(),
            sections=sections,
        )

    # ── PowerPoint 解析 ──────────────────────────────────

    def _parse_pptx(self, path: str) -> ParsedDocument:
        from pptx import Presentation
        prs = Presentation(path)
        metadata = {"slides": len(prs.slides)}

        sections = []
        full_parts = []

        for si, slide in enumerate(prs.slides):
            texts = []
            slide_title = ""
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = para.text.strip()
                        if t:
                            # 第一个 shape 的文本作为标题
                            if not slide_title and shape.has_text_frame:
                                slide_title = t[:80]
                            texts.append(t)
                if shape.has_table:
                    tbl = shape.table
                    for row in tbl.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        texts.append("\t".join(cells))

            slide_text = "\n".join(texts)
            if slide_text.strip():
                label = f"Slide {si + 1}"
                if slide_title:
                    label += f": {slide_title}"
                sections.append(Section(index=si, text=slide_text, label=label))
                full_parts.append(slide_text)

        return ParsedDocument(
            metadata=metadata,
            content="\n\n".join(full_parts),
            sections=sections,
        )

    # ── 独立图片 ─────────────────────────────────────────

    def _parse_image(self, path: str) -> ParsedDocument:
        """独立图片文件：无正文可解析，整张图作为资产交给流水线。

        content 用文件名兜底（产品图编号等文件名本身可检索）。
        """
        with open(path, "rb") as f:
            data = f.read()
        name = Path(path).stem
        ext = Path(path).suffix.lower().lstrip(".")
        return ParsedDocument(
            metadata={"title": name, "image": True},
            content=name,
            sections=[Section(index=0, text=name, label="Image")],
            assets=[{"kind": "image", "data": data, "ext": ext,
                     "label": "Image", "context": f"文件名：{name}"}],
        )

    # ── 工具 ─────────────────────────────────────────────

    @staticmethod
    def _format_table(data: List[List[Any]]) -> str:
        """将二维表格转为可读文本"""
        if not data:
            return ""
        lines = []
        for row in data:
            cells = [str(c) if c is not None else "" for c in row]
            lines.append("\t".join(cells))
        return "\n".join(lines)
