"""
text_chunker.py — 多策略文本分块模块
将文档文本按不同策略切分为适合向量化的 chunk。

策略:
    - fixed:      固定窗口 + 重叠
    - recursive:  递归按分隔符拆分（段落 → 句子 → 字符）
    - sentence:   按句子边界切分，保持语义完整
    - markdown:   感知 Markdown 结构（标题、代码块、列表）
    - semantic:   按 ParsedDocument 的 sections 直接分块

每个 chunk 附带元信息，便于溯源到原文档和位置。
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """文本块"""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0


class TextChunker:
    """
    多策略文本分块器。

    使用方式:
        chunker = TextChunker(strategy="recursive", chunk_size=512, overlap=64)
        chunks = chunker.split(document_content, metadata={})
        for c in chunks:
            print(c.text, c.metadata)
    """

    # 递归拆分默认分隔符
    DEFAULT_SEPARATORS = [
        "\n\n",     # 段落
        "\n",       # 行
        "。",       # 中文句号
        "！",       # 中文感叹号
        "？",       # 中文问号
        "；",       # 中文分号
        ". ",       # 英文句号（带空格，避免拆 URL）
        "! ",       # 英文感叹号
        "? ",       # 英文问号
        "; ",       # 英文分号
        " ",        # 空格（最后手段）
        "",         # 逐字符
    ]

    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 512,
        overlap: int = 64,
        separators: Optional[List[str]] = None,
        min_chunk_size: int = 50,
    ):
        """
        Args:
            strategy: 分块策略
                - "fixed": 固定窗口 + 重叠
                - "recursive": 递归按分隔符拆分
                - "sentence": 按句子边界切分
                - "markdown": Markdown 结构感知
                - "section": 按文档原有 sections 分块（自适应大小）
            chunk_size: 目标 chunk 大小（字符数）
            overlap: 相邻 chunk 重叠字符数
            separators: 递归策略的分隔符优先级（可选）
            min_chunk_size: 最小 chunk 大小，过小的 chunk 会被合并
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.min_chunk_size = min_chunk_size

    def split(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """对文本执行分块"""
        if not text or not text.strip():
            return []

        base_meta = metadata or {}

        if self.strategy == "fixed":
            chunks = self._split_fixed(text, base_meta)
        elif self.strategy == "recursive":
            chunks = self._split_recursive(text, base_meta)
        elif self.strategy == "sentence":
            chunks = self._split_sentence(text, base_meta)
        elif self.strategy == "markdown":
            chunks = self._split_markdown(text, base_meta)
        elif self.strategy == "section":
            chunks = self._split_sections(text, base_meta)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # 过滤过小的 chunk 或合并它们
        chunks = self._postprocess(chunks)
        return chunks

    def split_from_sections(
        self,
        sections: List[Any],  # List[Section] from ParsedDocument
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        从 ParsedDocument.sections 直接分块。
        每个 section 可能被递归拆分为多个 chunk（若超过 chunk_size）。
        """
        base_meta = metadata or {}
        all_chunks = []
        chunk_idx = 0

        for sec in sections:
            sec_meta = {
                **base_meta,
                "section_index": sec.index,
                "section_label": sec.label,
            }
            subs = self.split(sec.text, sec_meta)
            for sub in subs:
                sub.chunk_index = chunk_idx
                chunk_idx += 1
            all_chunks.extend(subs)

        return all_chunks

    # ── 各策略实现 ───────────────────────────────────────

    def _split_fixed(self, text: str, meta: dict) -> List[Chunk]:
        """固定窗口 + 重叠"""
        chunks = []
        step = self.chunk_size - self.overlap
        if step <= 0:
            step = self.chunk_size // 2

        idx = 0
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**meta, "chunk_method": "fixed"},
                    chunk_index=idx,
                    char_start=start,
                    char_end=end,
                ))
                idx += 1
            start += step

        return chunks

    def _split_recursive(self, text: str, meta: dict) -> List[Chunk]:
        """递归按分隔符优先级拆分"""
        return self._recursive_split(text, self.separators, meta, 0)

    def _recursive_split(
        self, text: str, seps: list, meta: dict, start_offset: int
    ) -> List[Chunk]:
        """递归核心：尝试当前分隔符，若仍超长则用下一级"""
        # 找到第一个能拆分的分隔符
        for i, sep in enumerate(seps):
            if sep == "":
                # 最后手段：逐字符
                return self._split_fixed(
                    text, {**meta, "chunk_method": f"recursive_char"}
                )

            parts = text.split(sep)
            if len(parts) == 1:
                continue  # 这个分隔符没匹配，换下一个

            # 合并回来，每段不超过 chunk_size
            chunks = []
            buf = []
            buf_len = 0
            chunk_idx = 0
            current_offset = start_offset

            for part in parts:
                # 加上分隔符长度（除了第一段）
                part_with_sep = part if not buf else sep + part
                part_len = len(part_with_sep)

                if buf_len + part_len <= self.chunk_size:
                    buf.append(part)
                    buf_len += part_len
                else:
                    # 当前 buf 已满，保存
                    if buf:
                        merged = sep.join(buf) if i > 0 else "".join(buf)
                        chunks.append(Chunk(
                            text=merged,
                            metadata={**meta, "chunk_method": f"recursive_{seps[0]}"},
                            chunk_index=chunk_idx,
                            char_start=current_offset,
                            char_end=current_offset + len(merged),
                        ))
                        current_offset += len(merged) + len(sep)  # +分隔符
                        chunk_idx += 1

                    # 处理新部分：若仍超长，递归下一级
                    if len(part) > self.chunk_size:
                        sub_chunks = self._recursive_split(
                            part, seps[i + 1:], meta, current_offset
                        )
                        # 修正 chunk_index
                        for sc in sub_chunks:
                            sc.chunk_index = chunk_idx
                            chunk_idx += 1
                        chunks.extend(sub_chunks)
                        current_offset += len(part) + len(sep)
                        buf = []
                        buf_len = 0
                    else:
                        buf = [part]
                        buf_len = len(part)

            # 收尾
            if buf:
                merged = sep.join(buf) if i > 0 else "".join(buf)
                chunks.append(Chunk(
                    text=merged,
                    metadata={**meta, "chunk_method": f"recursive_{seps[0]}"},
                    chunk_index=chunk_idx,
                    char_start=current_offset,
                    char_end=current_offset + len(merged),
                ))

            return self._merge_small(chunks)

        # 所有分隔符都无法拆分（单段就超长），退化为 fixed
        return self._split_fixed(text, {**meta, "chunk_method": "recursive_fallback"})

    def _split_sentence(self, text: str, meta: dict) -> List[Chunk]:
        """按句子边界切分，保持语义完整"""
        # 中英文句子分割
        sentences = re.split(
            r'(?<=[。！？.!?\n])\s*',
            text,
        )
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        buf = []
        buf_len = 0
        chunk_idx = 0
        current_offset = 0

        for sent in sentences:
            if buf_len + len(sent) <= self.chunk_size:
                buf.append(sent)
                buf_len += len(sent)
            else:
                if buf:
                    txt = "".join(buf)
                    chunks.append(Chunk(
                        text=txt,
                        metadata={**meta, "chunk_method": "sentence"},
                        chunk_index=chunk_idx,
                        char_start=current_offset,
                        char_end=current_offset + len(txt),
                    ))
                    current_offset += len(txt)
                    chunk_idx += 1
                    buf = []
                    buf_len = 0

                # 单句超长：退化为 recursive
                if len(sent) > self.chunk_size:
                    sub = self._split_recursive(sent, self.separators, meta, current_offset)
                    for sc in sub:
                        sc.chunk_index = chunk_idx
                        chunk_idx += 1
                    chunks.extend(sub)
                    current_offset += len(sent)
                else:
                    buf = [sent]
                    buf_len = len(sent)

        if buf:
            txt = "".join(buf)
            chunks.append(Chunk(
                text=txt,
                metadata={**meta, "chunk_method": "sentence"},
                chunk_index=chunk_idx,
                char_start=current_offset,
                char_end=current_offset + len(txt),
            ))

        return self._merge_small(chunks)

    def _split_markdown(self, text: str, meta: dict) -> List[Chunk]:
        """感知 Markdown 结构：标题作为分段锚点"""
        # 按 Markdown 标题切分
        sections = re.split(r'\n(?=#{1,6}\s)', text)

        chunks = []
        chunk_idx = 0
        current_offset = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # 提取标题作为 metadata
            title_match = re.match(r'^(#{1,6})\s+(.+)', section)
            sec_meta = {**meta}
            if title_match:
                level = len(title_match.group(1))
                title = title_match.group(2)
                sec_meta["heading_level"] = level
                sec_meta["heading"] = title

            if len(section) <= self.chunk_size:
                chunks.append(Chunk(
                    text=section,
                    metadata={**sec_meta, "chunk_method": "markdown"},
                    chunk_index=chunk_idx,
                    char_start=current_offset,
                    char_end=current_offset + len(section),
                ))
                chunk_idx += 1
            else:
                # 过长则递归拆分，但保留标题元信息
                sub_chunks = self._split_recursive(section, sec_meta)
                for sc in sub_chunks:
                    sc.metadata.setdefault("heading", sec_meta.get("heading", ""))
                    sc.metadata.setdefault("chunk_method", "markdown_recursive")
                    sc.chunk_index = chunk_idx
                    chunk_idx += 1
                chunks.extend(sub_chunks)

            current_offset += len(section) + 1

        return self._merge_small(chunks)

    def _split_sections(self, text: str, meta: dict) -> List[Chunk]:
        """将输入文本当作一个 section 处理"""
        if len(text) <= self.chunk_size:
            return [Chunk(
                text=text,
                metadata={**meta, "chunk_method": "section"},
                chunk_index=0,
                char_start=0,
                char_end=len(text),
            )]
        return self._split_recursive(text, self.separators, meta, 0)

    # ── 后处理 ───────────────────────────────────────────

    def _postprocess(self, chunks: List[Chunk]) -> List[Chunk]:
        """合并过小 chunk + 添加 overlap"""
        chunks = self._merge_small(chunks)
        chunks = [c for c in chunks if c.text.strip()]
        if self.overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        return chunks

    def _merge_small(self, chunks: List[Chunk]) -> List[Chunk]:
        """合并过小的 chunk 到前一个"""
        if not chunks:
            return chunks

        merged = []
        for c in chunks:
            if len(c.text) < self.min_chunk_size and merged:
                # 合并到前一个
                prev = merged[-1]
                prev.text = prev.text + " " + c.text
                prev.char_end = c.char_end
            else:
                merged.append(c)

        return merged

    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        为相邻 chunk 之间补充 overlap：
        每个 chunk 尾部追加下一个 chunk 的前 N 个字符，
        保证块边界处的语义在相邻 chunk 中都可见。
        fixed 策略跳过（它自己通过 step 已经处理了 overlap）。
        """
        for i in range(len(chunks) - 1):
            # 跳过已有 overlap 的（fixed 策略标记）
            if chunks[i].metadata.get("chunk_method") == "fixed":
                continue
            next_text = chunks[i + 1].text
            tail = next_text[:self.overlap]
            if tail:
                chunks[i].text = chunks[i].text + tail
                chunks[i].char_end = chunks[i].char_start + len(chunks[i].text)
        return chunks


