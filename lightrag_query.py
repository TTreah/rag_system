"""
lightrag_query.py — LightRAG 知识图谱查询工具
通过 LightRAG Server REST API 查询已构建的知识图谱，返回大模型生成的答案 + 引用来源。

使用方式:
    python lightrag_query.py                              # 交互式查询
    python lightrag_query.py "Lucia Bramati 的成绩"        # 直接传 query
    python lightrag_query.py "..." --mode local           # 指定查询模式
    python lightrag_query.py "..." --raw                  # 输出原始结构化数据(实体/关系/chunk)

交互命令:
    /mode <mode>  切换查询模式
    /raw          切换输出：大模型回答 ↔ 原始结构化数据

查询模式 (mode):
    mix     - 混合检索（默认，综合 local + global + naive）
    local   - 实体聚焦：从相关实体展开，适合具体人物/事件
    global  - 关系聚焦：从全局关系展开，适合宏观主题
    naive   - 朴素 RAG：纯向量相似度检索
    hybrid  - 混合（local + global）

查询举例：Giorgia Pellizotti 是谁？

"""

import os
import sys
import json
import urllib.request
import urllib.error

# Windows 控制台统一 UTF-8 输出，避免中文/特殊字符乱码（VSCode 终端默认 UTF-8）
if sys.stdout and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
#  连接配置（按需修改）
# ============================================================
LIGHTRAG_URL = "http://localhost:9621"
QUERY_ENDPOINT = f"{LIGHTRAG_URL}/query"
QUERY_DATA_ENDPOINT = f"{LIGHTRAG_URL}/query/data"

DEFAULT_MODE = "mix"
DEFAULT_TOP_K = 20
DEFAULT_CHUNK_TOP_K = 20
DEFAULT_RESPONSE_TYPE = None          # None = 默认；可选 "Bullet Points" / "Single Paragraph" / "Multiple Paragraphs"

# ============================================================


class LightRAGClient:
    """LightRAG Server 的轻量 HTTP 客户端（无第三方依赖，仅用标准库）。"""

    def __init__(self, base_url: str = LIGHTRAG_URL):
        self.base_url = base_url.rstrip("/")
        self.query_url = f"{self.base_url}/query"

    def query(
        self,
        query: str,
        mode: str = DEFAULT_MODE,
        top_k: int = DEFAULT_TOP_K,
        chunk_top_k: int = DEFAULT_CHUNK_TOP_K,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        include_references: bool = True,
    ) -> dict:
        """发起一次知识图谱查询。

        Returns:
            {"response": str, "references": [{"reference_id", "file_path", "content"}]}
        """
        payload = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
            "include_references": include_references,
        }
        if response_type:
            payload["response_type"] = response_type

        req = urllib.request.Request(
            self.query_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"无法连接 LightRAG 服务 ({self.base_url}): {e.reason}") from e

    def query_data(
        self,
        query: str,
        mode: str = DEFAULT_MODE,
        top_k: int = DEFAULT_TOP_K,
        chunk_top_k: int = DEFAULT_CHUNK_TOP_K,
    ) -> dict:
        """查询知识图谱的原始结构化数据（不经 LLM 加工）。

        走 /query/data 端点，返回图谱检索到的实体/关系/文本块。

        Returns:
            {"entities": [...], "relationships": [...],
             "chunks": [...], "references": [...]}
        """
        payload = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
        }
        req = urllib.request.Request(
            f"{self.base_url}/query/data",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"无法连接 LightRAG 服务 ({self.base_url}): {e.reason}") from e
        return body.get("data", body)

    def health(self) -> dict:
        """返回服务健康状态（用于测试连通性）。"""
        with urllib.request.urlopen(f"{self.base_url}/health", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ============================================================
#  交互入口
# ============================================================

def _safe(s) -> str:
    """输出兜底（UTF-8 下无需转换，仅防御 None/非字符串）。"""
    if s is None:
        return ""
    return str(s)


def _print_result(data: dict, mode: str):
    """打印一次查询的答案 + 引用。"""
    print("\n" + "=" * 70)
    print(f"[回答] (mode={mode})")
    print("-" * 70)
    print(_safe(data.get("response", "(空)")))
    print()

    refs = data.get("references") or []
    if refs:
        print(f"[引用来源] 共 {len(refs)} 条")
        print("-" * 70)
        for i, ref in enumerate(refs):
            path = _safe(ref.get("file_path", ""))
            name = os.path.basename(path) if path else "(未知)"
            print(f"  [{i + 1}] {name}")
        print()


def _print_raw_result(data: dict, mode: str):
    """打印结构化检索结果（实体/关系/文本块，不经 LLM 加工）。"""
    ents = data.get("entities") or []
    rels = data.get("relationships") or []
    chunks = data.get("chunks") or []
    refs = data.get("references") or []

    print("\n" + "=" * 70)
    print(f"[结构化检索结果] (mode={mode})  实体={len(ents)}  关系={len(rels)}  "
          f"chunk={len(chunks)}  引用={len(refs)}")
    print("-" * 70)

    if ents:
        print("\n── 实体 ──")
        for i, e in enumerate(ents):
            print(f"  [{i + 1}] {_safe(e.get('entity_name', ''))}  "
                  f"({_safe(e.get('entity_type', ''))})")
            desc = _safe(e.get("description", ""))
            if desc:
                print(f"      {desc[:120]}")

    if rels:
        print("\n── 关系 ──")
        for i, r in enumerate(rels):
            w = r.get("weight")
            print(f"  [{i + 1}] {_safe(r.get('src_id', ''))}  →  "
                  f"{_safe(r.get('tgt_id', ''))}  (weight={w})")
            desc = _safe(r.get("description", ""))
            if desc:
                print(f"      {desc[:120]}")

    if chunks:
        print("\n── 文本块 ──")
        for i, c in enumerate(chunks):
            text = _safe(c.get("content", "") or c.get("text", ""))
            print(f"  [{i + 1}] {text[:120]}")
    print()


VALID_MODES = ("mix", "local", "global", "naive", "hybrid")


def _parse_args(argv):
    """解析 query / --mode <mode> / --raw，返回 (query, mode, raw)。

    与参数顺序无关：`--mode local "查询"` 与 `"查询" --mode local` 等价。
    旧实现取 argv[:idx] 作为 query，导致 --mode 写在前面时 query 被静默丢弃、
    直接掉进交互模式。
    """
    raw = "--raw" in argv
    mode = None
    rest = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--raw":
            i += 1
        elif arg == "--mode":
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise SystemExit(f"--mode 缺少取值，可选: {', '.join(VALID_MODES)}")
            mode = argv[i + 1]
            i += 2
        elif arg.startswith("--mode="):
            mode = arg.split("=", 1)[1]
            i += 1
        else:
            rest.append(arg)
            i += 1

    raw = "--raw" in argv
    if mode is not None and mode not in VALID_MODES:
        raise SystemExit(f"未知 mode: {mode}，可选: {', '.join(VALID_MODES)}")

    return (" ".join(rest) if rest else None), mode, raw


if __name__ == "__main__":
    client = LightRAGClient()

    # 启动前检查连通性
    try:
        health = client.health()
        print(f"已连接 LightRAG: {_safe(health.get('webui_title', ''))} "
              f"(v{health.get('core_version', '?')}, 鉴权={health.get('auth_mode', '?')})")
        print(f"  存储目录: {_safe(health.get('working_directory', ''))}")
    except Exception as e:
        print(f"[错误] 无法连接 LightRAG 服务: {e}")
        print(f"       请确认服务运行在 {LIGHTRAG_URL}")
        sys.exit(1)

    query_str, mode, raw_mode = _parse_args(sys.argv[1:])
    current_mode = mode or DEFAULT_MODE

    # ── 命令行传参 → 单次查询 ──
    if query_str:
        try:
            if raw_mode:
                data = client.query_data(query_str, mode=current_mode)
                _print_raw_result(data, current_mode)
            else:
                data = client.query(query_str, mode=current_mode)
                _print_result(data, current_mode)
        except Exception as e:
            print(f"[错误] {e}")
            sys.exit(1)
        sys.exit(0)

    # ── 交互式循环 ──
    print()
    print("知识图谱查询 (输入空行退出)")
    print(f"当前模式: {current_mode} | 输出: {'结构化数据' if raw_mode else '大模型回答'}")
    print("可用命令:")
    print("  /mode <mode>  切换模式 (mix / local / global / naive / hybrid)")
    print("  /raw          切换输出：大模型回答 ↔ 原始结构化数据(实体/关系/chunk)")
    print("  /help         显示帮助")

    while True:
        line = input("\n请输入查询: ").strip()
        if not line:
            break

        if line == "/help":
            print("模式: mix(默认,综合) / local(实体) / global(关系) / naive(向量) / hybrid(local+global)")
            print("/raw: 切换输出方式（大模型回答 ↔ 原始结构化数据）")
            continue

        if line == "/raw":
            raw_mode = not raw_mode
            print(f"已切换到 {'结构化数据' if raw_mode else '大模型回答'} 模式")
            continue

        if line.startswith("/mode"):
            parts = line.split()
            if len(parts) == 2 and parts[1] in VALID_MODES:
                current_mode = parts[1]
                print(f"已切换到模式: {current_mode}")
            else:
                print(f"用法: /mode <{'|'.join(VALID_MODES)}>")
            continue

        try:
            if raw_mode:
                data = client.query_data(line, mode=current_mode)
                _print_raw_result(data, current_mode)
            else:
                data = client.query(line, mode=current_mode)
                _print_result(data, current_mode)
        except Exception as e:
            print(f"[错误] {e}")
