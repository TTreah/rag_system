"""
kg_docs_client.py — LightRAG 文档管理 API 客户端（知识图谱线）

只封装「文档生命周期」相关的端点，查询走 lightrag_query.py，两条线互不依赖。
纯标准库实现（urllib），不引入第三方依赖。

对照 Qdrant 线的分工：
    Qdrant 线   vectorize_pipeline.py / sync_index.py → 向量检索
    知识图谱线  kg_docs_client.py / kg_sync.py        → 实体关系图谱

已核实的服务端行为（LightRAG core 1.5.7 / api 0344）：
    - POST /documents/upload  一次只能传一个文件（multipart 字段名 "file"），
      文件会被拷进服务端 input 目录，服务端只认文件名，不认识客户端的目录结构
    - 同名冲突返回 HTTP 409（不是 200 duplicated），重传前必须先删旧文档
    - DELETE /documents/delete_document 是异步的（deletion_started），
      删除完成前同名文件仍然 409，所以删完要轮询确认
    - 文档状态机: pending → parsing → analyzing → processing → processed | failed
    - 处理是异步 LLM 抽取，upload 返回 track_id，用 track_status 轮询进度
"""

import os
import json
import uuid
import urllib.parse
import urllib.request
import urllib.error

DEFAULT_LIGHTRAG_URL = "http://localhost:9621"

# 终态：processed 成功 / failed 失败；其余都是中间态
TERMINAL_OK = "processed"
TERMINAL_BAD = "failed"
TERMINAL_STATES = {TERMINAL_OK, TERMINAL_BAD}


class KGApiError(RuntimeError):
    """LightRAG API 调用失败。status=None 表示网络层错误（服务没起来等）。"""

    def __init__(self, message: str, status: int = None):
        super().__init__(message)
        self.status = status

    @property
    def is_conflict(self) -> bool:
        """HTTP 409 — 同名文档已存在。"""
        return self.status == 409


class LightRAGDocsClient:
    """LightRAG Server 文档管理端点的薄封装。

    每个方法对应一个端点，返回解析后的 dict；HTTP 错误抛 KGApiError。
    """

    def __init__(self, base_url: str = DEFAULT_LIGHTRAG_URL,
                 api_key: str = None, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("LIGHTRAG_API_KEY")
        self.timeout = timeout

    # ── 状态 ─────────────────────────────────────────────

    def health(self) -> dict:
        return self._request("GET", "/health")

    def pipeline_status(self) -> dict:
        """返回 {"busy": bool, "job_name": str, "latest_message": str, ...}"""
        return self._request("GET", "/documents/pipeline_status")

    def status_counts(self) -> dict:
        """各状态文档数: {"processed": n, "failed": n, "all": n, ...}"""
        return self._request("GET", "/documents/status_counts").get("status_counts", {})

    # ── 文档列表 ─────────────────────────────────────────

    def list_documents(self, page_size: int = 200) -> list:
        """拉取全部文档（自动翻页）。

        Returns:
            [DocStatusResponse, ...]，字段含 id / file_path / status /
            chunks_count / error_msg / track_id
        """
        docs = []
        page = 1
        while True:
            resp = self._request("POST", "/documents/paginated", {
                "page": page,
                "page_size": max(page_size, 10),  # 服务端下限是 10
                "sort_field": "created_at",
                "sort_direction": "asc",
            })
            docs.extend(resp.get("documents", []))
            pagination = resp.get("pagination", {})
            if not pagination.get("has_next"):
                break
            page += 1
        return docs

    # ── 上传 ─────────────────────────────────────────────

    def insert_text(self, text: str, file_source: str = None) -> dict:
        """文本直传（用于入库前做过本地解析/预处理的文档）。

        file_source 即服务端的 file_path 字段，扮演和上传文件名相同的角色，
        同名去重/冲突规则与 /documents/upload 一致。
        """
        payload = {"text": text}
        if file_source:
            payload["file_source"] = file_source
        return self._request("POST", "/documents/text", payload)

    def upload(self, file_path: str, upload_name: str = None) -> dict:
        """上传单个文件，返回 {"status", "message", "track_id"}。

        同名（服务端 file_path 维度）已存在时抛 KGApiError(is_conflict=True)，
        调用方负责先删后传。
        """
        upload_name = upload_name or os.path.basename(file_path)
        with open(file_path, "rb") as f:
            data = f.read()

        # 手写 multipart/form-data，避免引入 requests 依赖。
        # 文件名同时给 filename 和 filename*（RFC 5987），中文名两头都不乱码。
        boundary = uuid.uuid4().hex
        quoted = urllib.parse.quote(upload_name)
        disposition = (
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{upload_name}"; filename*=UTF-8\'\'{quoted}\r\n'
        )
        body = b"".join([
            f"--{boundary}\r\n".encode(),
            disposition.encode("utf-8"),
            b"Content-Type: application/octet-stream\r\n\r\n",
            data,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ])
        return self._request(
            "POST", "/documents/upload", raw=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )

    # ── 删除 ─────────────────────────────────────────────

    def delete_documents(self, doc_ids: list, delete_file: bool = True) -> list:
        """按 doc_id 删除（异步，返回时删除可能还在进行）。

        delete_file=True 连同服务端 input 目录里的文件副本一起删，
        避免以后有人调 /documents/scan 把残留副本又扫回来。

        Returns:
            每个 doc_id 一条结果 [{"status": "deletion_started"|"busy"|..., ...}]
        """
        if not doc_ids:
            return []
        resp = self._request("DELETE", "/documents/delete_document", {
            "doc_ids": list(doc_ids),
            "delete_file": delete_file,
            "delete_llm_cache": False,
        })
        # 单个 id 返回 dict，多个返回 list，统一成 list
        return resp if isinstance(resp, list) else [resp]

    def track_status(self, track_id: str) -> dict:
        """{"track_id", "documents": [DocStatusResponse...], "status_summary": {...}}"""
        return self._request("GET", f"/documents/track_status/{track_id}")

    def clear_all(self) -> dict:
        """清空全部文档与图谱（--recreate 用，不可逆）。"""
        return self._request("DELETE", "/documents")

    # ── 内部 ─────────────────────────────────────────────

    def _request(self, method: str, path: str, payload: dict = None,
                 raw: bytes = None, content_type: str = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = {}
        data = raw
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            content_type = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise KGApiError(f"{method} {path} → HTTP {e.code}: {detail}",
                             status=e.code) from e
        except urllib.error.URLError as e:
            raise KGApiError(
                f"无法连接 LightRAG 服务 ({self.base_url}): {e.reason}"
            ) from e
