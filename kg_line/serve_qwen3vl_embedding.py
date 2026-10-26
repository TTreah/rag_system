"""
serve_qwen3vl_embedding.py — Qwen3-VL-Embedding 的 OpenAI 兼容嵌入服务

为什么需要它：LightRAG 服务端的嵌入走 OpenAI 兼容接口（EMBEDDING_BINDING=openai），
而 Ollama 不支持 VL 架构的嵌入（/api/embed 对 Qwen3-VL-Embedding 直接报错）。
这个脚本在本机起一个极简 HTTP 服务，把本地 transformers 加载的
Qwen3-VL-Embedding-2B 包装成 /v1/embeddings，LightRAG 容器经
host.docker.internal 访问它。

启动:
    python kg_line/serve_qwen3vl_embedding.py            # 默认端口 18001
    nohup python kg_line/serve_qwen3vl_embedding.py &    # 后台常驻

LightRAG .env 配套:
    EMBEDDING_BINDING=openai
    EMBEDDING_BINDING_HOST=http://host.docker.internal:18001/v1
    EMBEDDING_MODEL=Qwen3-VL-Embedding-2B
    EMBEDDING_DIM=2048
"""

import os
import sys
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embedder import Qwen3VLEmbedderAdapter, resolve_model_path

PORT = int(os.environ.get("QWEN3VL_EMBED_PORT", "18001"))
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("embed_server")

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        path = resolve_model_path(MODEL_NAME)
        logger.info("Loading %s from %s ...", MODEL_NAME, path)
        # 服务端必须 CPU：LightRAG 并发 8 路调嵌入，MPS 非线程安全会崩
        _model = Qwen3VLEmbedderAdapter(path)  # 默认走 MPS；请求串行锁在下面
        logger.info("Model ready. dim=%d device=%s",
                    _model.get_sentence_embedding_dimension(), _model.device)
    return _model


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if not self.path.rstrip("/").endswith("/v1/embeddings") \
                and self.path != "/v1/embeddings":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            inputs = body.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            model = get_model()
            # 即便 CPU 推理，也串行化模型调用，防御任何后端线程安全问题
            with _model_lock:
                vecs = model.encode(inputs, normalize_embeddings=True)
            data = [{"object": "embedding", "index": i,
                     "embedding": [float(x) for x in v]}
                    for i, v in enumerate(vecs)]
            resp = {"object": "list", "data": data,
                    "model": body.get("model", MODEL_NAME),
                    "usage": {"prompt_tokens": 0, "total_tokens": 0}}
            payload = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            logger.exception("embed failed")
            payload = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            payload = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        logger.info(fmt, *args)


if __name__ == "__main__":
    get_model()  # 启动时先加载，避免首个请求才付加载成本
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    logger.info("Qwen3-VL embedding service listening on 0.0.0.0:%d", PORT)
    server.serve_forever()
