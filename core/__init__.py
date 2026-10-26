import os

# 本地服务（Qdrant/LightRAG/嵌入中介）必须绕过系统代理。
# macOS 开代理时 Python 的 urllib/httpx 会把 localhost 也代理出去导致 502。
for _k in ("NO_PROXY", "no_proxy"):
    _cur = os.environ.get(_k, "")
    for _host in ("localhost", "127.0.0.1", "::1"):
        if _host not in _cur:
            _cur = (_cur + "," + _host).strip(",")
    os.environ[_k] = _cur

