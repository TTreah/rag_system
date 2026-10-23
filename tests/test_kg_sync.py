"""
test_kg_sync.py — 知识图谱增量同步的回归测试

用内存 FakeClient 模拟 LightRAG Server，不连真实服务、不调 LLM，秒跑。
重点覆盖会造成数据损坏或多烧 token 的分支：
  1. 抽取成功的文件必须写清单（含 doc_id），失败的绝不能写
  2. 修改过的文件必须先删旧文档再上传（否则 409）
  3. 同名冲突（409）要删掉服务端残档后重传
  4. 不同目录的同名文件要加前缀区分，不能互相覆盖
  5. 删除的文件要从服务端和清单同时消失
  6. 改名只动清单，不重复上传（省 LLM 抽取）
  7. dry-run 不能碰任何数据

运行: python test_kg_sync.py
"""

import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
del _sys


import os
import sys
import json
import hashlib
import tempfile
import shutil

from kg_line import kg_sync
from kg_line.kg_sync import KGSynchronizer
from kg_line.kg_docs_client import KGApiError
from kg_line.kg_manifest import KgManifest

# 测试不等真实轮询间隔
kg_sync.POLL_INTERVAL = 0

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


# ═══════════════════════════════════════════════════════════════
# 内存版 LightRAG 服务端
# ═══════════════════════════════════════════════════════════════

class FakeClient:
    """模拟 LightRAG Server 文档 API。

    行为对齐已核实的服务端语义：
      - 同名上传 → KGApiError(409)
      - 删除是异步的，但在测试里立即完成
      - 文件名含 "bad" 的模拟抽取失败（status=failed）
    """

    def __init__(self):
        self.docs = {}          # doc_id -> doc dict
        self.upload_calls = []  # (file_path或text, upload_name)
        self.delete_calls = []  # [doc_id...]

    def _register(self, name, content: bytes):
        if any(d["file_path"] == name for d in self.docs.values()):
            raise KGApiError(f"409 Conflict: {name}", status=409)
        doc_id = "doc-" + hashlib.md5(content).hexdigest()[:12]
        track_id = "track-" + doc_id
        status = "failed" if "bad" in name else "processed"
        self.docs[doc_id] = {
            "id": doc_id,
            "file_path": name,
            "status": status,
            "track_id": track_id,
            "chunks_count": 3 if status == "processed" else 0,
            "error_msg": "simulated failure" if status == "failed" else None,
        }
        return {"status": "success", "message": "", "track_id": track_id}

    def list_documents(self, page_size=200):
        return list(self.docs.values())

    def upload(self, file_path, upload_name=None):
        name = upload_name or os.path.basename(file_path)
        self.upload_calls.append((file_path, name))
        with open(file_path, "rb") as f:
            return self._register(name, f.read())

    def insert_text(self, text, file_source=None):
        name = file_source or "text"
        self.upload_calls.append(("(text)", name))
        return self._register(name, text.encode("utf-8"))

    def delete_documents(self, doc_ids, delete_file=True):
        self.delete_calls.extend(doc_ids)
        for i in doc_ids:
            self.docs.pop(i, None)
        return [{"status": "deletion_started", "doc_id": i} for i in doc_ids]

    def track_status(self, track_id):
        docs = [d for d in self.docs.values() if d.get("track_id") == track_id]
        summary = {}
        for d in docs:
            summary[d["status"]] = summary.get(d["status"], 0) + 1
        return {"track_id": track_id, "documents": docs,
                "total_count": len(docs), "status_summary": summary}

    def status_counts(self):
        counts = {"all": len(self.docs)}
        for d in self.docs.values():
            counts[d["status"]] = counts.get(d["status"], 0) + 1
        return counts

    def clear_all(self):
        self.docs.clear()
        return {"status": "success"}


def make_env():
    """临时目录 + FakeClient + 清单，返回 (root, client, manifest, sync)。"""
    tmp = tempfile.mkdtemp(prefix="kg_test_")
    client = FakeClient()
    manifest = KgManifest(os.path.join(tmp, "kg_manifest.json"))
    sync = KGSynchronizer(client, manifest, fingerprint="test-fp")
    return tmp, client, manifest, sync


def write(root, rel, content):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

def test_manifest_roundtrip():
    hr("清单读写（原子替换）")
    tmp = tempfile.mkdtemp(prefix="kg_mani_")
    path = os.path.join(tmp, "m.json")
    m = KgManifest(path)
    m.upsert("/a/b.pdf", {"source_file": "/a/b.pdf", "doc_id": "doc-1",
                          "file_hash": "h1", "size": 1, "mtime": 2,
                          "pipeline_fingerprint": "fp"})
    m.save()

    m2 = KgManifest(path)
    recs = m2.load()
    check(recs.get("/a/b.pdf", {}).get("doc_id") == "doc-1",
          "清单保存后能读回")
    m2.rename_path("/a/b.pdf", "/a/c.pdf")
    m2.save()
    m3 = KgManifest(path).load()
    check("/a/c.pdf" in m3 and "/a/b.pdf" not in m3,
          "改名后记录挪到新路径且 doc_id 保留",
          str(m3))
    shutil.rmtree(tmp, ignore_errors=True)


def test_first_ingest():
    hr("首次入库：成功写清单，失败不写")
    tmp, client, manifest, sync = make_env()
    write(tmp, "a.txt", "alpha content")
    write(tmp, "b.txt", "beta content")
    write(tmp, "bad.txt", "will fail")

    report = sync.run([tmp])
    recs = manifest.load()

    check(report.uploaded == 2, "2 个成功入库", f"uploaded={report.uploaded}")
    check(len(report.failed) == 1, "1 个失败", str(report.failed))
    check(len(recs) == 2, "清单只记成功的 2 个", str(list(recs)))
    ok = [r for r in recs.values() if r["doc_id"].startswith("doc-")]
    check(len(ok) == 2, "清单记录带 doc_id")
    # 服务端文档列表应包含 3 个（含 failed）
    check(len(client.docs) == 3, "服务端含 failed 文档共 3 个")

    # 失败文件下次运行仍是"新增"，自动重试
    report2 = sync.run([tmp])
    check(len(report2.plan.added) == 1 and "bad.txt" in report2.plan.added[0],
          "失败文件下次仍判为新增（自动重试）", str(report2.plan.added))
    shutil.rmtree(tmp, ignore_errors=True)


def test_no_change_no_upload():
    hr("未变化：完全不碰")
    tmp, client, manifest, sync = make_env()
    write(tmp, "a.txt", "alpha")
    sync.run([tmp])
    client.upload_calls.clear()

    report = sync.run([tmp])
    check(not client.upload_calls, "没有新上传")
    check(not client.delete_calls, "没有删除")
    check(report.plan.unchanged == 1, "1 个文件未变化")
    shutil.rmtree(tmp, ignore_errors=True)


def test_modified_delete_before_upload():
    hr("修改：先删旧文档再上传")
    tmp, client, manifest, sync = make_env()
    p = write(tmp, "a.txt", "version 1")
    sync.run([tmp])
    old_doc_ids = set(client.docs.keys())

    write(tmp, "a.txt", "version 2 changed")
    # mtime 可能没变（同秒内写两次），但 size 变了必然被判修改
    report = sync.run([tmp])

    check(len(report.plan.modified) == 1, "判为修改", str(report.plan))
    check(set(client.delete_calls) == old_doc_ids, "旧文档已删除")
    check(len(client.docs) == 1, "服务端只剩 1 个文档")
    new_doc = list(client.docs.values())[0]
    check(new_doc["id"] not in old_doc_ids, "新文档 doc_id 不同（内容变了）")
    rec = manifest.load()[kg_sync_canonical(p)]
    check(rec["doc_id"] == new_doc["id"], "清单 doc_id 已更新")
    shutil.rmtree(tmp, ignore_errors=True)


def kg_sync_canonical(p):
    from qdrant_line.vectorize_pipeline import canonical_path
    return canonical_path(p)


def test_deleted_file():
    hr("删除：服务端与清单同时清理")
    tmp, client, manifest, sync = make_env()
    pa = write(tmp, "a.txt", "alpha")
    write(tmp, "b.txt", "beta")
    sync.run([tmp])
    os.unlink(pa)

    report = sync.run([tmp])
    check(len(report.plan.deleted) == 1, "判为删除")
    check(len(client.docs) == 1, "服务端删了 1 个", str(len(client.docs)))
    check(len(manifest.load()) == 1, "清单删了 1 个")
    check(not any(d["file_path"] == "a.txt" for d in client.docs.values()),
          "a.txt 已从服务端消失")
    shutil.rmtree(tmp, ignore_errors=True)


def test_name_collision():
    hr("同名文件：加前缀区分")
    tmp, client, manifest, sync = make_env()
    write(tmp, "dir1/report.txt", "from dir1")
    write(tmp, "dir2/report.txt", "from dir2")
    report = sync.run([tmp])

    check(report.uploaded == 2, "两个同名文件都入库")
    names = sorted(d["file_path"] for d in client.docs.values())
    check("report.txt" in names, "第一个保留原名", str(names))
    check(any(n.endswith("__report.txt") for n in names),
          "第二个加了哈希前缀", str(names))
    recs = manifest.load()
    upload_names = {r["upload_name"] for r in recs.values()}
    check(len(upload_names) == 2, "清单里 upload_name 各不相同")
    shutil.rmtree(tmp, ignore_errors=True)


class FakeClient409(FakeClient):
    """模拟竞态：分配上传名时服务端还没有冲突文档，
    真正上传的那一刻同名文档才被别的途径塞进来（WebUI 手动上传等）。
    第一次上传该名字时：先落一个残档，再抛 409。"""

    def __init__(self, conflict_name):
        super().__init__()
        self.conflict_name = conflict_name
        self._conflicted = False

    def upload(self, file_path, upload_name=None):
        name = upload_name or os.path.basename(file_path)
        self._maybe_conflict(name, file_path)
        return super().upload(file_path, upload_name=upload_name)

    def insert_text(self, text, file_source=None):
        name = file_source or "text"
        self._maybe_conflict(name, "(text)")
        return super().insert_text(text, file_source=file_source)

    def _maybe_conflict(self, name, tag):
        if name == self.conflict_name and not self._conflicted:
            self._conflicted = True
            self.upload_calls.append((tag, name))  # 失败的上传尝试也记录
            self.docs["doc-stale"] = {
                "id": "doc-stale", "file_path": name, "status": "processed",
                "track_id": "track-stale", "chunks_count": 1, "error_msg": None,
            }
            raise KGApiError(f"409 Conflict: {name}", status=409)


def test_conflict_retry():
    hr("409 冲突（竞态）：删残档重传")
    tmp = tempfile.mkdtemp(prefix="kg_test_")
    p = write(tmp, "report.txt", "content")

    client = FakeClient409(conflict_name="report.txt")
    manifest = KgManifest(os.path.join(tmp, "kg_manifest.json"))
    sync = KGSynchronizer(client, manifest, fingerprint="test-fp")

    report = sync.run([tmp])
    check(report.uploaded == 1, "重传后成功入库", f"uploaded={report.uploaded}")
    check("doc-stale" in client.delete_calls, "409 后删掉了同名残档",
          str(client.delete_calls))
    check(len(client.upload_calls) == 2, "上传了两次（首次 409 + 重传）",
          str(client.upload_calls))
    check(len(manifest.load()) == 1, "成功文件进了清单")
    shutil.rmtree(tmp, ignore_errors=True)


def test_rename_manifest_only():
    hr("改名：只动清单，不重复抽取")
    tmp, client, manifest, sync = make_env()
    pa = write(tmp, "old.txt", "same content")
    sync.run([tmp])
    pb = os.path.join(tmp, "new.txt")
    os.rename(pa, pb)
    client.upload_calls.clear()
    client.delete_calls.clear()

    report = sync.run([tmp], detect_renames=True)
    check(len(report.plan.renamed) == 1, "识别出改名", str(report.plan.renamed))
    check(not client.upload_calls, "没有重新上传（省了 LLM 抽取）")
    recs = manifest.load()
    new_key = kg_sync_canonical(pb)
    check(new_key in recs, "清单已挪到新路径", str(list(recs)))
    shutil.rmtree(tmp, ignore_errors=True)


def test_dry_run():
    hr("dry-run：不改任何数据")
    tmp, client, manifest, sync = make_env()
    write(tmp, "a.txt", "alpha")
    report = sync.run([tmp], dry_run=True)
    check(len(report.plan.added) == 1, "报告了新增")
    check(not client.upload_calls, "没有上传")
    check(len(manifest.load()) == 0, "清单没写")
    shutil.rmtree(tmp, ignore_errors=True)


def test_delete_ratio_fuse():
    hr("删除比例熔断")
    tmp, client, manifest, sync = make_env()
    for i in range(4):
        write(tmp, f"f{i}.txt", f"content {i}")
    sync.run([tmp])
    # 删掉 3/4 的文件（75% > 30% 阈值）
    for i in range(3):
        os.unlink(os.path.join(tmp, f"f{i}.txt"))

    try:
        sync.run([tmp])
        check(False, "应该触发 SafetyAbort")
    except kg_sync.SafetyAbort:
        check(True, "超过 30% 删除被熔断")
    check(len(client.docs) == 4, "熔断后服务端没动")

    report = sync.run([tmp], force=True)
    check(report.deleted == 3, "--force 后删除生效")
    shutil.rmtree(tmp, ignore_errors=True)


def test_doc_conversion():
    hr(".doc 转换：以 .docx 名义上传，清单仍记原路径")
    tmp, client, manifest, sync = make_env()
    p = write(tmp, "spec.doc", "legacy doc content")

    # 模拟 textutil 转换：复制一份改后缀（真实转换由 document_parser 负责，
    # 这里只验证 kg_sync 的接线：转换、上传名、临时文件清理）
    original = kg_sync._maybe_convert_doc
    def fake_convert(path):
        assert path.endswith(".doc")
        tmp_docx = path + ".converted.docx"
        shutil.copyfile(path, tmp_docx)
        return tmp_docx, tmp_docx
    kg_sync._maybe_convert_doc = fake_convert
    try:
        report = sync.run([tmp])
    finally:
        kg_sync._maybe_convert_doc = original

    check(report.uploaded == 1, ".doc 文件入库成功")
    check(client.upload_calls[0][1] == "spec.docx",
          "上传名是 .docx", str(client.upload_calls))
    check(not os.path.exists(p + ".converted.docx"), "临时转换文件已清理")
    rec = manifest.load()[kg_sync_canonical(p)]
    check(rec["source_file"].endswith("spec.doc"), "清单仍记原始 .doc 路径")
    check(rec["upload_name"] == "spec.docx", "清单记录了实际上传名")
    shutil.rmtree(tmp, ignore_errors=True)


def test_only_files_partial_sync():
    hr("部分同步（only_files）：删除判定关闭")
    tmp, client, manifest, sync = make_env()
    pa = write(tmp, "a.txt", "alpha")
    pb = write(tmp, "b.txt", "beta")
    sync.run([tmp])  # 两个都入库

    # 磁盘上删掉 b.txt，但部分同步只看 a.txt → b 的旧文档必须保留
    os.unlink(pb)
    report = sync.run([tmp], only_files=[pa])
    check(len(report.plan.deleted) == 0, "部分同步不判定删除",
          str(report.plan.deleted))
    check(len(client.docs) == 2, "b 的服务端文档还在", str(len(client.docs)))
    check(len(manifest.load()) == 2, "b 的清单记录还在")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_manifest_roundtrip()
    test_first_ingest()
    test_no_change_no_upload()
    test_modified_delete_before_upload()
    test_deleted_file()
    test_name_collision()
    test_conflict_retry()
    test_rename_manifest_only()
    test_dry_run()
    test_delete_ratio_fuse()
    test_doc_conversion()
    test_only_files_partial_sync()

    print(f"\n{'=' * 65}")
    if FAILURES:
        print(f"  {len(FAILURES)} 个测试失败:")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  全部通过 ✓")
    sys.exit(0)
