"""
test_sync_index.py — 增量同步的差异规划回归测试

只测纯逻辑（plan_sync / scan_disk），不连 Qdrant、不加载嵌入模型，
因此可以随时秒跑。重点覆盖会造成数据损坏的分支：
  1. 改动过的文件必须进 paths_to_delete（否则残留"幽灵文本块"）
  2. 配置指纹变化必须触发全量重建
  3. 只有 mtime 变化时不能误判为修改（SMB 上 mtime 会抖），但必须去核对哈希
  4. 读不到的文件绝不能算作"已删除"
  5. 改名配对不能把内容不同的文件错配到一起

运行: python test_sync_index.py
"""

import os, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
del _sys


import os
import sys
import tempfile

from qdrant_line.sync_index import scan_disk, plan_sync, SyncPlan, DiskEntry
from qdrant_line.vectorize_pipeline import canonical_path

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


class HashCounter:
    """可注入的哈希函数：按内容返回稳定值，并记录被调用次数。"""

    def __init__(self, contents):
        self.contents = contents
        self.calls = []

    def __call__(self, path):
        self.calls.append(path)
        if path not in self.contents:
            raise OSError(f"cannot read {path}")
        return f"h:{self.contents[path]}"

    def reset(self):
        self.calls = []


FP = "fp-aaa"


def entry(path, size, mtime):
    return DiskEntry(path=path, size=size, mtime=mtime)


def record(path, content, size, mtime, fp=FP):
    return {
        "source_file": path,
        "file_hash": f"h:{content}",
        "size": size,
        "mtime": mtime,
        "pipeline_fingerprint": fp,
    }


# ─── 1. 新增 / 删除 / 未变化 ────────────────────────────────

def test_basic_diff():
    hr("1. 基础差异：新增 / 删除 / 未变化")

    h = HashCounter({"/a.txt": "A", "/b.txt": "B", "/c.txt": "C"})
    disk = {
        "/a.txt": entry("/a.txt", 10, 100),   # 清单里没有 → 新增
        "/b.txt": entry("/b.txt", 10, 100),   # 完全一致   → 未变化
    }
    manifest = {
        "/b.txt": record("/b.txt", "B", 10, 100),
        "/c.txt": record("/c.txt", "C", 10, 100),   # 磁盘没了 → 删除
    }

    plan = plan_sync(disk, manifest, FP, h)

    check(plan.added == ["/a.txt"], "新增文件被识别", f"got {plan.added}")
    check(plan.deleted == ["/c.txt"], "删除文件被识别", f"got {plan.deleted}")
    check(plan.modified == [], "无修改", f"got {plan.modified}")
    check(plan.unchanged == 1, "未变化计数正确", f"got {plan.unchanged}")
    check(h.calls == ["/a.txt"],
          "未变化的文件不读内容（只有新增文件算了哈希）", f"got {h.calls}")

    check(plan.to_ingest == ["/a.txt"], "待入库 = 新增 + 修改", f"got {plan.to_ingest}")
    check(plan.paths_to_delete == ["/c.txt"],
          "删除文件进待删列表", f"got {plan.paths_to_delete}")
    check(plan.has_changes, "有变化时 has_changes=True")


# ─── 2. 修改检测三层 ────────────────────────────────────────

def test_modification_detection():
    hr("2. 修改检测：size 变化 / 仅 mtime 变化 / 内容真变")

    # 2a. size 变了 → 直接判定修改，不必算哈希
    h = HashCounter({"/a.txt": "A-much-longer"})
    disk = {"/a.txt": entry("/a.txt", 99, 100)}
    manifest = {"/a.txt": record("/a.txt", "A", 10, 100)}
    plan = plan_sync(disk, manifest, FP, h)
    check(plan.modified == ["/a.txt"], "size 变化判为修改", f"got {plan.modified}")
    check(len(h.calls) == 1,
          "size 变化时仍取了一次哈希（要写进 payload）", f"got {h.calls}")

    # 2b. 只有 mtime 变了、内容没变 → 不算修改（SMB 上 mtime 会抖）
    h = HashCounter({"/a.txt": "A"})
    disk = {"/a.txt": entry("/a.txt", 10, 999)}       # mtime 变了
    manifest = {"/a.txt": record("/a.txt", "A", 10, 100)}
    plan = plan_sync(disk, manifest, FP, h)
    check(plan.modified == [], "仅 mtime 变化、内容未变 → 不算修改", f"got {plan.modified}")
    check(plan.unchanged == 1, "计入未变化", f"got {plan.unchanged}")
    check("/a.txt" in h.calls, "但确实去核对了哈希（第二层确认）", f"got {h.calls}")

    # 2c. 只有 mtime 变了、内容其实变了 → 必须判为修改（最危险的一种）
    h = HashCounter({"/a.txt": "A-CHANGED"})
    disk = {"/a.txt": entry("/a.txt", 10, 999)}
    manifest = {"/a.txt": record("/a.txt", "A", 10, 100)}
    plan = plan_sync(disk, manifest, FP, h)
    check(plan.modified == ["/a.txt"],
          "mtime 变化 + 哈希不同 → 判为修改（防漏更新）", f"got {plan.modified}")

    # 2d. 修改文件必须进待删列表，否则旧尾部 chunk 残留
    check("/a.txt" in plan.paths_to_delete,
          "修改文件进入待删列表（防止幽灵文本块）", f"got {plan.paths_to_delete}")


# ─── 3. 配置指纹变化 ────────────────────────────────────────

def test_fingerprint_change():
    hr("3. 分块/嵌入配置变化 → 全量重建")

    files = {f"/f{i}.txt": entry(f"/f{i}.txt", 10, 100) for i in range(3)}
    manifest = {f"/f{i}.txt": record(f"/f{i}.txt", "X", 10, 100) for i in range(3)}
    h = HashCounter({f"/f{i}.txt": "X" for i in range(3)})

    plan = plan_sync(files, manifest, FP, h)
    check(plan.modified == [], "指纹相同时不重建", f"got {plan.modified}")

    h2 = HashCounter({f"/f{i}.txt": "X" for i in range(3)})
    plan2 = plan_sync(files, manifest, "fp-DIFFERENT", h2)
    check(len(plan2.modified) == 3,
          "指纹变化 → 所有文件判为需重建（文件本身没变）", f"got {plan2.modified}")
    check(len(plan2.deleted) == 0, "指纹变化不产生删除")


# ─── 4. 读不到的文件 ────────────────────────────────────────

def test_unreadable_is_safe():
    hr("4. 读不到的文件 → 跳过，绝不当作已删除")

    h = HashCounter({"/ok.txt": "OK"})       # /locked.txt 不在字典里 → 抛 OSError
    disk = {
        "/ok.txt": entry("/ok.txt", 5, 100),       # 新增，可读
        "/locked.txt": entry("/locked.txt", 5, 100),  # 新增，但被占用读不了
    }
    plan = plan_sync(disk, {}, FP, h)

    check(plan.added == ["/ok.txt"], "可读的新增文件正常入库", f"got {plan.added}")
    check("/locked.txt" not in plan.deleted,
          "读不到的新文件不会被误删（它本来也不在清单里）")
    check(plan.unreadable == ["/locked.txt"],
          "读不到的文件记入 unreadable", f"got {plan.unreadable}")
    check("/locked.txt" not in plan.hashes, "读不到就不产生哈希")

    # 已入库的文件这周突然读不到 → 必须保留，不能删
    h2 = HashCounter({})
    disk2 = {"/x.txt": entry("/x.txt", 10, 999)}
    manifest2 = {"/x.txt": record("/x.txt", "X", 10, 100)}
    plan2 = plan_sync(disk2, manifest2, FP, h2)
    check(plan2.deleted == [], "读失败不会导致删除旧数据", f"got {plan2.deleted}")
    check(plan2.modified == [], "读失败也不重建（保持原状最安全）", f"got {plan2.modified}")
    check(plan2.unreadable == ["/x.txt"], "记入 unreadable 供人工排查")


# ─── 5. 改名检测 ────────────────────────────────────────────

def test_rename_detection():
    hr("5. 改名检测（仅哈希相同才配对）")

    # 5a. 内容相同、路径变了 → 配对为改名，不进 added/deleted
    h = HashCounter({"/new.txt": "SAME"})
    disk = {"/new.txt": entry("/new.txt", 10, 100)}
    manifest = {"/old.txt": record("/old.txt", "SAME", 10, 100)}
    plan = plan_sync(disk, manifest, FP, h, detect_renames=True)

    check(plan.renamed == [("/old.txt", "/new.txt")],
          "同内容改名被配对", f"got {plan.renamed}")
    check(plan.added == [], "改名项不进 added（省一次嵌入）", f"got {plan.added}")
    check(plan.deleted == [], "改名项不进 deleted（旧点要保留）", f"got {plan.deleted}")

    # 5b. 内容不同 → 不能配对，老老实实删旧的 + 加新的
    h2 = HashCounter({"/new.txt": "DIFFERENT"})
    plan2 = plan_sync(disk, manifest, FP, h2, detect_renames=True)
    check(plan2.renamed == [], "内容不同不配对", f"got {plan2.renamed}")
    check(plan2.added == ["/new.txt"], "不配对时按新增处理", f"got {plan2.added}")
    check(plan2.deleted == ["/old.txt"], "不配对时按删除处理", f"got {plan2.deleted}")

    # 5c. 默认关闭：不传就退化成 删+增
    h3 = HashCounter({"/new.txt": "SAME"})
    plan3 = plan_sync(disk, manifest, FP, h3)
    check(plan3.renamed == [], "默认不做改名检测", f"got {plan3.renamed}")

    # 5d. 两个同内容文件，一个改名一个真删 → 只配对一次
    h4 = HashCounter({"/new.txt": "DUP"})
    disk4 = {"/new.txt": entry("/new.txt", 3, 100)}
    manifest4 = {
        "/old1.txt": record("/old1.txt", "DUP", 3, 100),
        "/old2.txt": record("/old2.txt", "DUP", 3, 100),
    }
    plan4 = plan_sync(disk4, manifest4, FP, h4, detect_renames=True)
    check(len(plan4.renamed) == 1, "同内容只配对一次", f"got {plan4.renamed}")
    check(plan4.deleted == ["/old2.txt"],
          "多出来的同内容旧文件仍按删除处理", f"got {plan4.deleted}")


# ─── 6. 真实目录扫描 ────────────────────────────────────────

def test_scan_disk():
    hr("6. scan_disk：递归 / 过滤垃圾文件 / 路径规范化")

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "sub", "deep"))

        for rel in ["a.txt", "b.md", "sub/c.txt", "sub/deep/d.md",
                    "skip.pdf", "image.png"]:
            with open(os.path.join(tmp, rel), "w", encoding="utf-8") as f:
                f.write("x" * 10)

        # 这些都应该被忽略
        for rel in ["~$draft.docx", "half.tmp", "part.pdf.crdownload"]:
            with open(os.path.join(tmp, rel), "w", encoding="utf-8") as f:
                f.write("x")

        found = scan_disk(tmp, allowed_exts=[".txt", ".md"], recursive=True)
        names = sorted(os.path.basename(p) for p in found)

        check(names == ["a.txt", "b.md", "c.txt", "d.md"],
              "递归找到全部受支持文件，且只找受支持的", f"got {names}")

        check(all(os.path.isabs(p) for p in found),
              "返回的是绝对路径")
        check(not any("~$" in p or p.endswith(".tmp") for p in found),
              "Office 锁文件 / .tmp 被过滤")

        for p, e in found.items():
            check(e.size == 10, f"{os.path.basename(p)} 的 size 被读到", f"got {e.size}")
            break
        check(all(e.mtime > 0 for e in found.values()), "mtime 被读到")

        # 规范化：同一目录的不同写法必须映射到同一个 key
        alt = os.path.join(tmp, ".", "sub", "..")
        found2 = scan_disk(alt, allowed_exts=[".txt", ".md"], recursive=True)
        check(set(found2) == set(found),
              "路径经 canonical_path 规范化后 key 稳定", f"{len(found2)} vs {len(found)}")

        # 非递归只扫顶层
        found3 = scan_disk(tmp, allowed_exts=[".txt", ".md"], recursive=False)
        check(sorted(os.path.basename(p) for p in found3) == ["a.txt", "b.md"],
              "recursive=False 时只扫顶层")

        # 空目录不炸
        empty = os.path.join(tmp, "empty")
        os.makedirs(empty, exist_ok=True)
        check(scan_disk(empty, allowed_exts=[".txt"]) == {},
              "空目录返回空字典")


# ─── 7. 幂等性 ──────────────────────────────────────────────

def test_idempotent():
    hr("7. 幂等：同步完成后再次规划应无变化")

    # 模拟"已同步完"的状态：清单 = 磁盘现状
    h = HashCounter({"/a.txt": "A", "/b.txt": "B"})
    disk = {
        "/a.txt": entry("/a.txt", 10, 100),
        "/b.txt": entry("/b.txt", 20, 200),
    }
    manifest = {
        "/a.txt": record("/a.txt", "A", 10, 100),
        "/b.txt": record("/b.txt", "B", 20, 200),
    }

    plan = plan_sync(disk, manifest, FP, h)
    check(not plan.has_changes, "无变化", f"added={plan.added} modified={plan.modified}")
    check(plan.unchanged == 2, "全部计入未变化", f"got {plan.unchanged}")
    check(h.calls == [], "完全没有读取任何文件内容", f"got {h.calls}")


# ─── 8. 报告渲染 ────────────────────────────────────────────

def test_report_render():
    hr("8. SyncReport.describe() 不崩")

    from qdrant_line.sync_index import SyncReport

    plan = SyncPlan(added=["/x"], modified=["/y"], deleted=["/z"],
                    unreadable=["/u"], unchanged=7)
    r = SyncReport(plan=plan, chunks=42, files_ingested=2,
                   skipped_by_limit=5, dry_run=True)
    text = r.describe()
    check("DRY-RUN" in text, "dry-run 有醒目提示")
    check("42" not in text, "dry-run 不显示入库 chunk 数", text[:200])
    check(all(k in text for k in ["新增", "修改", "删除", "改名", "未变化"]),
          "各分类都在报告里")

    r2 = SyncReport(plan=SyncPlan(), dry_run=False)
    check("没有任何变化" in r2.describe(), "无变化时有明确提示")


# ─── 9. 编排流程（桩对象，不连 Qdrant）──────────────────────

class StubConfig:
    supported_extensions = [".txt"]
    recursive = True


class StubPipeline:
    """记录调用顺序的假 pipeline。"""

    def __init__(self, log):
        self.config = StubConfig()
        self.log = log
        self.fail_paths = set()

    def delete_by_source_files(self, paths):
        if paths:
            self.log.append(("delete", sorted(paths)))

    def repoint_source_file(self, old, new):
        self.log.append(("repoint", old, new))

    def run(self, paths, store_in_qdrant=True, file_meta=None):
        self.log.append(("ingest", sorted(paths)))
        # 记录 file_hash 是否真的透传下去了
        self.last_file_meta = file_meta or {}
        # fail_paths 模拟"解析失败"：这些文件不会出现在返回结果里
        ok = [p for p in paths if p not in self.fail_paths]
        return [{"chunk_count": 3, "file": p} for p in ok]


class StubManifest:
    """内存版清单。"""

    def __init__(self, log, records=None):
        self.log = log
        self.records = {canonical_path(k): v for k, v in (records or {}).items()}

    def load(self):
        return dict(self.records)

    def upsert_many(self, recs):
        if not recs:
            return          # 与 FileManifest.upsert_many 的行为保持一致
        self.log.append(("manifest_upsert", len(recs)))
        for r in recs:
            self.records[r["source_file"]] = r

    def delete_paths(self, paths):
        if paths:
            self.log.append(("manifest_delete", sorted(paths)))
        for p in paths:
            self.records.pop(p, None)

    def drop_collection(self):
        self.records = {}


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    print("=" * 65)
    print("  sync_index 差异规划回归测试")
    print("=" * 65)

    test_basic_diff()
    test_modification_detection()
    test_fingerprint_change()
    test_unreadable_is_safe()
    test_rename_detection()
    test_scan_disk()
    test_idempotent()
    test_report_render()

    hr("SUMMARY")
    if FAILURES:
        print(f"  {len(FAILURES)} 项失败:")
        for f in FAILURES:
            print(f"    - {f}")
        print("\n  Overall: FAILED")
        return 1
    print("  Overall: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
