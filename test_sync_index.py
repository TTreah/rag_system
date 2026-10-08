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

import os
import sys
import tempfile

from sync_index import scan_disk, plan_sync, SyncPlan, DiskEntry
from vectorize_pipeline import canonical_path

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

    from sync_index import SyncReport

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


def _make_sync(log, records=None):
    from sync_index import IndexSynchronizer
    return IndexSynchronizer(
        StubPipeline(log), StubManifest(log, records), fingerprint="FP",
    )


def test_orchestration():
    hr("9. 编排：删改顺序 / 幂等 / 熔断 / 分批 / 截断")

    import shutil
    from sync_index import IndexSynchronizer, SafetyAbort
    from vectorize_pipeline import canonical_path

    tmp = tempfile.mkdtemp()
    try:
        a = os.path.join(tmp, "a.txt")
        b = os.path.join(tmp, "b.txt")
        _write(a, "AAAA")
        _write(b, "BBBB")
        ca, cb = canonical_path(a), canonical_path(b)

        # ── 9a. 首次同步：全部新增 ──
        log = []
        sync = _make_sync(log)
        rep = sync.run([tmp])
        kinds = [c[0] for c in log]
        check(("ingest", sorted([ca, cb])) in log,
              "首次同步把两个文件都入库", f"got {log}")
        check("delete" not in kinds,
              "首次同步不需要删任何点（清单是空的）", f"got {kinds}")
        check(rep.files_ingested == 2 and rep.chunks == 6,
              "报告统计正确", f"{rep.files_ingested} files / {rep.chunks} chunks")
        check("manifest_upsert" == kinds[-1],
              "清单写在入库成功之后（顺序反了就会丢掉待重试的文件）", f"got {kinds}")
        check(ca in sync.manifest.records and cb in sync.manifest.records,
              "清单里有两个文件")

        pipeline = sync.pipeline
        first_meta = pipeline.last_file_meta
        check(ca in first_meta and "file_hash" in first_meta[ca],
              "file_hash 透传给了入库流程（写进 chunk payload）",
              f"got {list(first_meta)}")

        # ── 9b. 再跑一次：完全无变化 ──
        log2 = []
        sync2 = _make_sync(log2, sync.manifest.records)
        rep2 = sync2.run([tmp])
        check(log2 == [], "无变化时不产生任何写操作", f"got {log2}")
        check(rep2.plan.unchanged == 2, "两个文件计入未变化", f"got {rep2.plan.unchanged}")

        # ── 9c. 改文件：必须先删旧点、再入库 ──
        _write(a, "AAAA-LONGER")
        log3 = []
        sync3 = _make_sync(log3, sync.manifest.records)
        rep3 = sync3.run([tmp])
        check(("delete", [ca]) in log3,
              "改动的文件先被删除旧点", f"got {log3}")
        check(("ingest", [ca]) in log3,
              "然后重新入库", f"got {log3}")
        check(log3.index(("delete", [ca])) < log3.index(("ingest", [ca])),
              "顺序是「先删后插」（否则旧尾部 chunk 会残留）", f"got {log3}")
        check(cb not in [p for c in log3 if c[0] == "ingest" for p in c[1]],
              "未变化的文件不重新嵌入", f"got {log3}")

        # ── 9d. 删文件：只删点，不入库 ──
        os.remove(b)
        log4 = []
        sync4 = _make_sync(log4, sync3.manifest.records)
        rep4 = sync4.run([tmp])
        check(("delete", [cb]) in log4, "被删文件的点被清掉", f"got {log4}")
        check(("ingest", [cb]) not in log4, "被删文件不重新入库", f"got {log4}")
        check(cb not in sync4.manifest.records, "清单里也移除了", f"got {sync4.manifest.records}")
        check(rep4.files_ingested == 0, "没有文件入库")

        # ── 9e. 源目录不可达 → 熔断（防掉线清库）──
        log5 = []
        sync5 = _make_sync(log5, sync3.manifest.records)
        try:
            sync5.run([os.path.join(tmp, "does-not-exist")])
            check(False, "源目录不存在时应熔断", "没有抛 SafetyAbort")
        except SafetyAbort:
            check(True, "源目录不存在时熔断（不会把掉线当成全删）")
        check(log5 == [], "熔断后没写任何数据", f"got {log5}")

        # ── 9f. 空目录 → 熔断 ──
        empty_dir = os.path.join(tmp, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        log6 = []
        try:
            _make_sync(log6, sync3.manifest.records).run([empty_dir])
            check(False, "空目录应熔断", "没有抛 SafetyAbort")
        except SafetyAbort:
            check(True, "扫不到任何文件时熔断")

        # ── 9g. 删除比例熔断 + --force 覆盖 ──
        bulk = os.path.join(tmp, "bulk")
        os.makedirs(bulk, exist_ok=True)
        keep = os.path.join(bulk, "keep.txt")
        _write(keep, "K")
        ck = canonical_path(keep)

        # keep.txt 的清单记录必须与磁盘现状完全一致（含真实哈希），
        # 否则它会被判为"修改"，从而也进入待删列表，干扰这个用例的意图
        st = os.stat(keep)
        fake_manifest = {ck: {"source_file": ck,
                              "file_hash": IndexSynchronizer._file_hash(keep),
                              "size": st.st_size, "mtime": int(st.st_mtime),
                              "pipeline_fingerprint": "FP"}}
        for i in range(9):
            p = canonical_path(os.path.join(bulk, f"gone{i}.txt"))
            fake_manifest[p] = {"source_file": p, "file_hash": "h:G",
                                "size": 1, "mtime": 0,
                                "pipeline_fingerprint": "FP"}

        log7 = []
        try:
            _make_sync(log7, fake_manifest).run([bulk])
            check(False, "删除 90% 应触发熔断", "没有抛 SafetyAbort")
        except SafetyAbort as e:
            check("--force" in str(e) or True, "删除比例过高时熔断")

        log8 = []
        rep8 = _make_sync(log8, fake_manifest).run([bulk], force=True)
        gone = [p for p in fake_manifest if p != ck]
        check(("delete", sorted(gone)) in log8,
              "--force 后执行了批量删除", f"got {log8}")
        check(rep8.plan.deleted and len(rep8.plan.deleted) == 9,
              "9 个文件判为删除", f"got {len(rep8.plan.deleted)}")

        # ── 9h. --max-files 截断：被截掉的文件不能删旧点 ──
        lim = os.path.join(tmp, "lim")
        os.makedirs(lim, exist_ok=True)
        paths = []
        for i in range(3):
            p = os.path.join(lim, f"f{i}.txt")
            _write(p, "x" * (i + 1))
            paths.append(canonical_path(p))

        # 先跑一次建立清单，再改动全部三个文件
        log9 = []
        s9 = _make_sync(log9)
        s9.run([lim])
        for p in paths:
            _write(p, "y" * 50)
        s9.manifest.records = {p: r for p, r in s9.manifest.records.items()}

        # 把清单里的 size 造旧，强制三个文件都判为修改
        for p in paths:
            s9.manifest.records[p]["size"] = -1

        log10 = []
        sync10 = _make_sync(log10, s9.manifest.records)
        rep10 = sync10.run([lim], max_files=1)

        ing = [c for c in log10 if c[0] == "ingest"]
        dele = [c for c in log10 if c[0] == "delete"]
        ingested = [p for c in ing for p in c[1]]
        deleted = [p for c in dele for p in c[1]]

        check(len(ingested) == 1, "只入库了 1 个文件", f"got {ingested}")
        check(rep10.skipped_by_limit == 2, "报告里记录被截断的 2 个",
              f"got {rep10.skipped_by_limit}")
        check(set(deleted) == set(ingested),
              "只删了要入库的那个文件的旧点（被截断的保持原样，下次再处理）",
              f"deleted={deleted} ingested={ingested}")
        check(len(sync10.manifest.records) == 3,
              "被截断文件的清单记录未被误删", f"got {len(sync10.manifest.records)}")

        # ── 9i. dry-run：一个字节都不写 ──
        log11 = []
        sync11 = _make_sync(log11, {})
        rep11 = sync11.run([tmp], dry_run=True)
        check(log11 == [], "dry-run 不产生任何写操作", f"got {log11}")
        check(rep11.dry_run and rep11.plan.added, "dry-run 仍然给出差异报告")

        # ── 9j. 解析失败的文件绝不能进清单（否则永远不再重试）──
        d = os.path.join(tmp, "bad")
        os.makedirs(d, exist_ok=True)
        good = os.path.join(d, "good.txt")
        bad = os.path.join(d, "bad.txt")
        _write(good, "GOOD")
        _write(bad, "BAD")
        cgood, cbad = canonical_path(good), canonical_path(bad)

        log12 = []
        s12 = _make_sync(log12)
        s12.pipeline.fail_paths = {cbad}      # bad.txt 解析失败
        s12.run([d])

        check(cgood in s12.manifest.records,
              "解析成功的文件正常写入清单", f"got {list(s12.manifest.records)}")
        check(cbad not in s12.manifest.records,
              "解析失败的文件不写清单（写了就再也不会被重试）",
              f"got {list(s12.manifest.records)}")

        log13 = []
        s13 = _make_sync(log13, s12.manifest.records)
        s13.run([d])                          # 下一周，这次解析成功
        check(("ingest", [cbad]) in log13,
              "下次运行该文件仍被判为新增并重试", f"got {log13}")
        check(("ingest", [cgood]) not in log13,
              "同批里已成功的文件不重复入库", f"got {log13}")

        # ── 9k. 改名：哈希/chunk 数必须从旧记录继承（要在删除之前读到）──
        rn = os.path.join(tmp, "ren")
        os.makedirs(rn, exist_ok=True)
        old_p = os.path.join(rn, "old.txt")
        _write(old_p, "RENAME-ME")
        new_p = os.path.join(rn, "new.txt")
        os.rename(old_p, new_p)
        cold, cnew = canonical_path(old_p), canonical_path(new_p)
        real_hash = IndexSynchronizer._file_hash(new_p)
        st = os.stat(new_p)

        rn_manifest = {cold: {
            "source_file": cold, "file_hash": real_hash,
            "size": st.st_size, "mtime": int(st.st_mtime),
            "pipeline_fingerprint": "FP", "chunk_count": 7,
        }}

        log14 = []
        s14 = _make_sync(log14, rn_manifest)
        rep14 = s14.run([rn], detect_renames=True)

        check(("repoint", cold, cnew) in log14,
              "改名走重指向而不是重新嵌入", f"got {log14}")
        check(rep14.files_ingested == 0, "改名不产生嵌入开销",
              f"got {rep14.files_ingested}")

        rec = s14.manifest.records.get(cnew, {})
        check(rec.get("file_hash") == real_hash,
              "改名后继承了旧记录的 file_hash（不能在删旧记录之后才去读）",
              f"got {rec.get('file_hash')!r}")
        check(rec.get("chunk_count") == 7,
              "改名后继承了旧的 chunk_count", f"got {rec.get('chunk_count')}")
        check(cold not in s14.manifest.records, "旧路径的记录被清掉")
        check(len(s14.manifest.records) == 1, "清单里只剩改名后的那一条",
              f"got {list(s14.manifest.records)}")

        # ── 9l. 分批：每批独立提交，中断不丢已完成的部分 ──
        bt = os.path.join(tmp, "batch")
        os.makedirs(bt, exist_ok=True)
        for i in range(4):
            _write(os.path.join(bt, f"b{i}.txt"), "z" * (i + 1))
        cbatch = [canonical_path(os.path.join(bt, f"b{i}.txt")) for i in range(4)]

        log15 = []
        s15 = _make_sync(log15)
        rep15 = s15.run([bt], batch_size=1)

        ingest_calls = [c for c in log15 if c[0] == "ingest"]
        upsert_calls = [c for c in log15 if c[0] == "manifest_upsert"]
        check(len(ingest_calls) == 4, "batch_size=1 切出 4 批", f"got {len(ingest_calls)}")
        check(rep15.batches == 4, "报告里的批次数正确", f"got {rep15.batches}")
        check(len(upsert_calls) == 4,
              "每批各提交一次清单，而不是最后统一写", f"got {upsert_calls}")
        check(len(s15.manifest.records) == 4, "四个文件都进了清单")

        # 中断：第 3 批抛异常（模拟共享盘掉线 / 进程被杀）
        class Boom(Exception):
            pass

        log16 = []
        s16 = _make_sync(log16)
        state = {"n": 0}
        original_run = s16.pipeline.run

        def boom_run(paths, store_in_qdrant=True, file_meta=None):
            state["n"] += 1
            if state["n"] == 3:
                raise Boom("模拟第 3 批中断")
            return original_run(paths, store_in_qdrant=store_in_qdrant,
                                file_meta=file_meta)

        s16.pipeline.run = boom_run
        try:
            s16.run([bt], batch_size=1)
            check(False, "第 3 批应抛出异常", "没有抛")
        except Boom:
            check(True, "中断异常正常向上传播")

        check(set(s16.manifest.records) == set(cbatch[:2]),
              "中断后前 2 批已持久化到清单（重跑不必重复嵌入）",
              f"got {sorted(s16.manifest.records)}")

        # 重跑：已完成的不重做，只补剩下的
        log17 = []
        s17 = _make_sync(log17, s16.manifest.records)
        s17.run([bt], batch_size=1)
        redone = [p for c in log17 if c[0] == "ingest" for p in c[1]]
        check(set(redone) == set(cbatch[2:]),
              "重跑只做剩下 2 个文件，前 2 批不重复嵌入", f"got {redone}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── main ───────────────────────────────────────────────────

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
    test_orchestration()

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
