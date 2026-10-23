"""
gen_ceo_testcases.py — CEO-files 测试用例生成器

从千里达运动文档库随机抽样文件，基于真实文本块与图片/表格资产，
用 LLM 生成 100 条测试用例：25 精确 + 60 模糊（边界）+ 15 无关。

运行（依赖入库任务已生成的 kb-assets 目录）:
    python eval/gen_ceo_testcases.py
"""

import os
import sys
import json
import random
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document_parser import DocumentParser
from augment.query_rewriter import _load_api_key, LLM_HOST, LLM_MODEL
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = "/Volumes/Trinx-CEO/KnowledgeBase/KB-GeneralBusiness/千里达运动文档"
ASSETS = os.path.join(ROOT, "kb-assets")
OUT = "/Users/trinxai/Documents/pythonfile/测试用例/CEO-files检索测试用例300条.docx"
SEED = 42

N_FILES = 130         # 随机抽样的文件数
N_PRECISE = 75
N_FUZZY = 180
N_IRRELEVANT = 45

# 无关问题：完全离域（公司管理文档库里不该有答案的；近似域的已剔除——
# 这个库含营销/培训/财务内容，"短视频涨粉""财务报表"之类其实是相关题）
IRRELEVANT = [
    "怎么做红烧肉？",
    "感冒发烧吃什么药？",
    "怎么考驾照？",
    "推荐几部悬疑电影",
    "怎么养多肉植物？",
    "下周天气怎么样？",
    "怎么学习英语口语？",
    "怎么做瑜伽下犬式？",
    "怎么给手机刷机？",
    "番茄炒蛋先放番茄还是先放蛋？",
    "怎么预防近视？",
    "怎么做蛋糕？",
    "怎么练习书法？",
    "怎么选择跑鞋？",
    "怎么给猫剪指甲？",
    "咖啡拉花怎么做？",
    "怎么挑选西瓜？",
    "怎么腌制泡菜？",
    "怎么弹钢琴入门？",
    "怎么折纸飞机？",
    "怎么去除衣服油渍？",
    "怎么养金鱼？",
    "怎么学摄影构图？",
    "怎么泡功夫茶？",
    "怎么做瑜伽冥想？",
    "怎么写小说大纲？",
    "怎么养仙人掌？",
    "怎么做风筝？",
    "怎么学滑板？",
    "怎么挑选榴莲？",
    "怎么做美甲？",
    "怎么学吉他？",
    "怎么搭帐篷？",
    "怎么腌制咸鸭蛋？",
    "怎么养护绿萝？",
    "怎么做手工皂？",
    "怎么学街舞？",
    "怎么挑新鲜的鱼？",
    "怎么自制酸奶？",
    "怎么学围棋？",
    "怎么做香薰蜡烛？",
    "怎么养护兰花？",
    "怎么学潜水？",
    "怎么挑西瓜最甜？",
    "怎么做布贴画？",
]


def ask_llm(prompt: str, max_tokens: int = 1500) -> str:
    import urllib.error
    payload = {"model": LLM_MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.7, "max_tokens": max_tokens}
    req = urllib.request.Request(
        f"{LLM_HOST}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {_load_api_key()}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())["choices"][0]["message"]["content"].strip()


def gen_cases_from_text(title: str, text: str, n_fuzzy: int) -> list:
    """从一个文本块出 1 精确 + n 模糊题"""
    prompt = f"""以下是企业管理文档《{title}》中的一个片段：

{text[:900]}

请基于这个片段的**真实内容**生成检索测试用例，输出 JSON（不要其它内容）：
{{"precise": {{"question": "针对片段中具体事实/数字/步骤的精确问题", "points": "答案要点，分号分隔"}},
  "fuzzy": [{{"question": "同一件事但用一线员工日常口语的模糊问法", "points": "答案要点，分号分隔"}} x {n_fuzzy}]}}

要求：问题的答案必须能在片段里找到；模糊题要像随口说话（如"这事儿归谁管啊"），
不要用文档原文措辞；points 里保留片段中的关键数字和名词原文。"""
    try:
        raw = ask_llm(prompt)
        raw = raw.strip().strip("`").removeprefix("json").strip()
        data = json.loads(raw)
        out = []
        p = data.get("precise")
        if p:
            out.append(("precise", p["question"], p["points"]))
        for f in (data.get("fuzzy") or [])[:n_fuzzy]:
            out.append(("fuzzy", f["question"], f["points"]))
        return out
    except Exception as e:
        logger.warning("生成失败(%s): %s", title, e)
        return []


def gen_case_from_asset(title: str, asset_path: str, kind: str) -> "tuple | None":
    """从表格/图片资产出 1 道模糊题"""
    if kind == "table":
        try:
            content = open(asset_path, encoding="utf-8").read()[:700]
        except OSError:
            return None
        material = f"表格内容：\n{content}"
    else:
        material = f"（图片文件，文件名：{os.path.basename(asset_path)}，所属文档：《{title}》）"
    prompt = f"""企业管理文档《{title}》中有一个{'表格' if kind=='table' else '图片'}资产。
{material}

请生成 1 道检索测试用题，输出 JSON（不要其它内容）：
{{"question": "一线员工想找这个{'表格' if kind=='table' else '图片'}时会随口问的话（口语化）",
  "points": "这个{'表格' if kind=='table' else '图片'}的关键内容点，分号分隔"}}
要求：问题里不要出现文件编号/文件名原文；points 要包含资产里的关键信息。"""
    try:
        raw = ask_llm(prompt).strip().strip("`").removeprefix("json").strip()
        data = json.loads(raw)
        return ("fuzzy", data["question"], data["points"] + f"；资产文件：{os.path.basename(asset_path)}")
    except Exception as e:
        logger.warning("资产题生成失败(%s): %s", asset_path, e)
        return None


def main():
    random.seed(SEED)
    parser = DocumentParser()

    # 收集可解析文件（排除资产目录）
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != "kb-assets"]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in (".pdf", ".docx", ".doc",
                                                   ".txt", ".md", ".pptx"):
                files.append(os.path.join(dirpath, fn))
    logger.info("可解析文件 %d 个", len(files))

    sampled = random.sample(files, min(N_FILES, len(files)))

    precise, fuzzy = [], []
    asset_candidates = []
    done_files = set()

    for i, fp in enumerate(sampled, 1):
        title = os.path.basename(fp)
        logger.info("[%d/%d] %s", i, len(sampled), title)
        try:
            doc = parser.parse(fp)
        except Exception as e:
            logger.warning("解析失败 %s: %s", title, e)
            continue

        # 文本块：随机挑一个足够长的 section
        texts = [s.text for s in doc.sections if len(s.text) > 150]
        if texts:
            chunk = random.choice(texts)
            for kind, q, pts in gen_cases_from_text(title, chunk, n_fuzzy=2):
                (precise if kind == "precise" else fuzzy).append(
                    (q, title, pts))
            done_files.add(fp)

        # 收集该文件的资产（kb-assets/<文档名>/）
        safe_stem = "".join(c if c.isalnum() or c in "-_（）()" else "_"
                            for c in os.path.splitext(title)[0])[:60]
        adir = os.path.join(ASSETS, safe_stem)
        if os.path.isdir(adir):
            for fn in os.listdir(adir):
                kind = "table" if fn.endswith(".md") else "image"
                asset_candidates.append((title, os.path.join(adir, fn), kind))

    # 精确题补足/裁剪到 25
    precise = precise[:N_PRECISE]

    # 资产题：随机抽最多 15 个，归入模糊题
    random.shuffle(asset_candidates)
    asset_cases = []
    for title, apath, kind in asset_candidates[:40]:
        c = gen_case_from_asset(title, apath, kind)
        if c:
            _, q, pts = c
            asset_cases.append((q, title, pts))

    fuzzy = (fuzzy + asset_cases)[:N_FUZZY]
    # 模糊题不足：从已解析文档的其它文本块继续生成（不许重复）
    seen_q = {q for q, _, _ in fuzzy}
    extra_files = [f for f in sampled if f not in done_files]
    for fp in extra_files:
        if len(fuzzy) >= N_FUZZY:
            break
        title = os.path.basename(fp)
        try:
            doc = parser.parse(fp)
        except Exception:
            continue
        for sct in [s.text for s in doc.sections if len(s.text) > 150][:2]:
            if len(fuzzy) >= N_FUZZY:
                break
            for kind, q, pts in gen_cases_from_text(title, sct, n_fuzzy=2):
                if kind == "fuzzy" and q not in seen_q:
                    seen_q.add(q)
                    fuzzy.append((q, title, pts))

    irrelevant = [(q, "（无关问题——不应命中任何入库文档）",
                   "应返回低相似度或被闸门拒答") for q in IRRELEVANT]

    cases = precise + fuzzy + irrelevant
    logger.info("精确 %d / 模糊 %d / 无关 %d", len(precise), len(fuzzy),
                len(irrelevant))

    # 写 docx（与其他用例集同格式）
    from docx import Document
    from docx.shared import Pt
    from docx.oxml.ns import qn
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    doc.add_heading("CEO-files 检索测试用例（300 条）", level=0)
    doc.add_paragraph(
        "适用对象：CEO-files collection ｜ 随机抽样 130 份文件的真实文本块 + "
        "图片/表格资产出题 ｜ 75 精确 + 180 模糊 + 45 无关 ｜ 2026-10-21")
    doc.add_paragraph("资产类题目在“期望命中内容要点”末尾标注了资产文件名。")
    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    for i, h in enumerate(["编号", "测试查询", "期望命中文件", "期望命中内容要点", "实测结果"]):
        table.rows[0].cells[i].text = h
    for idx, (q, f, pts) in enumerate(cases, 1):
        cells = table.add_row().cells
        cells[0].text = str(idx)
        cells[1].text = q
        cells[2].text = f
        cells[3].text = pts
    doc.save(OUT)
    print(f"已生成: {OUT}（{len(cases)} 条）")


if __name__ == "__main__":
    main()
