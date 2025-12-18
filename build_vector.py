from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings  # ✅ QWEN 用这个
from dotenv import load_dotenv
from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json,os,sys,fitz,re
import os
import json
import fitz  # PyMuPDF
from statistics import median
import re
from pathlib import Path
import sys
from env_config import MODEL


def resolve_base_dir() -> Path:
    """Return the working dir for runtime data (next to exe when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "DaggerHeart_ChatBot"
    return Path(__file__).resolve().parent.parent / "DaggerHeart_ChatBot"


BASE_DIR = resolve_base_dir()
PDF_DIR = BASE_DIR / "pdf"
DATA_DIR = BASE_DIR / "data"
VDB_DIR = BASE_DIR / "vectorstore_daggerheart"

# ========= 工具函数 =========
def extract_page_lines_with_font(page):
    """
    从一页中提取所有“行”：
    每行包括：页码、文本、平均字体大小
    """
    result = []
    page_dict = page.get_text("dict")  # 带结构信息

    for block in page_dict.get("blocks", []):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            spans = line.get("spans", [])
            if not spans:
                continue

            # 拼一整行文本
            line_text = "".join(span.get("text", "") for span in spans).strip()
            if not line_text:
                continue

            # 计算按字符数加权的平均字体大小
            total_chars = 0
            size_sum = 0.0
            for span in spans:
                span_text = span.get("text", "")
                span_size = span.get("size", 0.0)
                n = len(span_text.strip())
                if n <= 0:
                    continue
                total_chars += n
                size_sum += span_size * n

            if total_chars == 0:
                continue

            avg_size = size_sum / total_chars

            result.append({
                "page": page.number,   # 0-based
                "text": line_text,
                "font_size": avg_size,
            })

    return result


def extract_pdf_lines_with_font(pdf_path):
    """
    对整个 PDF，提取所有行 + 字体大小。
    返回：按阅读顺序排列的行列表。
    """
    doc = fitz.open(pdf_path)
    all_lines = []
    for page in doc:
        all_lines.extend(extract_page_lines_with_font(page))
    doc.close()
    return all_lines


def find_body_font_size(lines):
    """
    估计“正文”的字体大小：
    用所有较长行（len>6）的字体大小中位数作为正文字号。
    """
    sizes = [l["font_size"] for l in lines if len(l["text"]) > 6]
    if not sizes:
        return None
    body_size = median(sizes)
    return body_size


def mark_headings(lines, body_size, min_len=2, max_len=25, scale=1.1):
    """
    给每一行打上 is_heading 标记。
    条件（全部满足）：
      - 行长在 [min_len, max_len]
      - 字号 >= body_size * scale（比正文大）
      - 行尾不是句号/问号/感叹号/逗号/分号/冒号等（避免整句）
    """
    bad_endings = ("。", "！", "？", "，", "；", ":", "：")

    for l in lines:
        text = l["text"].strip()
        size = l["font_size"]
        if not text or body_size is None:
            l["is_heading"] = False
            continue

        is_short = (min_len <= len(text) <= max_len)
        is_bigger = (size >= body_size * scale)
        not_sentence_end = not text.endswith(bad_endings)

        l["is_heading"] = bool(is_short and is_bigger and not_sentence_end)

    return lines


def looks_like_chapter_title(text):
    """
    额外规则：匹配“第X章”这种大标题。
    即使字号没明显变大，也可以认定为标题。
    """
    t = text.strip()
    # 例如：第 一 章、第1章、第一章、第一章：战斗
    # 简单正则：以“第”开头，中间若干字，包含“章”
    return bool(re.match(r"^第.{0,10}章", t))


def group_sections_by_heading(lines):
    """
    按标题分段：
    - 遇到 is_heading=True 或 looks_like_chapter_title() 的行 → 作为新标题
    - 标题下直到下一个标题之前的所有行 → 作为该 section 的内容
    返回：[(title, content, page_start, page_end, heading_font_size), ...]
    """
    sections = []
    current_title = "全文"
    current_buf = []
    current_pages = []
    current_heading_font = None  # 新增：记录这一段标题的字号

    def flush():
        nonlocal sections, current_title, current_buf, current_pages, current_heading_font
        if not current_buf:
            return
        content = "\n".join(current_buf).strip()
        if not content:
            return
        page_start = min(current_pages) if current_pages else None
        page_end = max(current_pages) if current_pages else None
        sections.append((current_title, content, page_start, page_end, current_heading_font))

    for l in lines:
        text = l["text"].strip()
        if not text:
            continue

        is_heading = l.get("is_heading", False) or looks_like_chapter_title(text)

        if is_heading:
            # 碰到新标题，先结算之前的段
            flush()
            current_title = text
            current_buf = []
            current_pages = [l["page"]]
            current_heading_font = l["font_size"]   # 记录这个标题的字号
        else:
            current_buf.append(text)
            current_pages.append(l["page"])

    flush()

    if not sections:
        # 如果完全没识别出标题，就整本书当一个 section
        all_text = "\njoin(l['text'] for l in lines)"
        pages = [l["page"] for l in lines] or [0]
        sections = [("全文", all_text, min(pages), max(pages), None)]

    return sections



def guess_rule_type(title: str, source_name: str) -> str:
    """
    根据章节标题 + 源文件名粗略猜测 rule_type。
    你可以按需要修改关键词。
    """
    text = f"{title} {source_name}".lower()

    # combat
    if any(k in text for k in ["战斗", "攻击", "伤害", "先攻", "回合"]):
        return "combat"

    # rules / checks
    if any(k in text for k in ["检定", "判定", "特质", "规则", "骰", "二元骰","经历"]):
        return "rule_check"

    # class / abilities
    if any(k in text for k in ["领域", "职业", "能力", "专长", "天赋","奥术","利刃","骸骨","典籍","优雅","午夜","贤者","辉耀","勇气","特性"]):
        return "class"

    # equipment
    if any(k in text for k in ["装备", "武器", "护甲", "防具", "道具", "消耗品"]):
        return "equipment"

    # lore / world
    if any(k in text for k in ["世界", "设定", "背景", "故事", "剧情","角色"]):
        return "lore"

    return "general"


# ========= 主逻辑：处理单个 PDF =========

def process_single_pdf(pdf_path):
    """
    处理单个 PDF：
      - 提取行 + 字体
      - 估计正文字号
      - 识别标题行
      - 按标题分段
      - 生成结构化记录列表
    """
    base_name = os.path.basename(pdf_path)
    source_name = os.path.splitext(base_name)[0]  # 去掉 .pdf

    print(f"[INFO] Processing PDF: {base_name}")

    lines = extract_pdf_lines_with_font(pdf_path)
    if not lines:
        print(f"[WARN] No text found in {base_name}, skip.")
        return []

    body_size = find_body_font_size(lines)
    print(f"  Detected body font size ~ {body_size:.2f}" if body_size else "  Could not detect body font size")

    lines = mark_headings(lines, body_size)
    sections = group_sections_by_heading(lines)

    records = []

    context_rule_type = None          # 最近一个非 general 的 rule_type（比如 class）
    context_heading_font = None       # 这个“父标题”的字号

    for idx, (title, content, page_start, page_end, heading_font) in enumerate(sections):
        base_rule_type = guess_rule_type(title, source_name)  # 先用原来的关键词规则
        effective_rule_type = base_rule_type

        # 1）如果本身就命中一个清晰类型（非 general），更新上下文
        if base_rule_type != "general":
            context_rule_type = base_rule_type
            context_heading_font = heading_font

        # 2）如果是 general，但有一个最近的“父标题类型”，并且字号更小 → 继承父类型
        elif context_rule_type is not None and heading_font is not None and context_heading_font is not None:
            # 子标题字号通常比父标题小，给一点浮动空间
            if heading_font <= context_heading_font * 1.01:
                effective_rule_type = context_rule_type

        record_id = f"{source_name}_{idx}"

        records.append({
            "id": record_id,
            "parent_id": record_id,
            "source_file": base_name,
            "chapter": title,
            "section": title,
            "rule_type": effective_rule_type,   # 用 effective_rule_type
            "page_start": None if page_start is None else int(page_start) + 1,
            "page_end": None if page_end is None else int(page_end) + 1,
            "content": content
        })

    print(f"  -> got {len(records)} sections")
    return records

# ========= 批量处理所有 PDF =========
def run_structured_stage_from_pdfs(pdf_dir,chunk_json):
    os.makedirs(os.path.dirname(chunk_json), exist_ok=True)

    all_records = []

    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(pdf_dir, fname)
        recs = process_single_pdf(pdf_path)
        all_records.extend(recs)

    print(f"[DONE] Total sections: {len(all_records)}")

    with open(chunk_json, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {chunk_json}")

# vectorstore
def load_chunks(chunk_json,chunk_size,chunk_overlap):
    """从 JSON 文件加载文本块并转换为 Document 对象。"""
    try:
        with open(chunk_json, "r", encoding="utf-8") as f:
            items = json.load(f)

        docs = []
        for x in items:
            # 1）新格式：{"content": "...", "metadata": {...}}
            if "metadata" in x:
                content = x.get("content", "")
                metadata = x.get("metadata", {}) or {}
            else:
                # 2）旧/扁平格式：所有字段在一层
                #   把 "content" 单独拿出来，其余全部当作 metadata
                content = x.get("content", "")
                metadata = {k: v for k, v in x.items() if k != "content"}

            docs.append(Document(page_content=content, metadata=metadata))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        docs = text_splitter.split_documents(docs)

        print(f"Loaded {len(docs)} chunks from {chunk_json}")
        return docs

    except FileNotFoundError:
        print(f"❌ Error: The file {chunk_json} was not found.")
        return []



def build_vectorstore(embedding_fn, chunk_json ,vectordb_dir,chunk_size,chunk_overlap,db_suffix=""):
    """
    使用指定的嵌入函数构建 Chroma 向量数据库。
    """
    docs = load_chunks(chunk_json,chunk_size,chunk_overlap)
    if not docs:
        print("No documents loaded. Exiting.")
        return

    # 根据模式修改持久化路径，避免数据冲突
    final_db_dir = f"{vectordb_dir}"
    print(f"Loaded {len(docs)} chunks. Start embedding using {db_suffix}...")
    print(f"Persisting to directory: {final_db_dir}")

    # 构建向量库
    vectordb = Chroma.from_documents(
        docs,
        embedding=embedding_fn,
        persist_directory=final_db_dir,
    )
    vectordb.persist()
    print(f"Success! Vectorstore built using {db_suffix}.")


def run_vectorstore_stage(api_key, model, vectordb_dir, chunk_json, chunk_size, chunk_overlap):
    embedding_fn = None
    db_suffix = model

    if model == "OPENAI":
        embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small")
    elif model == "GEMINI":
        embedding_fn = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)
    elif model == "QWEN":
        embedding_fn = DashScopeEmbeddings(model="text-embedding-v4", dashscope_api_key=api_key)
    else:
        print(f"❌Error: Unknown EMBEDDING_MODE: {model}")
        sys.exit(1)

    # 这里用 chunk_json，而不是模型名
    build_vectorstore(embedding_fn, chunk_json, vectordb_dir, chunk_size, chunk_overlap, db_suffix=db_suffix)

def build_vectorstore_from_pdfs(
    api_key: str,
    model: str = "QWEN",
    pdf_dir: str = str(PDF_DIR),
    chunk_json: str = str(DATA_DIR / "daggerheart_rules_chunks.json"),
    vectordb_dir: str = str(VDB_DIR),
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
):
    run_structured_stage_from_pdfs(pdf_dir=pdf_dir, chunk_json=chunk_json)
    run_vectorstore_stage(
        api_key=api_key,
        model=model,
        vectordb_dir=vectordb_dir,
        chunk_json=chunk_json,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
