from pathlib import Path
import sys
from spider import (
    get_tenant_access_token,
    parse_node_token_from_url,
    get_wiki_node_info,
    crawl_docx_for_pdf_mentions_all,
    download_all_pdfs,
)


def resolve_base_dir() -> Path:
    """Return the working dir for runtime data (next to exe when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "DaggerHeart_ChatBot"
    return Path(__file__).resolve().parent.parent / "DaggerHeart_ChatBot"


BASE_DIR = resolve_base_dir()
PDF_DIR = BASE_DIR / "pdf"
# Ensure base directories exist for crawler operations
PDF_DIR.mkdir(parents=True, exist_ok=True)


def crawl_feishu_pdfs(wiki_url: str, out_dir: Path | str = PDF_DIR, max_blocks: int = 5000):
    node_token = parse_node_token_from_url(wiki_url)
    access_token = get_tenant_access_token()
    node = get_wiki_node_info(access_token, node_token)

    doc_id = node["obj_token"]
    obj_type = node["obj_type"]
    if obj_type not in ("docx", "doc"):
        raise RuntimeError(f"not a doc/docx node: obj_type={obj_type}")

    pdf_items = crawl_docx_for_pdf_mentions_all(access_token, doc_id, max_blocks=max_blocks)
    download_all_pdfs(access_token, pdf_items, str(out_dir))
    return len(pdf_items)


def has_any_pdf(pdf_dir: Path | str = PDF_DIR) -> bool:
    p = Path(pdf_dir)
    return p.exists() and any(x.suffix.lower() == ".pdf" for x in p.iterdir())

