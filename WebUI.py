import gradio as gr
from pathlib import Path
import sys
import shutil

from build_agent import answer_ui
from build_vector import build_vectorstore_from_pdfs
from crawler import PDF_DIR as CRAWLER_PDF_DIR, has_any_pdf, crawl_feishu_pdfs


def resolve_base_dir() -> Path:
    """Return the working dir for runtime data (next to exe when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "DaggerHeart_ChatBot"
    return Path(__file__).resolve().parent.parent / "DaggerHeart_ChatBot"


BASE_DIR = resolve_base_dir()
PDF_DIR = BASE_DIR / "pdf"
DATA_DIR = BASE_DIR / "data"
VDB_DIR = BASE_DIR / "vectorstore_daggerheart"
WAIT_HINT = "（整个过程需要 5-10 分钟，请耐心等待）"

# Ensure runtime directories exist
for d in [PDF_DIR, DATA_DIR, VDB_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def has_vectordb(db_dir: Path = VDB_DIR):
    return db_dir.exists() and any(db_dir.iterdir())


def ui_check_or_crawl(api_key, wiki_url):
    if not api_key:
        return "请先输入 QWEN API KEY"
    if not wiki_url:
        return f"请输入飞书 wiki 链接以开始爬取。目标目录：{PDF_DIR}"
    if has_any_pdf(PDF_DIR):
        return f"检测到目录下已有 PDF，跳过爬取。（如需重爬，请点击“重新爬取 PDF”按钮。）"

    n = crawl_feishu_pdfs(wiki_url, out_dir=PDF_DIR)
    return f"爬取完成，共下载 {n} 个 PDF，保存到 {PDF_DIR}。"


def ui_force_crawl(api_key, wiki_url):
    if not api_key:
        return "请先输入 QWEN API KEY"
    if not wiki_url:
        return f"请输入飞书 wiki 链接以开始爬取。目标目录：{PDF_DIR}"
    n = crawl_feishu_pdfs(wiki_url, out_dir=PDF_DIR)
    return f"已重新爬取，下载 {n} 个 PDF，保存到 {PDF_DIR}。"


def ui_build_vector(api_key):
    if not api_key:
        return "请先输入 QWEN(DashScope) API KEY"
    if not has_any_pdf(PDF_DIR):
        return f"{PDF_DIR} 为空：请先上传或爬取 PDF"
    if has_vectordb():
        return f"检测到现有向量库，跳过重建。（如需重建，请点击“重新构建向量库”按钮。）"

    chunk_json = DATA_DIR / "daggerheart_rules_chunks.json"
    build_vectorstore_from_pdfs(
        api_key,
        pdf_dir=str(PDF_DIR),
        vectordb_dir=str(VDB_DIR),
        chunk_json=str(chunk_json),
    )
    return f"向量库构建完成✅（目录：{VDB_DIR}）。"


def ui_force_build(api_key):
    if not api_key:
        return "请先输入 QWEN(DashScope) API KEY"
    if not has_any_pdf(PDF_DIR):
        return f"{PDF_DIR} 为空：请先上传或爬取 PDF"

    if VDB_DIR.exists():
        shutil.rmtree(VDB_DIR, ignore_errors=True)
    VDB_DIR.mkdir(parents=True, exist_ok=True)

    chunk_json = DATA_DIR / "daggerheart_rules_chunks.json"
    build_vectorstore_from_pdfs(
        api_key,
        pdf_dir=str(PDF_DIR),
        vectordb_dir=str(VDB_DIR),
        chunk_json=str(chunk_json),
    )
    return f"已重新构建向量库✅（目录：{VDB_DIR}）。"


def ui_chat(api_key, question, history, model="QWEN"):
    if not api_key:
        return history + [{"role": "assistant", "content": "请先输入 API KEY"}], history
    if not question:
        return history, history
    ans, src = answer_ui(question, api_key, model)
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ans + "\n\n【来源】\n" + src},
    ]
    return history, history


def ui_toggle_guide(is_open: bool):
    new_state = not is_open
    return new_state, gr.update(visible=new_state)


with gr.Blocks(
    css="""
#status-row {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}
#wait-hint {
  font-size: 12px;
  color: #444;
  white-space: nowrap;
}
#status-box {
  text-align: right;
}
#usage-guide-box {
  font-size: 13px;
  line-height: 1.6;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 8px;
  padding: 12px 14px;
  background: rgba(0,0,0,0.02);
}
""",
) as demo:
    model = "QWEN"
    gr.Markdown("## 刀锋之心规则问答 (QWEN + RAG)")
    api_key = gr.Textbox(label="DashScope API KEY", type="password")
    wiki_url = gr.Textbox(label="飞书 Wiki URL（pdf 目录为空时用于爬取）", placeholder="https://.../wiki/<node_token>")

    with gr.Row(elem_id="status-row"):
        gr.Markdown(WAIT_HINT, elem_id="wait-hint")
        status = gr.Textbox(label="状态", interactive=False, elem_id="status-box")

    btn1 = gr.Button("1) 检测或爬取 PDF（存在则跳过）")
    btn2 = gr.Button("2) 检测/构建 向量库（存在则跳过）")
    btn_force_crawl = gr.Button("重新爬取 PDF")
    btn_force_build = gr.Button("重新构建向量库")

    guide_btn = gr.Button("使用指引")
    guide_box = gr.Markdown(
        """
            **使用指引：**
        1. 该应用为基于QWEN3_MAX构建的匕首之心TRPG规则ChatBot,在运行前需要输入QWEN API Key。
            QWEN API Key的获取地址如下：
            “https://bailian.console.aliyun.com/?spm=5176.12818093_47.resourceCenter.1.3be92cc9itjVbq&tab=model#/api-key”
            请在运行时注意token使用量，超出免费用量需要在QWEN平台上进行充值
        2. 运行该程序将在程序所在文件夹下创建./DaggerHeart_ChatBot文件夹，并在匕首之心非官方中文规则资料站上爬取PDF文件。
            《匕首之心》非官方中文规则资料站的网址如下：
            https://lcnw6r5edbf2.feishu.cn/wiki/NwsAwL4JpiFsx2kXhCwc5FzWnsh
            输入该网址后，点击“1）检测或爬取PDF”将自动爬取pdf并作为大模型回答内容的提示器，爬取需要较长时间，请耐心等待
        3. 在爬取完成后，请点击“2）检测/构建向量库”，整个构建也需要花费一些时间。
        4. 在爬取PDF，向量库构建完成后，即可在下方问答框中输入你的问题,首次提问也需要较长时间（大约30s）
            该应用还处于初期调试阶段，部分问题无法得到准确的解答，请以实际判断为准
        5. 重新爬取按钮仅在你想更新模型知识库时（一般是规则网站有重大更新时）使用，一般情况不推荐点击
        """,
        elem_id="usage-guide-box",
        visible=False,
    )
    guide_open = gr.State(False)
    guide_btn.click(fn=ui_toggle_guide, inputs=guide_open, outputs=[guide_open, guide_box])

    chatbot = gr.Chatbot(label="规则问答")
    question = gr.Textbox(label="你的问题")
    history_state = gr.State([])

    btn1.click(ui_check_or_crawl, inputs=[api_key, wiki_url], outputs=status)
    btn2.click(ui_build_vector, inputs=[api_key], outputs=status)
    btn_force_crawl.click(ui_force_crawl, inputs=[api_key, wiki_url], outputs=status)
    btn_force_build.click(ui_force_build, inputs=[api_key], outputs=status)

    question.submit(ui_chat, inputs=[api_key, question, history_state], outputs=[chatbot, history_state])

demo.queue().launch(inbrowser=True, show_error=True, share=True)
