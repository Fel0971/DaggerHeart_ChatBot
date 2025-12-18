import os
import dashscope
import sys
from pathlib import Path
from typing import List

from env_config import TOP_K

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma

def resolve_base_dir() -> Path:
    """Return the working dir for runtime data (next to exe when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "DaggerHeart_ChatBot"
    return Path(__file__).resolve().parent.parent / "DaggerHeart_ChatBot"


BASE_DIR = resolve_base_dir()
VDB_DIR = BASE_DIR / "vectorstore_daggerheart"


SYSTEM_PROMPT = """你是一名熟悉 **Daggerheart** (匕首之心) 桌面角色扮演游戏规则的中文规则问答助手。

【你的职责】
1. 严格根据提供的“规则文本”进行回答，不要凭空编造规则。
2. 优先给出明确的结论（可以/不可以/有条件可以），然后解释原因。
3. 在回答末尾，列出你引用的规则来源（文件名、章节标题等）。

【注意事项】
- 如果规则文本中没有明确说明，请诚实说明“不确定/未找到明确规则”，并根据常理给出可能的解释，但要标明是推测。
- 不要擅自改写规则。
- 回答使用简体中文。"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human",
         "【规则文本】\n"
         "{context}\n\n"
         "【玩家问题】\n"
         "{question}\n\n"
         "请根据上方规则文本，直接给出裁定和解释。"
        ),
    ]
)

def load_vectorstore(md, vectordb_dir, api_key) -> Chroma:
    print(f"--- DEBUG: 正在使用的 MODEL 配置是: '{md}' ---")
    if not os.path.exists(vectordb_dir):
        raise RuntimeError(f"找不到向量库目录{vectordb_dir}，请先运行build_vector.py")
    
    # ⚠️ 必须在这里声明或初始化 embedding_fn，或使用 else 块确保赋值！
    embedding_fn = None # 显式初始化，增加健壮性

    if md == 'OPENAI':
        embedding_fn = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
    elif md == "GEMINI":
        embedding_fn = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=api_key,
        )
    elif md == "QWEN":

        # 使用 LangChain 的 DashScopeEmbeddings 封装
        embedding_fn = DashScopeEmbeddings(
            model="text-embedding-v4", 
            dashscope_api_key=api_key,
        )
    else:
        # ✨ 新增的 else 块，用于处理 MODEL 变量值不匹配的情况
        print(f"❌ Fatal Error: 未知的 MODEL 配置值 '{md}'。")
        print("请检查您的 env_config 或环境变量，MODEL 必须是 'OPENAI', 'GEMINI', 或 'QWEN' 之一。")
        sys.exit(1)

    # 再次检查 embedding_fn，以防万一（虽然有了 else 块，这通常是多余的）
    if embedding_fn is None:
         # 这一行主要用于捕获理论上不应该发生的错误，例如 sys.exit(1)失败等
         raise RuntimeError("未能成功创建 Embedding Function。")

    vectordb = Chroma(
        embedding_function=embedding_fn,
        persist_directory=vectordb_dir
    )
    return vectordb

def build_llm(md,api_key) -> ChatOpenAI:
    """
    根据 MODEL 环境变量构建并返回相应的 LangChain LLM 实例。
    支持 OPENAI, GEMINI, 和 QWEN (DashScope) 模型。
    """
    temperature = 0.2
    
    if md == 'OPENAI':
        llm = ChatOpenAI(
            model='gpt-4o-moni',
            temperature=temperature,
            api_key=api_key,
        )
    elif md == "GEMINI":
        llm = ChatGoogleGenerativeAI(
            model = "gemini-2.5-flash",
            temperature=temperature,
            google_api_key=api_key,
        )
    elif md == "QWEN":
        llm = ChatOpenAI(
            model='qwen3-max', 
            temperature=temperature,
            api_key=api_key, # 传入 QWEN 的 API Key
            # 设置 DashScope 的 OpenAI 兼容模式的 base_url
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    else:
        # 如果 MODEL 变量是未知的，可以抛出错误或使用默认值
        raise ValueError(f"❌ Error: 未知的 MODEL 类型: {md}. 仅支持 OPENAI, GEMINI, QWEN。")
    
    print(f"✅ LLM 构建完成：{md} - {llm.model_name}")
    return llm
    
def format_context(docs:List[Document]) -> str:
    '''把检索到的Document拼成给LLM的上下文字符串'''
    parts = []
    for i,d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source_file","")
        chap = meta.get("chapter","")
        sec = meta.get("section","")
        header = f"[来源{i}] 文件:{src}|章节:{chap}|小节：{sec}"
        parts.append(header + "\n" + d.page_content)
    return "\n\n".join(parts)

def format_sources(docs:List[Document]) -> str:
    '''在回答后面附上一些参考来源信息'''
    seen = set()
    lines = []
    for d in docs:
        meta = d.metadata or {}
        key = (meta.get("source_file",""),meta.get("chapter",""),meta.get("section",""))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"-{key[0]} / {key[1]} / {key[2]}")
    if not lines:
        return"(未检测到明确的规则来源)"
    return "\n".join(lines)

from langchain_chroma import Chroma
def load_vectordb(vectordb_dir:str,embedding_fn):
    return Chroma(persist_directory=vectordb_dir,embedding_function=embedding_fn)
def format_context(docs:list[Document],max_chars:int=12000)->str:
    parts = []
    total = 0
    for i,d in enumerate(docs,1):
        meta = d.metadata or {}
        header = (
            f"[来源(i)]文件：{meta.get('source_file','')}|"
            f"章节:{meta.get('chapter','')} | 小节:{meta.get('section','')}"
        )
        block = header + "\n" + d.page_content
        total += len(block)
        if total > max_chars:
            break
        parts.append(block)
    return "\n\n".join(parts)

def format_sources(docs:list[Document],max_items:int = 10)->str:
    seen = set()
    out = []
    for d in docs:
        meta = d.metadata or {}
        key = (
            meta.get("source_file",""),
            meta.get("chapter",""),
            meta.get("section",""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(f"- {key[0]}/{key[1]}/{key[2]}")
        if len(out) >= max_items:
            break
    return "\n".join(out) if out else "(未检测到明确的规则来源)"

def answer_question(
    question:str,
    llm,
    retriever,
    prompt_template,
    top_k:int = 5,
)->tuple[str,str]:
    docs = retriever.invoke(question)
    context = format_context(docs)
    sources = format_sources(docs)
    
    messages = prompt_template.format_messages(
        context=context,
        question=question,
    )
    resp = llm.invoke(messages)
    return resp.content,sources

from functools import lru_cache
@lru_cache(maxsize=1)
def get_runtime(model: str, api_key: str):
    # 仅支持 QWEN；如果你未来切换其他模型，可以移除缓存或调整 key
    if model != "QWEN":
        raise ValueError("当前缓存仅支持 QWEN")
    vectordb = load_vectorstore(md=model, vectordb_dir=str(VDB_DIR), api_key=api_key)
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    llm = build_llm(md=model, api_key=api_key)
    return vectordb, retriever, llm

def answer_ui(question: str, dashscope_api_key: str, model: str):
    _, retriever, llm = get_runtime(model, dashscope_api_key)
    docs = retriever.invoke(question)
    context = format_context(docs)
    sources = format_sources(docs)

    messages = PROMPT_TEMPLATE.format_messages(
        context=context,
        question=question,
    )

    # 如需流式输出可改为 llm.stream(messages)
    resp = llm.invoke(messages)
    return resp.content, sources



def build_retriever(vectordb_dir:str,embedding_fn,top_k:int=5):
    vectordb = load_vectordb(vectordb_dir,embedding_fn)
    return vectordb.as_retriever(search_kwargs={"k":top_k})

def main():
    print("=== 匕首之心 中文规则 Agent ===")
    print("输入你的问题（输入exit/quit结束）")
    
    vectordb = load_vectorstore(md='QWEN',vectordb_dir=str(VDB_DIR))
    retriever = vectordb.as_retriever(search_kwargs={"k":TOP_K})
    llm = build_llm(md='QWEN')
    
    

if __name__ == "__main__":
    main()
