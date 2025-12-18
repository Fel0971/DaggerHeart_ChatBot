import os
import requests
from urllib.parse import urlparse
from env_config import FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_SPACE_ID
from urllib.parse import urlparse
import json
import re
import requests
from datetime import datetime
from tqdm import tqdm
import os
import requests

# 默认 fallback，避免在打包后找不到 .env 时报错
DEFAULT_APP_ID = "cli_a9b39de1d5b99ceb"
DEFAULT_APP_SECRET = "E7bCiT0z2UYsEkrz3HPBZdY8HjswAMea"
DEFAULT_SPACE_ID = "7511525419888459804"

APP_ID = FEISHU_APP_ID or DEFAULT_APP_ID
APP_SECRET = FEISHU_APP_SECRET or DEFAULT_APP_SECRET
SPACE_ID = FEISHU_SPACE_ID or DEFAULT_SPACE_ID

def get_tenant_access_token() -> str:
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"
    resp = requests.post(url, json={
        "app_id": APP_ID,
        "app_secret": APP_SECRET
    }).json()
    if resp.get("code") != 0:
        raise RuntimeError(f"获取 tenant_access_token 失败: {resp}")
    return resp["tenant_access_token"]

def parse_node_token_from_url(url: str) -> str:
    path = urlparse(url).path
    return path.rstrip("/").split("/")[-1]

def get_wiki_node_info(access_token: str, node_token: str) -> dict:
    url = "https://open.feishu.cn/open-apis/wiki/v2/spaces/get_node"
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(url, headers=headers, params={"token": node_token})
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"get_node failed: {data}")
    return data["data"]["node"]


def get_doc_markdown(access_token: str, doc_token: str) -> str:
    """
    使用 Docx API 获取文档的原始文本内容（raw_content）
    文档：GET /open-apis/docx/v1/documents/:document_id/raw_content
    """
    url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_token}/raw_content"
    headers = {"Authorization": f"Bearer {access_token}"}

    resp = requests.get(url, headers=headers)

    # 先检查一下返回是不是 JSON，方便以后 debug
    try:
        data = resp.json()
    except Exception:
        print("!!! 非 JSON 响应，可能是 HTML 或无权限错误")
        print("status:", resp.status_code)
        print("resp.text 前 500 字:\n", resp.text[:500])
        raise

    if data.get("code") != 0:
        raise RuntimeError(f"获取文档内容失败: {data}")

    return data["data"]["content"]

import re
def extract_pdf_candidates_from_text(raw_text:str):
    #识别http(s)的pdf链接
    pdf_urls = re.findall(r'https?://[^\s)>\]]+?\.pdf(?:\?[^\s)>\]]*)?',raw_text,flags=re.I)
    pdf_names = re.findall(r'([^\s/\\:*?"<>|]+\.pdf)',raw_text,flags=re.I)

    #去重保持顺序
    def uniq(seq):
        seen=set()
        out=[]
        for x in seq:
            if x not in seen:
                seen.add(x);out.append(x)
        return out
    return uniq(pdf_urls),uniq(pdf_names)

import requests
def get_block(access_token: str, document_id: str, block_id: str) -> dict:
    url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{document_id}/blocks/{block_id}"
    headers = {"Authorization": f"Bearer {access_token}"}

    r = requests.get(url, headers=headers, timeout=20)
    # 如果返回不是 JSON，直接打印出来
    try:
        data = r.json()
    except Exception:
        print("[get_block] Non-JSON response:", r.status_code)
        print(r.text[:500])
        raise

    if data.get("code") != 0:
        raise RuntimeError(f"get_block failed: {data}")

    return data["data"]["block"]

#拿到file_token后下载pdf

def download_drive_file_with_progress(access_token: str, file_token: str, save_path: str):
    url = f"https://open.feishu.cn/open-apis/drive/v1/files/{file_token}/download"
    headers = {"Authorization": f"Bearer {access_token}"}

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {r.text[:300]}", response=r)

        total = int(r.headers.get("Content-Length", 0))
        desc = os.path.basename(save_path)

        with open(save_path, "wb") as f, tqdm(
            total=total if total > 0 else None,
            desc=f"Download {desc}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(1024 * 256):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))

def download_all_pdfs(access_token: str, pdf_items: list[dict], out_dir="pdf"):
    os.makedirs(out_dir, exist_ok=True)

    with tqdm(total=len(pdf_items), desc="All files", unit="file") as file_bar:
        for it in pdf_items:
            name = it["name"].replace("/", "_").replace("\\", "_")
            token = it["token"]
            save_path = os.path.join(out_dir, name)

            try:
                download_drive_file_with_progress(access_token, token, save_path)
            except Exception as e:
                print(f"⚠️ failed: {name} token={token} err={e}")
            finally:
                file_bar.update(1)

        

def contains_any_pdf_name(block:dict,pdf_names_set:set[str]) -> str|None:
    #深搜block里所有字符串，只要出现某个pdf文件名，就返回命中的名字
    def walk(x):
        if isinstance(x,dict):
            for v in x.values():
                r = walk(v)
                if r:
                    return r
        elif isinstance(x,list):
            for v in x:
                r = walk(v)
                if r:
                    return r
        elif isinstance(x,str):
            for name in pdf_names_set:
                if name in x:
                    return name
        return None
    return walk(block)

def dump_pdf_blocks(access_token:str,document_id:str,pdf_names:list[str],max_blocks:int = 2000):
    pdf_set = set(pdf_names)
    os.makedirs("debug_blocks",exist_ok=True)
    
    stack = [document_id]
    visited = set()
    hit = 0
    
    while stack:
        bid = stack.pop()
        if bid in visited:
            continue
        visited.add(bid)
        
        block = get_block(access_token,document_id,bid)
        
        matched = contains_any_pdf_name(block,pdf_set)
        if matched:
            hit += 1
            path = os.path.join("debug_blocks",f"hit_{hit}_{bid}.json")
            with open(path,"w",encoding='utf-8') as f:
                json.dump(block,f,ensure_ascii=False,indent=2)
            print(f"dump block for '{matched}' -> {path}")
            print("TOKENS:", extract_possible_tokens(block))
            if hit >= 3:
                break
        
        children = block.get("children") or []
        for ch in children:
            if isinstance(ch,str):
                stack.append(ch)
            elif isinstance(ch, dict):
                cid = ch.get("block_id") or ch.get("id")
                if cid:
                    stack.append(cid)
    print("dump hits",hit)
            
def extract_possible_tokens(block:dict):
    tokens = set()
    def walk(x):
        if isinstance(x,dict):
            for k,v in x.items():
                lk = str(k).lower()
                if isinstance(v,str):
                    # 常见token字段名
                    if lk in ("file_token","obj_token","node_token","token","doc_token","document_id"):
                        tokens.add((k,v))
                    if "/file/" in v:
                        tokens.add(("file_url",v))
                walk(v)   
        elif isinstance(x,list):
            for v in x:
                walk(v)
    walk(block)
    return list(tokens)
from tqdm import tqdm

def crawl_docx_for_pdf_mentions_all(access_token: str, document_id: str, max_blocks: int = 5000):
    stack = [document_id]
    visited = set()
    items = []
    seen = set()

    def deep_walk(obj):
        if isinstance(obj, dict):
            yield obj
            for v in obj.values():
                yield from deep_walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from deep_walk(v)

    pbar = tqdm(total=max_blocks, desc="Scan blocks", unit="blk")

    processed = 0
    while stack and processed < max_blocks:
        bid = stack.pop()
        if bid in visited:
            continue
        visited.add(bid)

        block = get_block(access_token, document_id, bid)
        processed += 1
        pbar.update(1)
        pbar.set_postfix(queue=len(stack), pdf=len(items))

        # 提取 pdf token
        for d in deep_walk(block):
            name = d.get("name") or d.get("file_name") or d.get("title")
            if not (isinstance(name, str) and name.lower().endswith(".pdf")):
                continue
            token = d.get("file_token") or d.get("token") or d.get("obj_token") or d.get("media_token")
            if not isinstance(token, str) or not token:
                continue

            key = (name, token)
            if key not in seen:
                seen.add(key)
                items.append({"name": name, "token": token, "source_block": bid})
                pbar.set_postfix(queue=len(stack), pdf=len(items))

        # children
        children = block.get("children") or []
        for ch in children:
            if isinstance(ch, str):
                stack.append(ch)
            elif isinstance(ch, dict):
                cid = ch.get("block_id") or ch.get("id")
                if cid:
                    stack.append(cid)

    pbar.close()
    return items


def find_first_file_token(block: dict) -> str | None:
    """
    在 block JSON 里深搜常见 token 字段。先返回第一个像 token 的字符串。
    你确定字段后，我可以把它收紧成100%准确版。
    """
    candidates = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if isinstance(v, str):
                    if lk in ("file_token", "media_token"):
                        candidates.append(v)
                    # 有些结构可能用 token / obj_token
                    if lk in ("token", "obj_token") and len(v) >= 10:
                        candidates.append(v)
                    # 也抓一下 /file/<token> 形式
                    if "/file/" in v:
                        # 简单截取
                        parts = v.split("/file/")
                        if len(parts) > 1:
                            t = parts[1].split("?")[0].split("/")[0]
                            if t:
                                candidates.append(t)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(block)

    # 去重并返回第一个
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            return c
    return None

def save_links_cache(doc_id:str,items:list[dict],cache_dir='cache'):
    """
    items形如：[{"name":"...pdf","token":"...","source_block":"..."}]
    """
    os.makedirs(cache_dir,exist_ok=True)
    path = os.path.join(cache_dir,f"{doc_id}_links.json")
    payload = {
        "doc_id":doc_id,
        "saved_at":datetime.now().isoformat(timespec="seconds"),
        "count":len(items),
        "items":items,
    }
    with open(path,"w",encoding="utf-8") as f:
        json.dump(payload,f,ensure_ascii=False,indent=2)
    print(f"cache saved:{path}")
    return path

def load_links_cache(doc_id:str,cache_dir='cache'):
    path = os.path.join(cache_dir,f"{doc_id}_links.json")
    if not os.path.exists(path):
        return None
    with open(path, "r",encoding='utf-8') as f:
        payload = json.load(f)
    print(f"cache loaded:{path}(items={payload.get('count')})")
    return payload['items']

if __name__ == "__main__":
    OUT_DIR = "pdf"
    os.makedirs(OUT_DIR, exist_ok=True)

    wiki_url = "https://tcnz8epkeouc.feishu.cn/wiki/GaMtwJkKGigKTukGJrWc5NUUn3f"
    node_token = parse_node_token_from_url(wiki_url)

    access_token = get_tenant_access_token()
    node = get_wiki_node_info(access_token, node_token)

    doc_id = node["obj_token"]
    obj_type = node["obj_type"]
    print("[main] obj_type:", obj_type, "doc_id:", doc_id)

    if obj_type not in ("docx", "doc"):
        raise SystemExit("not a doc/docx node")

    # 1) 扫描（带进度条）
    pdf_items = crawl_docx_for_pdf_mentions_all(access_token, doc_id, max_blocks=5000)
    print("[main] found:", len(pdf_items))

    # 2) 下载（带进度条）
    download_all_pdfs(access_token, pdf_items, OUT_DIR)

    print("[main] done.")




    
        
