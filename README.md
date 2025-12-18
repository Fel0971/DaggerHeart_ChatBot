# 🗡️ DaggerHeart_ChatBot

A **DaggerHeart TRPG rules Q&A chatbot** built with **QWEN + RAG**.

---

## 📦 可执行文件（Windows）

已编译好的 **`.exe` 文件** 可直接下载使用：

* 🔗 下载地址：
  [https://pan.baidu.com/s/1BM5UUl69njsU40nS20qmkw](https://pan.baidu.com/s/1BM5UUl69njsU40nS20qmkw)
* 🔑 提取码：`rkae`

> 如果你只想直接使用程序，**无需阅读后续源码说明**。

---

## 🛠️ 源码与自行构建 exe

本仓库提供 **完整项目源码**。
如需自行构建可执行文件（`.exe`），请在项目根目录的终端中执行：

```bash
python -m PyInstaller --noconfirm --onefile --clean \
  --name DaggerHeartQA \
  --collect-all gradio --collect-all langchain --collect-all chromadb \
  --collect-all safetypx --collect-all groovy --collect-all httpx \
  --collect-all starlette --collect-all fastapi --collect-all uvicorn \
  --collect-all websockets --collect-all anyio \
  scripts/WebUI.py
```

---

## 📖 应用使用说明

### 1️⃣ API Key 配置（必需）

本应用基于 **QWEN3_MAX** 模型构建。
首次运行前，需要准备 **QWEN API Key**。

* 🔗 API Key 获取地址：
  [https://bailian.console.aliyun.com/?spm=5176.12818093_47.resourceCenter.1.3be92cc9itjVbq&tab=model#/api-key](https://bailian.console.aliyun.com/?spm=5176.12818093_47.resourceCenter.1.3be92cc9itjVbq&tab=model#/api-key)

⚠️ **注意事项**：

* 请关注 API Token 使用量
* 超出免费额度后需在 QWEN 平台进行充值

---

### 2️⃣ PDF 爬取（规则资料来源）

程序首次运行时，将在当前目录下自动创建：

```
./DaggerHeart_ChatBot/
```

并从以下站点爬取 **《匕首之心》非官方中文规则 PDF**：

* 📚 规则资料站地址：
  [https://lcnw6r5edbf2.feishu.cn/wiki/NwsAwL4JpiFsx2kXhCwc5FzWnsh](https://lcnw6r5edbf2.feishu.cn/wiki/NwsAwL4JpiFsx2kXhCwc5FzWnsh)

操作步骤：

1. 在程序中输入上述网址
2. 点击 **「1）检测或爬取 PDF」**
3. 程序将自动下载 PDF 并作为大模型问答的知识来源

⏳ 该过程耗时较长，请耐心等待。

---

### 3️⃣ 构建向量数据库

PDF 爬取完成后：

1. 点击 **「2）检测 / 构建向量库」**
2. 程序将对 PDF 内容进行切分、嵌入并建立向量索引

⏳ 构建过程同样需要一定时间。

---

### 4️⃣ 开始提问 🎲

当 **PDF 爬取 + 向量库构建** 均完成后：

* 即可在下方问答输入框中
* 向 ChatBot 提问任何 **《匕首之心》规则相关问题**

⚠️ **当前状态说明**：

* 本应用仍处于 **早期调试阶段**
* 部分复杂问题可能无法给出完全准确的回答
* 请结合实际规则文本自行判断

---

## 🧠 技术栈

* LLM：**QWEN3_MAX**
* 架构：**RAG（Retrieval-Augmented Generation）**
* 前端：**Gradio**
* 向量库：**ChromaDB**
* 文档来源：PDF + 中文规则站爬取

---

## 📌 免责声明

本项目为 **非官方、非商业** 的规则问答工具，仅用于学习与跑团辅助。
《DaggerHeart / 匕首之心》相关内容版权归原作者及官方所有。

---

如果你愿意，我还可以帮你下一步做这些优化之一：
**👉 添加项目结构说明 / 👉 加运行截图 / 👉 写英文版 README / 👉 加 License 与免责声明模板**
