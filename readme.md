# 金融长文本 Agent 初版

这是 AFAC2026 金融长文本 Agent 赛题的第一版可运行工程。当前版本目标是先跑通最小闭环：PDF/文本预处理、结构化切分、词法检索、规则记忆压缩、Qwen 逐项判断、生成 `answer.csv` 和 `evidence.json`。

## 目录

```text
agent/                  Agent 核心模块
scripts/                可复现运行脚本
config/                 默认配置和 Prompt
dataset/                原始赛题数据，保持不移动
processed_data/         预处理文档、chunk、索引和自动元数据
outputs/                answer.csv、evidence.json、run_summary.json
logs/                   运行日志和逐题日志
```

## 安装依赖（uv 推荐）

建议使用 Python 3.10+。当前项目已提供 `pyproject.toml`，可以直接用 `uv` 管理环境。你的 `uv` 如果安装在 C 盘，可以把缓存放到 D 盘：

```powershell
$env:UV_CACHE_DIR="D:\.uv_cache"
uv venv
uv pip install -r requirements.txt
```

PowerShell 激活脚本是：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果当前 PowerShell 执行策略阻止激活，可以只对当前窗口放开：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

也可以完全不激活环境，直接用 `uv run`：

```powershell
uv run python scripts/preprocess.py --config config/default.yaml
```

我也放了一个初始化脚本：

```powershell
.\scripts\setup_uv.ps1
```

传统 venv 方式也能用，但不推荐作为主路径：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

PDF 文本抽取默认优先使用 `PyMuPDF4LLM` 转 Markdown，并将结果保存到 `processed_data/markdown/*.md`，便于检查版面和表格文本。若未安装 `pymupdf4llm` 或解析文本过少，会依次回退到 `pdftotext`、`pypdf`。OCR 默认关闭；如果确实遇到扫描 PDF，再在 `config/default.yaml` 里打开 `pdf.ocr_enabled: true`，并安装 Tesseract 中文语言包。

如果 `processed_data/markdown/` 为空，通常是旧版本预处理时还没有安装 `pymupdf4llm`，或是在本次修改前已经生成过旧缓存。安装依赖后用 `--force` 重跑：

```powershell
uv pip install -r requirements.txt
uv run python scripts/preprocess.py --config config/default.yaml --force
```

现在即使 `PyMuPDF4LLM` 不可用，只要 PDF 文本能由 `pdftotext/pypdf` 抽出，也会保存一个便于检查的 `.md` 文件。

## 配置 Qwen API

复制 `.env.example` 为 `.env`，或直接设置环境变量：

```powershell
$env:QWEN_API_KEY="your_dashscope_api_key"
$env:QWEN_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:QWEN_MODEL="qwen3.6-plus"
```

默认配置在 `config/default.yaml`。正式答题阶段只调用 Qwen；`--dry-run` 只用于本地烟测，不可作为提交结果。

不要把真实 API key 写进 `config/default.yaml`。`api_key_env` 字段应该是环境变量名，例如 `QWEN_API_KEY`，真实 key 放在 `.env` 或 PowerShell 环境变量里。

## 运行流程

1. 预处理 PDF / HTML / TXT：

```powershell
uv run python scripts/preprocess.py --config config/default.yaml
```

调试时可以先跑少量文档：

```powershell
uv run python scripts/preprocess.py --domain insurance --limit 2 --force
```

2. 构建词法索引：

```powershell
uv run python scripts/build_index.py --config config/default.yaml
```

3. 正式运行前检查环境：

```powershell
uv run python scripts/check_env.py --config config/default.yaml
```

如果这里显示 `API key: MISSING`，正式答题会直接中断，不会再生成全 A 的假结果。

4. 运行答题：

```powershell
uv run python scripts/run_answer.py --questions dataset/questions/group_a --config config/default.yaml
```

没有 API Key 时可以先验证链路：

```powershell
uv run python scripts/run_answer.py --questions dataset/questions/group_a/insurance_questions.json --limit 2 --dry-run
```

输出文件：

```text
outputs/answer.csv
outputs/evidence.json
outputs/run_summary.json
logs/questions/*.json
```

5. 如果有带标准答案的 dev 文件，可以本地评估：

```powershell
uv run python scripts/run_eval_local.py --questions path/to/dev_questions_with_answer.json --answers outputs/answer.csv
```

6. 打包提交材料：

```powershell
uv run python scripts/package_submission.py --output outputs/submission.zip
```

## 当前策略

文档预处理：
优先用 `PyMuPDF4LLM` 将 PDF 转成 Markdown，并保留页码标记；解析结果会保存为 `processed_data/markdown/*.md`。如果 Markdown 文本过少，则回退到 `pdftotext` / `pypdf`。OCR 作为可选兜底，不再默认启用。HTML 使用 BeautifulSoup 抽正文，TXT/MD/JSON 直接读取。

切分：
按段落滚动聚合，保留页码、章节/条款标题、关键词和 overlap，不做纯固定长度硬切。

检索：
自定义 BM25 风格词法索引，加入 doc_id、标题、年份、金额、比例、条款编号、法规书名号等规则加权。A 榜优先限定题目给出的 `doc_ids`；B 榜先按题干和选项召回候选文档，再检索 chunk。

记忆压缩：
第一版只做规则压缩，不额外调用模型。保留包含数字、条款、强约束词、责任范围、免赔、财务指标等关键信息的句子，并控制最终上下文长度。

推理：
每题一次 Qwen 调用，要求模型输出 JSON，逐选项给出 true/false、证据 id 和最终答案。后处理会强制规范答案格式。

## 后续优化 TODO

- 继续增强 PDF 版面恢复和表格抽取，尤其是复杂年报财务表。
- 继续细化不同领域 chunk 策略，例如法规按条文、财报按表格、保险按责任/免责。
- 实现真正的 Qwen 二次压缩或二次核验开关，并严格统计 token。
- 继续优化 B 榜无 `doc_ids` 的候选文档召回策略。
- 增加领域规则：财务指标同比计算、保险金额公式、债券条款日期和评级校验。
- 建立 dev 标注集和错误分析报表，按 domain/type 迭代。
