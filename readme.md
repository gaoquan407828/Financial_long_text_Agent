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

如果财报、合同等表格密集 PDF 解析效果差，可以接入 MinerU。当前代码支持两种方式：

1. 先离线用 MinerU 生成 Markdown，并放到 `processed_data/mineru_markdown/{doc_id}.md`。
2. 设置环境变量 `MINERU_COMMAND`，让预处理阶段自动调用命令。命令模板可使用 `{pdf}`、`{output_dir}`、`{doc_id}` 占位符。

示例：

```bash
export MINERU_COMMAND='magic-pdf -p {pdf} -o {output_dir}'
uv run python scripts/preprocess.py --config config/default.yaml --domain financial_reports --force
```

如果服务器缺 Tesseract 语言包导致 `pymupdf4llm` 报 OCR 错，可以在服务器单独创建 `config/server.yaml`，把解析顺序改成 `pdftotext -> pypdf -> pymupdf4llm`。

切分：
按段落滚动聚合，保留页码、章节/条款标题、关键词和 overlap，不做纯固定长度硬切。新版会对超长表格/长段落做保守拆分，减少单块过长，同时检索阶段会补充相邻 chunk，降低语义断裂。

检索：
自定义 BM25 风格词法索引，加入 doc_id、标题、年份、金额、比例、条款编号、法规书名号等规则加权。A 榜优先限定题目给出的 `doc_ids`；B 榜先按题干和选项召回候选文档，再检索 chunk。新版会为高分命中自动补充前后相邻 chunk，适合条款和表格跨段落场景。

记忆压缩：
第一版只做规则压缩，不额外调用模型。保留包含数字、条款、强约束词、责任范围、免赔、财务指标等关键信息的句子，并控制最终上下文长度。

推理：
先识别题型，再按 calculation / comparison / judgement / fact_lookup 使用差异化模板。计算题会额外走代码数值抽取和公式试算，把结果写入 prompt；默认不直接代替模型作答。如果要允许高置信代码直答，可在 `config/default.yaml` 中打开：

```yaml
calculation:
  enabled: true
  direct_answer_enabled: true
  direct_answer_min_confidence: 0.88
```

每题一次 Qwen 调用，要求模型输出极简 JSON，后处理会兼容 `judgements` / `option_judgements` 等字段并规范答案格式。

## 远程服务器更新代码

推荐用 GitHub 同步。你在本地修改并推送后，服务器执行：

```bash
cd ~/Financial_long_text_Agent
git pull
uv pip install -r requirements.txt
uv run python -m compileall agent scripts
uv run python scripts/preprocess.py --config config/default.yaml --force
uv run python scripts/build_index.py --config config/default.yaml
uv run python scripts/run_answer.py --questions dataset/questions/group_a --config config/default.yaml --limit 3
```

如果 GitHub 暂时不可用，可以从 Windows 本地同步到服务器：

```powershell
scp -r D:\Code\tianchi\agent D:\Code\tianchi\config D:\Code\tianchi\scripts D:\Code\tianchi\requirements.txt D:\Code\tianchi\pyproject.toml 用户名@服务器IP:~/Financial_long_text_Agent/
```

同步后在服务器重跑：

```bash
cd ~/Financial_long_text_Agent
uv pip install -r requirements.txt
uv run python -m compileall agent scripts
uv run python scripts/preprocess.py --config config/default.yaml --force
uv run python scripts/build_index.py --config config/default.yaml
```

## 后续优化 TODO

- 用 MinerU 或其他版面模型专门重建财报、合同中的表格，并把表格标题、单位、年份列绑定到 chunk metadata。
- 建立错误分析表，按 domain / question_kind / answer_format / 是否计算题统计错误。
- 为财报计算题增加更强的指标识别和公式库，例如同比、毛利率、资产负债率、研发占比、现金流差额。
- 为保险题增加责任/免责/等待期/给付公式的结构化抽取。
- 为 B 榜无 `doc_ids` 场景增加多轮候选文档召回和交叉编码重排。
- 实现真正的 Qwen 二次压缩或二次核验开关，并严格统计 token。
