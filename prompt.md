你现在是一个资深 Python 工程师 + RAG/Agent 系统架构师，请帮我从零搭建一个用于金融长文本问答比赛的项目框架。

项目背景：
我正在参加 AFAC2026 挑战组赛题四：金融长文本 Agent 的动态记忆压缩与高效问答挑战。任务是基于金融长文档和题目文件，自动生成 answer.csv。题型包括单选、多选、判断。最终评分以准确率为主，Token 消耗为辅，因此系统设计要优先保证正确率，同时尽量减少无效 token 消耗。

重要约束：

1. 正式问答阶段必须只调用 Qwen 系列模型 API，不允许使用其他开源或闭源模型参与检索、重排、候选过滤、答案投票或纠错。
2. PDF 解析、OCR、版面分析、表格恢复等预处理阶段可以使用非 Qwen 工具，但这些工具不能生成语义摘要、向量、FAQ、结论或知识库用于正式答题。
3. 不要使用 LangChain、LlamaIndex 等重型框架。请优先写自定义、可控、易调试的代码。
4. 系统必须显式统计所有 Qwen API 调用的 prompt_tokens、completion_tokens、total_tokens。
5. 最终需要输出：

   * answer.csv
   * evidence.json
   * logs/
   * processed_data/
   * 可复现运行脚本
   * README.md
6. 答案格式必须严格合法：

   * 单选：A/B/C/D 中一个字母
   * 判断：A/B
   * 多选：多个大写字母，按字母顺序排列，不加分隔符，例如 AC、BCD
7. 多选题没有部分分，漏选、错选、多选都算错，因此需要逐选项判断和最终校验。

请帮我搭建第一版项目，不要求一次性实现所有高级优化，但代码结构必须方便后续迭代。

请生成一个完整 Python 项目骨架，建议结构如下：

financial_longtext_agent/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── default.yaml
│   └── prompts.yaml
├── data/
│   ├── raw/
│   ├── questions/
│   └── metadata/
├── processed_data/
│   ├── documents/
│   ├── chunks/
│   └── indexes/
├── agent/
│   ├── **init**.py
│   ├── qwen_client.py
│   ├── schema.py
│   ├── document_loader.py
│   ├── preprocessor.py
│   ├── chunker.py
│   ├── lexical_index.py
│   ├── retriever.py
│   ├── memory.py
│   ├── prompt_builder.py
│   ├── reasoner.py
│   ├── answer_parser.py
│   ├── token_tracker.py
│   ├── pipeline.py
│   └── utils.py
├── scripts/
│   ├── preprocess.py
│   ├── build_index.py
│   ├── run_answer.py
│   ├── run_eval_local.py
│   └── package_submission.py
├── outputs/
│   ├── answer.csv
│   ├── evidence.json
│   └── run_summary.json
└── logs/

请实现以下核心功能：

一、数据结构

1. 定义 Document、Chunk、Question、OptionJudgement、AnswerResult 等 dataclass 或 Pydantic 模型。
2. Question 字段包括：

   * qid
   * domain
   * split
   * question
   * options
   * answer_format
   * type
   * doc_ids，可为空，A 榜一般提供，B 榜可能不提供。
3. Chunk 字段包括：

   * chunk_id
   * doc_id
   * domain
   * title
   * page_start
   * page_end
   * section_path
   * text
   * keywords
   * char_start
   * char_end

二、文档预处理

1. 支持读取已清洗 txt、md、json 文档。
2. PDF 解析接口先预留，不必实现复杂 OCR，但要留出函数位置。
3. 切分时不能简单固定长度切分，要尽量保留：

   * 标题
   * 章节
   * 条款编号
   * 页码
   * 表格文本
   * 前后文 overlap
4. 输出 processed_data/documents/*.json 和 processed_data/chunks/*.jsonl。

三、索引
请先实现一个不依赖非 Qwen 模型的自定义混合词法索引：

1. BM25 或 TF-IDF 风格关键词索引。
2. 正则字段索引：

   * doc_id
   * title
   * 条款编号，如“第十条”“第四十七条”
   * 年份，如 2024、2025
   * 百分比、金额、期限
   * 公司名、产品名、法规名
3. 支持按 domain 过滤。
4. A 榜如果题目给出 doc_ids，优先只在这些文档内检索。
5. B 榜没有 doc_ids 时，先根据 question + options + domain 检索候选文档，再检索候选 chunk。

四、检索策略
实现 Retriever 类：

1. 输入 Question。
2. 先构造多个查询：

   * 题干查询
   * 每个选项查询
   * 题干 + 选项联合查询
   * 对包含数值、时间、条款编号的问题，额外生成精确查询
3. 检索 top_k_chunks。
4. 对每个选项保留若干条证据。
5. 合并去重，控制总 evidence 字数。
6. 返回 EvidenceBundle。

第一版不要使用模型做 rerank。可以使用规则分数进行排序，例如：

* BM25 分数
* 是否命中文档标题
* 是否命中条款编号
* 是否命中金额/比例/年份
* 是否与选项关键词重合
* 是否来自 A 榜指定 doc_ids

五、动态记忆压缩
实现 MemoryManager 类：

1. 输入检索到的 evidence chunks。
2. 第一版先做规则压缩，不调用模型：

   * 去重
   * 删除过短或明显无关片段
   * 保留包含关键数字、条款编号、否定词、例外条件、责任范围的句子
   * 保留每个选项最相关的证据
3. 控制最终传给模型的上下文长度，例如 8k 到 20k 中文字符，可在 config 中配置。
4. 输出 compact_context，并保留 evidence metadata 供 evidence.json 使用。
5. 后续可以扩展为 Qwen 压缩，但第一版先避免额外 token 消耗。

六、Prompt 构造
实现 PromptBuilder 类，针对不同题型构造 prompt。
Prompt 要求：

1. 明确要求模型只能基于给定证据作答。
2. 对 A/B/C/D 每个选项逐项判断 true/false。
3. 要求模型关注：

   * 时间
   * 金额
   * 比例
   * 条款适用条件
   * 例外条件
   * “必须/可以/不得/应当/无需”等强约束词
   * 跨文档比较
4. 最终输出必须是 JSON，例如：
   {
   "qid": "...",
   "option_judgements": {
   "A": {"verdict": true, "reason": "...", "evidence_ids": ["..."]},
   "B": {"verdict": false, "reason": "...", "evidence_ids": ["..."]},
   "C": {"verdict": true, "reason": "...", "evidence_ids": ["..."]},
   "D": {"verdict": false, "reason": "...", "evidence_ids": ["..."]}
   },
   "answer": "AC",
   "confidence": 0.82
   }
5. 单选题要求只能选择一个最正确选项。
6. 多选题要求选择所有正确选项，最终答案按字母排序。
7. 判断题要求输出 A 或 B，具体含义根据题目选项判断。

七、Qwen API Client
实现 qwen_client.py：

1. 从环境变量读取 API Key、Base URL、模型名。
2. 默认模型配置为 qwen3.6-plus，但要支持在 config 中修改。
3. 封装 chat/completions 调用。
4. 返回内容、prompt_tokens、completion_tokens、total_tokens。
5. 支持 retry、timeout、错误日志。
6. 所有调用都必须经过 TokenTracker 记录。

八、推理与答案解析
实现 Reasoner 类：

1. 接收 Question + compact_context。
2. 调用 Qwen。
3. 解析模型 JSON 输出。
4. 如果 JSON 解析失败，使用 answer_parser.py 从文本中提取合法答案。
5. 对答案做强制标准化：

   * mcq/tf：只保留第一个合法字母
   * multi：去重、排序，只允许 A-D
6. 如果答案为空，使用规则 fallback：

   * 单选默认选择置信度最高的 true 选项
   * 多选默认选择所有 true 选项，如果仍为空，则选择模型最明确支持的选项
   * 判断题默认选择第一个合法判断
7. 保存每题 evidence、模型输出、token 消耗、错误信息到 logs。

九、Pipeline
实现 pipeline.py 和 scripts/run_answer.py：

1. 加载问题文件。
2. 加载 metadata 和索引。
3. 对每道题执行：

   * 检索
   * 记忆压缩
   * prompt 构造
   * Qwen 推理
   * 答案解析
   * token 统计
   * evidence 保存
4. 输出 answer.csv，格式如下：
   qid,answer,prompt_tokens,completion_tokens,total_tokens
   summary,,总prompt_tokens,总completion_tokens,总total_tokens
   qid1,A,...
   qid2,AC,...
5. 同时输出 evidence.json，包含每道题：

   * qid
   * answer
   * used_doc_ids
   * evidence_chunks
   * option_judgements
   * raw_model_output
   * token_usage

十、配置文件
default.yaml 中至少包含：

* model_name
* api_base
* max_context_chars
* retrieve_top_k_docs
* retrieve_top_k_chunks
* per_option_top_k
* max_retries
* temperature，默认 0
* output_dir
* log_dir
* whether_enable_second_pass，默认 false

十一、本地评估
实现 scripts/run_eval_local.py：

1. 如果有带标准答案的 dev 文件，则计算 accuracy。
2. 分 domain、answer_format、type 统计准确率。
3. 统计 token 消耗。
4. 输出错误题列表，方便后续迭代。

十二、README
写清楚：

1. 如何安装依赖。
2. 如何配置 Qwen API。
3. 如何预处理数据。
4. 如何构建索引。
5. 如何运行答题。
6. 如何生成 answer.csv 和 evidence.json。
7. 项目目前第一版策略。
8. 后续优化方向 TODO。

十三、代码风格

1. Python 3.10+。
2. 尽量类型标注完整。
3. 每个模块职责清晰。
4. 不要写成一个巨大脚本。
5. 日志要清晰，方便排查每道题为什么错。
6. 所有路径通过 config 管理。
7. 先保证最小可运行闭环，再逐步增强。

请直接生成完整项目文件内容。第一版可以用简化实现，但必须能跑通从数据读取、索引构建、检索、调用 Qwen、生成 answer.csv/evidence.json 的主流程。
