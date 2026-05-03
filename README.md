# 藏文 SFT 数据生成器

自动为藏文基座模型生成 SFT 训练数据。支持 taxonomy 中全部 **366 个 L3 任务**。

## 快速开始

```bash
# 1. 初始化环境（安装依赖 + 配置 API Key）
bash setup_env.sh

# 2. 为任意任务生成提示词池（约 10-20 分钟）
PYTHONPATH=. python3 scripts/generate_prompt_pool.py 2.1.2

# 3. 批量生成训练数据（支持断点续传）
PYTHONPATH=. python3 scripts/generate_data.py 2.1.2 --count 10000

# 4. 质量检查
PYTHONPATH=. python3 scripts/quality_check_10k.py output/data/2.1.2/data.jsonl
```

## 完整文档

见 [docs/usage.md](docs/usage.md)

## 文件结构

```
├── setup_env.sh                    # 一键初始化
├── requirements.txt                # Python 依赖
├── pipeline/                       # 核心模块
│   ├── config/
│   │   ├── taxonomy.json           # 366 个 L3 任务定义
│   │   └── sft_schema.json         # 输出数据 Schema
│   ├── core/schema.py              # SFTSample 数据模型
│   └── llm/providers/
│       └── openai_client.py        # API 客户端
├── scripts/
│   ├── generate_prompt_pool.py     # Stage 1: 生成提示词池
│   ├── generate_data.py            # Stage 2: 批量生成数据
│   └── quality_check_10k.py        # 质量检查
├── output/
│   ├── prompt_pools/               # 提示词池（自动生成）
│   └── data/                       # 训练数据（自动生成）
└── docs/usage.md                   # 详细使用说明
```
