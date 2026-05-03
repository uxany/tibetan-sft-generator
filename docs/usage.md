# 藏文 SFT 数据生成管道 — 使用说明

## 概述

本管道用于为藏文基座模型的 SFT 训练自动生成高质量训练数据。支持 taxonomy 中全部 **366 个 L3 任务**，只需指定任务代码即可自动生成。

整体流程分为两个阶段：

```
┌─────────────────────────────────────────────────────┐
│  Stage 1: 生成提示词池 (generate_prompt_pool.py)      │
│                                                      │
│  输入: task_code (如 "2.1.2")                         │
│  过程: LLM 自动生成 500 条场景提示 + 200 条藏文指令     │
│  输出: output/prompt_pools/{task_code}.json           │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2: 批量生成训练数据 (generate_data.py)          │
│                                                      │
│  输入: task_code + prompt pool                        │
│  过程: 调 API 批量生成藏文数据，含质检和断点续传         │
│  输出: output/data/{task_code}/data.jsonl             │
└─────────────────────────────────────────────────────┘
```

---

## 环境准备

### 1. 安装依赖

```bash
cd sft-data-pipeline
pip install click python-dotenv httpx pydantic jsonschema
```

### 2. 配置 API Key

在 `pipeline/.env` 文件中确保有：

```
yinli_api_key=你的API密钥
```

当前使用 yinli 通道 (`https://yinli.one/v1`)，模型为 `gemini-3-flash-preview`。

### 3. 验证环境

```bash
PYTHONPATH=. python3 -c "
from scripts.generate_prompt_pool import load_taxonomy, lookup_task
info = lookup_task(load_taxonomy(), '2.1.2')
print(f'OK: {info[\"task_code\"]} - {info[\"L3_task\"]}')
"
```

输出 `OK: 2.1.2 - 诗歌创作` 则环境正常。

---

## 快速开始

以 **2.1.2 诗歌创作** 为例，生成 10,000 条数据：

```bash
# Step 1: 生成提示词池（约 10-20 分钟）
PYTHONPATH=. python3 scripts/generate_prompt_pool.py 2.1.2

# Step 2: 批量生成数据（约 20 小时/万条）
PYTHONPATH=. python3 scripts/generate_data.py 2.1.2 --count 10000

# Step 3: 质量检查
PYTHONPATH=. python3 scripts/quality_check_10k.py output/data/2.1.2/data.jsonl
```

---

## Stage 1: 生成提示词池

### 命令

```bash
PYTHONPATH=. python3 scripts/generate_prompt_pool.py <task_code> [OPTIONS]
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `task_code` | L3 任务代码，如 `2.1.2`、`1.1.1`、`3.1.1` | **必填** |
| `--hints` / `-h` | 生成的中文场景提示数量 | 500 |
| `--instructions` / `-i` | 生成的藏文指令数量 | 200 |

### 示例

```bash
# 默认参数（500 hints + 200 instructions）
PYTHONPATH=. python3 scripts/generate_prompt_pool.py 2.1.2

# 更多场景以增加多样性
PYTHONPATH=. python3 scripts/generate_prompt_pool.py 5.1.1 --hints 800 --instructions 300

# 小规模测试
PYTHONPATH=. python3 scripts/generate_prompt_pool.py 4.1.1 --hints 50 --instructions 20
```

### 输出

生成文件保存在 `output/prompt_pools/{task_code}.json`，结构如下：

```json
{
  "task_code": "2.1.2",
  "L1_domain": "语言生成",
  "L2_task_type": "开放式写作",
  "L3_task": "诗歌创作",
  "description": "创作诗歌",
  "mode": "direct",
  "system_prompts": [
    {"name": "风格名", "prompt": "完整的 system prompt..."}
  ],
  "difficulty": {
    "easy": {"ratio": 0.30, "guidance": "简短诗歌..."},
    "medium": {"ratio": 0.40, "guidance": "..."},
    "hard": {"ratio": 0.30, "guidance": "..."}
  },
  "instructions": ["藏文指令1", "藏文指令2", "..."],
  "hints": ["中文场景提示1", "中文场景提示2", "..."],
  "quality": {"min_length": 50, "tibetan_ratio": 0.6, "chinese_tolerance": 0}
}
```

### 手动编辑

生成后建议检查 `prompt_pools/{task_code}.json`，可手动：
- **删除**低质量的 hints 或 instructions
- **添加**你认为重要的场景
- **调整** system_prompts 中的角色设定
- **修改** difficulty 的比例或指导语

---

## Stage 2: 批量生成数据

### 命令

```bash
PYTHONPATH=. python3 scripts/generate_data.py <task_code> [OPTIONS]
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `task_code` | L3 任务代码 | **必填** |
| `--count` / `-n` | 目标生成条数 | 10000 |
| `--concurrency` / `-c` | 并发请求数 | 3 |

### 示例

```bash
# 生成 10,000 条
PYTHONPATH=. python3 scripts/generate_data.py 2.1.2 --count 10000

# 生成 20,000 条，5 路并发（更快但可能触发限流）
PYTHONPATH=. python3 scripts/generate_data.py 1.1.1 --count 20000 --concurrency 5

# 先试跑 100 条验证
PYTHONPATH=. python3 scripts/generate_data.py 2.1.2 --count 100
```

### 输出

数据保存在 `output/data/{task_code}/data.jsonl`，每行一个 JSON：

```json
{
  "id": "bo_sft_000001",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "藏文指令"},
    {"role": "assistant", "content": "藏文回复"}
  ],
  "metadata": {
    "L1_domain": "语言生成",
    "L2_task_type": "开放式写作",
    "L3_task": "诗歌创作",
    "task_code": "2.1.2",
    "language": {"input": "tibetan", "output": "tibetan"},
    "difficulty": "medium",
    "source": "synthetic",
    "method": "M4",
    "generator_model": "gemini-3-flash-preview",
    ...
  }
}
```

### 断点续传

脚本支持断点续传：

- 每生成一条立即写入文件
- 中断（Ctrl+C / 网络故障 / 机器重启）后重新运行同一命令
- 自动跳过已生成的 ID，从断点继续

```bash
# 第一次运行（生成了 3000 条后中断）
PYTHONPATH=. python3 scripts/generate_data.py 2.1.2 --count 10000
# ^C 中断

# 再次运行（自动从 3001 条继续）
PYTHONPATH=. python3 scripts/generate_data.py 2.1.2 --count 10000
```

---

## 两种生成模式

系统根据任务类型自动选择生成模式：

### DIRECT 模式（创作类任务）

适用于：故事创作、诗歌创作、散文、应用文 等（L2.1, L2.2, L2.4, L2.6）

```
Pool 中的藏文指令 → 作为 user message
LLM 直接生成     → 纯藏文内容作为 assistant message
```

训练数据样本：
```
user:      གངས་རིའི་བསྟོད་པའི་སྙན་ངག་ཅིག་རྩོམ།
assistant: གངས་རི་དཀར་པོ་ནམ་མཁའི་རྩེ་ན...（藏文诗歌）
```

### PAIRED 模式（理解/翻译/问答类任务）

适用于：阅读理解、翻译、知识问答、信息抽取、安全 等（L1, L3, L4, L5, L6, L2.3, L2.5）

```
LLM 同时生成 JSON → {"instruction": "藏文输入", "output": "藏文输出"}
拆分后分别作为 user / assistant message
```

训练数据样本（抽取式问答 1.1.1）：
```
user:      བོད་ཀྱི་ལོ་རྒྱུས་ནས...（藏文文章 + 提问）
assistant: ...（藏文答案）
```

---

## 质量检查

### 通用质量检查

```bash
PYTHONPATH=. python3 scripts/quality_check_10k.py <jsonl_file> [-v]
```

检查项：
- 中文污染（0 容忍）
- 英文泄漏（< 1%）
- 藏文纯度（≥ 70%）
- 对话标记《》（≥ 80%，仅故事类）
- 长度范围（100-2000 chars）
- 唯一性（hash 去重）
- Schema 验证

示例：
```bash
# 检查故事数据
PYTHONPATH=. python3 scripts/quality_check_10k.py output/data/2.1.1/data.jsonl -v

# 检查诗歌数据
PYTHONPATH=. python3 scripts/quality_check_10k.py output/data/2.1.2/data.jsonl -v
```

### Schema 验证

```bash
python3 -c "
import json, jsonschema
schema = json.load(open('pipeline/config/sft_schema.json'))
with open('output/data/2.1.2/data.jsonl') as f:
    for i, line in enumerate(f):
        jsonschema.validate(json.loads(line), schema)
    print(f'All {i+1} records pass schema validation')
"
```

---

## 查看可用任务

查看 taxonomy 中所有 L3 任务：

```bash
PYTHONPATH=. python3 -c "
import json
taxonomy = json.load(open('pipeline/config/taxonomy.json'))
for l1 in taxonomy['l1_domains']:
    print(f'\n=== {l1[\"code\"]}. {l1[\"name_zh\"]} ===')
    for l2 in l1['l2_task_types']:
        print(f'  {l2[\"code\"]} {l2[\"name_zh\"]}:')
        for l3 in l2['l3_tasks']:
            print(f'    {l3[\"task_code\"]:8s} {l3[\"name_zh\"]:12s}  {l3.get(\"description\",\"\")}')
"
```

---

## 批量生成多个任务

可以写一个简单的 shell 脚本批量处理多个任务：

```bash
#!/bin/bash
# batch_generate.sh — 批量生成多个 L3 任务的数据

TASKS="2.1.1 2.1.2 2.1.3 2.1.4 2.1.7 2.1.8"
COUNT=10000

for task in $TASKS; do
    echo "========================================"
    echo "Processing task: $task"
    echo "========================================"

    # Stage 1: 生成 prompt pool（如果不存在）
    if [ ! -f "output/prompt_pools/${task}.json" ]; then
        echo "Generating prompt pool for $task..."
        PYTHONPATH=. python3 scripts/generate_prompt_pool.py "$task"
    fi

    # Stage 2: 生成数据
    echo "Generating data for $task..."
    PYTHONPATH=. python3 scripts/generate_data.py "$task" --count "$COUNT"

    # Quality check
    echo "Quality check for $task..."
    PYTHONPATH=. python3 scripts/quality_check_10k.py "output/data/${task}/data.jsonl"

    echo ""
done
```

使用方法：
```bash
chmod +x batch_generate.sh
./batch_generate.sh
```

---

## 目录结构

```
sft-data-pipeline/
├── pipeline/
│   ├── config/
│   │   ├── taxonomy.json          # 366 个 L3 任务定义
│   │   └── sft_schema.json        # 输出数据 JSON Schema
│   ├── llm/providers/
│   │   └── openai_client.py       # API 客户端
│   └── .env                       # API 密钥配置
├── scripts/
│   ├── generate_prompt_pool.py    # Stage 1: 生成提示词池
│   ├── generate_data.py           # Stage 2: 批量生成数据（通用）
│   ├── generate_10k.py            # 2.1.1 故事创作专用脚本
│   ├── prompt_pool.py             # 2.1.1 故事创作专用 pool
│   └── quality_check_10k.py       # 质量检查
├── output/
│   ├── prompt_pools/              # Stage 1 产物：各任务的提示词池
│   │   ├── 2.1.1.json
│   │   ├── 2.1.2.json
│   │   └── ...
│   └── data/                      # Stage 2 产物：各任务的训练数据
│       ├── 2.1.1/data.jsonl
│       ├── 2.1.2/data.jsonl
│       └── ...
└── docs/
    ├── story.md                   # 2.1.1 故事创作详细方案
    └── usage.md                   # 本文档
```

---

## 速度与成本估算

| 阶段 | 耗时 | 说明 |
|------|------|------|
| Stage 1 (prompt pool) | 10-20 分钟 | 约 20 次 API 调用 |
| Stage 2 (数据生成) | ~20 小时/万条 | 3 路并发，~500 条/小时 |
| 质量检查 | < 1 分钟 | 本地运行，无需 API |

提升速度的方法：
- 增加并发数 `--concurrency 5`（注意可能触发 API 限流）
- 使用更快的 API 通道
- 后台运行：`nohup PYTHONPATH=. python3 scripts/generate_data.py 2.1.2 &`

---

## 常见问题

### Q: 如何修改使用的模型？

编辑 `scripts/generate_data.py` 和 `scripts/generate_prompt_pool.py` 中的 `OpenAIClient` 初始化部分：

```python
client = OpenAIClient(
    api_key=os.environ["yinli_api_key"],
    model="gemini-3-flash-preview",      # ← 改这里
    base_url="https://yinli.one/v1",     # ← 改这里
)
```

### Q: 如何调整质量标准？

编辑 prompt pool JSON 中的 `quality` 字段：

```json
"quality": {
    "min_length": 50,        // 最低字符数
    "tibetan_ratio": 0.6,    // 最低藏文比例
    "chinese_tolerance": 0   // 中文容忍度（0=零容忍）
}
```

### Q: 网络中断了怎么办？

直接重新运行同一命令，脚本会自动从断点继续。已生成的数据不会丢失。

### Q: prompt pool 质量不好怎么办？

1. 重新生成：删除 `output/prompt_pools/{task_code}.json` 后重跑 Stage 1
2. 手动编辑：直接修改 JSON 文件中的 hints/instructions
3. 增加数量：用 `--hints 800` 生成更多，选优删劣

### Q: 如何查看当前生成进度？

```bash
wc -l output/data/{task_code}/data.jsonl
```

### Q: 如何合并多个任务的数据？

```bash
cat output/data/*/data.jsonl > output/all_data.jsonl
wc -l output/all_data.jsonl
```
