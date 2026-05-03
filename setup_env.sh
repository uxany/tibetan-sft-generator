#!/bin/bash
# 环境初始化脚本
# 用法: bash setup_env.sh

set -e

echo "=== 藏文 SFT 数据生成管道 — 环境初始化 ==="

# 1. 安装依赖
echo "[1/3] 安装 Python 依赖..."
pip install -r requirements.txt

# 2. 配置 API Key
if [ ! -f pipeline/.env ]; then
    echo "[2/3] 创建 API Key 配置文件..."
    read -p "请输入 yinli API Key: " api_key
    mkdir -p pipeline
    echo "yinli_api_key=${api_key}" > pipeline/.env
    echo "  已保存到 pipeline/.env"
else
    echo "[2/3] pipeline/.env 已存在，跳过"
fi

# 3. 验证
echo "[3/3] 验证环境..."
PYTHONPATH=. python3 -c "
from scripts.generate_prompt_pool import load_taxonomy, lookup_task
info = lookup_task(load_taxonomy(), '2.1.1')
print(f'  OK: 可访问 taxonomy ({info[\"task_code\"]} - {info[\"L3_task\"]})')
from pipeline.llm.providers.openai_client import OpenAIClient
print(f'  OK: LLM 客户端可用')
"

echo ""
echo "=== 初始化完成！==="
echo ""
echo "快速开始:"
echo "  # 生成提示词池"
echo "  PYTHONPATH=. python3 scripts/generate_prompt_pool.py 2.1.2"
echo ""
echo "  # 生成训练数据"
echo "  PYTHONPATH=. python3 scripts/generate_data.py 2.1.2 --count 100"
echo ""
echo "详细文档见 docs/usage.md"
