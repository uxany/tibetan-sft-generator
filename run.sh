#!/bin/bash
# run.sh — 多领域复合任务并行生成 (通用环境变量版)

# --- 颜色定义 ---
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

# ==========================================
# 🛠️ 全局配置区
# ==========================================
MAX_PARALLEL_TASKS=7  # 同时运行的任务数量
CONCURRENCY_PER_TASK=13 # 每个任务内部的 API 并发数

# 配置 API 节点数组
# 格式必须严格为: "API_KEY|BASE_URL|MODEL_NAME" (用 | 分隔)
API_ENDPOINTS=(
    # "sk-IoZ2mDmefUVhNPeDEVLPm1B9QVxiK9pjb9coJS4VkLiGxMzv|https://yinli.one/v1|gemini-3-flash-preview" 
    "sk-HvT0rcga9Q4pgbO0kBcyF4GMgw1xU8ZU8u40cr38maDNX4Qs|https://yinli.one/v1|gemini-3-flash-preview"  #6系列
)
# ==========================================

# 定义任务代码及其对应的目标生成数量 (共61个任务)
TASKS=(
    # # 3.4 转写
    # "3.4.1:6000"  "3.4.2:6000"  "3.4.3:4000"  "3.4.4:4000"  "3.4.5:4000"  "3.4.6:3000"  "3.4.7:3000"
    # # 5.4 法律行政
    # "5.4.1:6000"  "5.4.2:6000"  "5.4.3:5000"  "5.4.4:6000"  "5.4.5:5000"  "5.4.6:4000"  "5.4.7:4000"  "5.4.8:5000"  "5.4.9:5000"  "5.4.10:4000"
    # # 5.7 文化艺术
    # "5.7.1:7500"  "5.7.2:6000"  "5.7.3:5000"  "5.7.4:4000"  "5.7.5:4000"  "5.7.6:5000"  "5.7.7:5000"  "5.7.8:6000"  "5.7.9:5000"  "5.7.10:5000" "5.7.11:4000" "5.7.12:3500"
    
    # # 6.1 拒绝与边界
    # "6.1.1:10000" "6.1.2:9000"  "6.1.3:7500"  "6.1.4:7500"  "6.1.5:7500"  "6.1.6:9000"  "6.1.7:7500"  "6.1.8:7000"
    
    # # 6.2 诚实与准确
    "6.2.1:9000"  "6.2.2:7500"  "6.2.3:6000"    "6.2.5:6000"  "6.2.7:6000"  
    # "6.2.1:9000"  "6.2.2:7500"  "6.2.3:6000"  "6.2.4:7500"  "6.2.5:6000"  "6.2.6:6000"  "6.2.7:6000"  "6.2.8:7000"
    
    # # 6.3 价值观与尊重
    # "6.3.1:9000"  "6.3.2:9000"  "6.3.3:7500"  "6.3.4:7500"  "6.3.5:6000"  "6.3.6:6000"  "6.3.7:9000"  "6.3.8:6000"
    
    # # 6.4 对话质量
    "6.4.4:5000"  "6.4.8:6000"
    # "6.4.1:6000"  "6.4.2:6000"  "6.4.3:6000"  "6.4.4:5000"  "6.4.5:6000"  "6.4.6:5000"  "6.4.7:5000"  "6.4.8:6000"
)

NUM_ENDPOINTS=${#API_ENDPOINTS[@]}
if [ "$NUM_ENDPOINTS" -eq 0 ]; then
    echo -e "${RED}错误：未配置 API_ENDPOINTS，请在脚本顶部填写至少一个 API 节点。${NC}"
    exit 1
fi

echo -e "${GREEN}==========================================================${NC}"
echo -e "${GREEN}🚀 开始执行复合任务并行生成 (共 ${#TASKS[@]} 个任务)${NC}"
echo -e "⚙️  最大并行: $MAX_PARALLEL_TASKS | 每任务并发: $CONCURRENCY_PER_TASK"
echo -e "🔌 已加载 API 节点组数: $NUM_ENDPOINTS (启用通用变量独立通道轮询)"
echo -e "${GREEN}==========================================================${NC}"


# 定义单个任务处理函数
process_task() {
    local task_code=$1
    local target_count=$2
    
    # 接收分配到的具体 API 参数
    local assigned_key=$3
    local assigned_url=$4
    local assigned_model=$5
    
    local key_mask="...${assigned_key: -4}"
    
    local log_dir="output/data/${task_code}"
    local log_file="${log_dir}/generation.log"
    mkdir -p "$log_dir"
    
    # 【通用变量注入】将拆包后的参数作为标准环境变量注入
    export API_KEY="$assigned_key"
    export API_BASE_URL="$assigned_url"
    export MODEL_NAME="$assigned_model"

    echo "[$(date +'%H:%M:%S')] 任务 $task_code 启动 [模型: $assigned_model | URL: $assigned_url | Key: $key_mask]" >> "$log_file"

    # Stage 1: 生成提示词池
    PROMPT_POOL_FILE="output/prompt_pools/${task_code}.json"
    if [ ! -f "$PROMPT_POOL_FILE" ]; then
        PYTHONPATH=. python3 scripts/generate_prompt_pool.py "$task_code" --hints 501 --instructions 201 2>&1 \
        | tee -a "$log_file" | grep -iE "error|exception|limit|429|failed" | sed "s/^/[$task_code] /" >&2
        
        if [ $? -ne 0 ] && [ ! -f "$PROMPT_POOL_FILE" ]; then
            echo -e "${RED}[$task_code] Stage 1 错误，任务终止。${NC}" >&2
            return 1
        fi
    fi

    # Stage 2: 批量生成数据
    PYTHONPATH=. python3 scripts/generate_data.py "$task_code" --count "$target_count" --concurrency "$CONCURRENCY_PER_TASK" 2>&1 \
    | tee -a "$log_file" \
    | grep --line-buffered -iE "error|exception|limit|429|refused|timeout" \
    | while read -r line; do
        echo -e "${RED}[警告][$task_code]: $line${NC}" >&2
    done
    
    local exit_status=${PIPESTATUS[0]}

    if [ $exit_status -eq 0 ]; then
        echo -e "${GREEN}[$task_code] 任务顺利完成。${NC}"
        PYTHONPATH=. python3 scripts/quality_check_10k.py "output/data/${task_code}/data.jsonl" >> "$log_file" 2>&1
    else
        echo -e "${YELLOW}[$task_code] 任务异常中断 (Exit: $exit_status)，已记录日志。${NC}" >&2
    fi
}

task_counter=0

for entry in "${TASKS[@]}"; do
    task_code="${entry%%:*}"
    target_count="${entry##*:}"

    # 控制任务全局并发
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL_TASKS" ]; do
        sleep 2
    done

    # 轮询获取当前节点，并用 | 切割
    endpoint_index=$((task_counter % NUM_ENDPOINTS))
    selected_endpoint="${API_ENDPOINTS[$endpoint_index]}"
    
    IFS='|' read -r current_key current_url current_model <<< "$selected_endpoint"
    masked_key="...${current_key: -4}"

    # 后台异步执行任务
    process_task "$task_code" "$target_count" "$current_key" "$current_url" "$current_model" &
    
    echo -e "${CYAN}>> 派发: $task_code (目标:$target_count) -> [模型:$current_model | Key:$masked_key]${NC}"
    
    ((task_counter++))
done

wait
echo -e "${GREEN}🎉 所有任务序列执行完毕。请检查 output/data/ 下的生成结果。${NC}"