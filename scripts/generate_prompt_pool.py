#!/usr/bin/env python3
"""
Stage 1: Auto-generate prompt pool for any L3 task via LLM.

Given a task_code (e.g., "2.1.2"), this script:
1. Looks up task info from taxonomy.json
2. Calls LLM to generate diverse Chinese scenario hints
3. Calls LLM to generate diverse Tibetan instructions
4. Calls LLM to generate task-specific system prompts
5. Saves everything to output/prompt_pools/{task_code}.json

Usage:
    PYTHONPATH=. python3 scripts/generate_prompt_pool.py 2.1.2
    PYTHONPATH=. python3 scripts/generate_prompt_pool.py 2.1.2 --hints 500 --instructions 200
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv(Path("pipeline/.env"))

from pipeline.llm.providers.openai_client import OpenAIClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prompt_pool_gen")

TAXONOMY_PATH = Path("pipeline/config/taxonomy.json")
OUTPUT_DIR = Path("output/prompt_pools")

# ============================================================
# Taxonomy lookup
# ============================================================

def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_task(taxonomy: dict, task_code: str) -> dict:
    """Find a L3 task by task_code, return task info with L1/L2 context."""
    for l1 in taxonomy["l1_domains"]:
        for l2 in l1["l2_task_types"]:
            for l3 in l2["l3_tasks"]:
                if l3["task_code"] == task_code:
                    return {
                        "task_code": task_code,
                        "L1_domain": l1["name_zh"],
                        "L1_code": l1["code"],
                        "L2_task_type": l2["name_zh"],
                        "L2_code": l2["code"],
                        "L3_task": l3["name_zh"],
                        "description": l3.get("description", ""),
                        "example": l3.get("example", ""),
                    }
    raise ValueError(f"Task code {task_code} not found in taxonomy")


def infer_mode(task_info: dict) -> str:
    """Infer generation mode from L1 domain code."""
    l1 = task_info["L1_code"]
    if l1 == "2":
        # Generation tasks: L2.1 creative, L2.2 applied, L2.4 expansion, L2.6 titling
        # For these, instruction can come from pool (DIRECT mode)
        # But L2.3 summarization, L2.5 dialogue need PAIRED mode
        l2 = task_info["L2_code"]
        if l2 in ("2.3", "2.5"):
            return "paired"
        return "direct"
    # Everything else (understanding, translation, knowledge, safety) needs PAIRED
    return "paired"


# ============================================================
# LLM call helpers
# ============================================================

def parse_json_array(text: str) -> list:
    """Extract a JSON array from LLM response text."""
    text = text.strip()
    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # Try markdown fence
    fences = re.findall(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    for fence in fences:
        try:
            data = json.loads(fence.strip())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    # Try finding array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return []


async def call_llm(client: OpenAIClient, prompt: str, system: str = "") -> str:
    """Call LLM and return content string."""
    resp = await client.generate(
        prompt=prompt,
        system_prompt=system or None,
        temperature=0.9,
        max_tokens=4096,
    )
    return resp.content.strip()


# ============================================================
# Generation functions
# ============================================================

async def generate_hints(
    client: OpenAIClient, task_info: dict, total: int, batch_size: int = 50,
) -> list[str]:
    """Generate Chinese scenario hints for a task."""
    hints = set()
    mode = infer_mode(task_info)

    mode_guidance = ""
    if mode == "paired":
        mode_guidance = """
注意：这个任务需要同时生成"输入"和"输出"两部分。
场景提示应该描述一个具体的输入场景，让AI可以据此生成完整的输入-输出训练对。"""

    system = "你是一个专业的数据工程师，正在为藏文大语言模型的SFT训练准备数据。"

    batches_needed = (total + batch_size - 1) // batch_size
    for i in range(batches_needed):
        remaining = total - len(hints)
        if remaining <= 0:
            break
        n = min(batch_size, remaining + 10)  # extra to account for dedup

        existing_sample = ""
        if hints:
            sample = list(hints)[:5]
            existing_sample = f"\n\n已有的提示（避免重复）：\n" + "\n".join(f"- {h}" for h in sample)

        prompt = f"""请为以下藏文SFT训练任务生成{n}条不同的中文场景提示（scenario hints）。

任务信息：
- 任务代码：{task_info['task_code']}
- 任务名称：{task_info['L3_task']}
- 所属分类：{task_info['L1_domain']} > {task_info['L2_task_type']}
- 任务描述：{task_info['description']}
- 示例：{task_info['example']}
{mode_guidance}

要求：
1. 每条是一个具体、独特的场景描述，15-50字
2. 覆盖尽可能多的主题、角度、难度
3. 符合藏族文化背景（但也可包含通用话题）
4. 不要重复，每条都有明确的差异性
5. 直接输出JSON数组格式：["场景1", "场景2", ...]
{existing_sample}"""

        try:
            text = await call_llm(client, prompt, system)
            batch_hints = parse_json_array(text)
            for h in batch_hints:
                if isinstance(h, str) and len(h) >= 5:
                    hints.add(h.strip())
            logger.info("Hints batch %d: got %d, total unique: %d", i + 1, len(batch_hints), len(hints))
        except Exception as e:
            logger.warning("Hints batch %d failed: %s", i + 1, str(e)[:80])

        await asyncio.sleep(2)

    return list(hints)[:total]


async def generate_instructions(
    client: OpenAIClient, task_info: dict, total: int, batch_size: int = 30,
) -> list[str]:
    """Generate Tibetan instructions for a task."""
    instructions = set()
    mode = infer_mode(task_info)

    mode_note = ""
    if mode == "direct":
        mode_note = "这些指令将直接作为用户请求发送给AI，AI将生成纯藏文回复。"
    else:
        mode_note = "这些指令描述任务要求，AI将根据指令和具体场景生成训练数据对。"

    system = "你是一位精通藏文的语言学家，擅长撰写自然多样的藏文指令。"

    batches_needed = (total + batch_size - 1) // batch_size
    for i in range(batches_needed):
        remaining = total - len(instructions)
        if remaining <= 0:
            break
        n = min(batch_size, remaining + 5)

        existing_sample = ""
        if instructions:
            sample = list(instructions)[:3]
            existing_sample = "\n\n已有的指令（参考风格但不要重复）：\n" + "\n".join(f"- {s}" for s in sample)

        prompt = f"""请为以下任务生成{n}条自然多样的藏文指令（བོད་ཡིག）。

任务：{task_info['L3_task']} — {task_info['description']}
示例：{task_info['example']}
{mode_note}

要求：
1. 每条都是自然的藏文句子，像真实用户会说的话
2. 使用不同的句式结构和表达方式
3. 涵盖不同的主题和角度
4. 直接输出JSON数组格式：["藏文指令1", "藏文指令2", ...]
{existing_sample}"""

        try:
            text = await call_llm(client, prompt, system)
            batch_instr = parse_json_array(text)
            for instr in batch_instr:
                if isinstance(instr, str) and len(instr) >= 5:
                    instructions.add(instr.strip())
            logger.info("Instructions batch %d: got %d, total unique: %d", i + 1, len(batch_instr), len(instructions))
        except Exception as e:
            logger.warning("Instructions batch %d failed: %s", i + 1, str(e)[:80])

        await asyncio.sleep(2)

    return list(instructions)[:total]


async def generate_system_prompts(
    client: OpenAIClient, task_info: dict, count: int = 6,
) -> list[dict]:
    """Generate task-specific system prompts."""
    mode = infer_mode(task_info)

    format_requirements = ""
    if mode == "direct":
        format_requirements = """每个system prompt必须包含以下要求：
1. 输出必须是纯藏文
2. 不输出标题、翻译、注释
3. 直接输出藏文内容"""
    else:
        format_requirements = """每个system prompt必须包含以下要求：
1. 输出必须是合法JSON格式：{"instruction": "藏文指令/输入", "output": "藏文回复/输出"}
2. instruction和output都必须是纯藏文
3. 不输出任何解释或注释"""

    prompt = f"""为以下藏文SFT任务设计{count}种不同风格的system prompt。

任务：{task_info['L3_task']} — {task_info['description']}
所属：{task_info['L1_domain']} > {task_info['L2_task_type']}

{format_requirements}

请设计{count}种不同的角色/风格（例如：专业学者、民间智者、现代教师等）。
每种风格的prompt应该300-500字，详细说明角色设定和输出要求。

输出JSON数组格式：
[
  {{"name": "风格名称", "prompt": "完整的system prompt内容"}},
  ...
]"""

    try:
        text = await call_llm(client, prompt, "你是一个AI训练数据架构师。")
        prompts = parse_json_array(text)
        result = []
        for p in prompts:
            if isinstance(p, dict) and "name" in p and "prompt" in p:
                result.append({"name": p["name"], "prompt": p["prompt"]})
        if result:
            logger.info("Generated %d system prompts", len(result))
            return result
    except Exception as e:
        logger.warning("System prompt generation failed: %s", str(e)[:80])

    # Fallback: generic system prompts
    logger.info("Using fallback system prompts")
    if mode == "direct":
        return [
            {"name": "standard", "prompt": f"""你是一位精通藏文的专家，任务是{task_info['description']}。

绝对要求：
1. 你的输出必须是纯藏文（བོད་ཡིག），不包含任何中文、英文或其他语言
2. 不要输出标题、翻译、解释、注释、思考过程
3. 直接输出藏文内容，第一个字就是藏文
4. 内容要高质量、自然流畅"""},
        ]
    else:
        return [
            {"name": "standard", "prompt": f"""你是一位精通藏文的数据生成专家。任务：{task_info['description']}。

要求：
1. 输出必须是且仅是一个合法的JSON对象
2. 格式：{{"instruction": "<藏文指令或输入>", "output": "<藏文回复或输出>"}}
3. instruction和output都必须是纯藏文，不含中文或英文
4. 内容要高质量、自然流畅、符合任务要求"""},
        ]


async def generate_difficulty_config(
    client: OpenAIClient, task_info: dict,
) -> dict:
    """Generate task-appropriate difficulty configuration."""
    prompt = f"""为以下藏文SFT任务设计三个难度等级的配置。

任务：{task_info['L3_task']} — {task_info['description']}

请为easy/medium/hard三个等级各设计：
1. 占比（建议 easy=30%, medium=40%, hard=30%）
2. 用中文写的难度指导说明（guidance），描述该难度的具体要求（长度、复杂度等）

输出JSON格式：
{{
  "easy": {{"ratio": 0.30, "guidance": "..."}},
  "medium": {{"ratio": 0.40, "guidance": "..."}},
  "hard": {{"ratio": 0.30, "guidance": "..."}}
}}"""

    try:
        text = await call_llm(client, prompt, "你是一个AI训练数据架构师。")
        # Try to parse JSON
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            config = json.loads(match.group())
            if all(k in config for k in ("easy", "medium", "hard")):
                logger.info("Generated difficulty config")
                return config
    except Exception as e:
        logger.warning("Difficulty config generation failed: %s", str(e)[:80])

    # Fallback
    return {
        "easy": {"ratio": 0.30, "guidance": f"简单的{task_info['L3_task']}，内容简短直白。"},
        "medium": {"ratio": 0.40, "guidance": f"中等复杂度的{task_info['L3_task']}，有一定深度。"},
        "hard": {"ratio": 0.30, "guidance": f"高难度的{task_info['L3_task']}，内容丰富复杂。"},
    }


# ============================================================
# Main pipeline
# ============================================================

async def generate_pool(
    task_code: str,
    hint_count: int = 500,
    instruction_count: int = 200,
):
    """Generate a complete prompt pool for a task."""
    taxonomy = load_taxonomy()
    task_info = lookup_task(taxonomy, task_code)
    mode = infer_mode(task_info)

    logger.info("Task: %s - %s (%s)", task_code, task_info["L3_task"], task_info["description"])
    logger.info("Mode: %s", mode)

    client = OpenAIClient(
        api_key=os.environ.get("API_KEY"),
        model=os.environ.get("MODEL_NAME"),
        base_url=os.environ.get("API_BASE_URL"),
    )

    try:
        # Generate all components
        logger.info("Generating %d hints...", hint_count)
        hints = await generate_hints(client, task_info, hint_count)
        logger.info("Got %d unique hints", len(hints))

        logger.info("Generating %d instructions...", instruction_count)
        instructions = await generate_instructions(client, task_info, instruction_count)
        logger.info("Got %d unique instructions", len(instructions))

        logger.info("Generating system prompts...")
        system_prompts = await generate_system_prompts(client, task_info)

        logger.info("Generating difficulty config...")
        difficulty = await generate_difficulty_config(client, task_info)

        # Assemble pool
        pool = {
            "task_code": task_code,
            "L1_domain": task_info["L1_domain"],
            "L2_task_type": task_info["L2_task_type"],
            "L3_task": task_info["L3_task"],
            "description": task_info["description"],
            "mode": mode,
            "language": {"input": "tibetan", "output": "tibetan"},
            "system_prompts": system_prompts,
            "difficulty": difficulty,
            "instructions": instructions,
            "hints": hints,
            "quality": {
                "min_length": 50 if mode == "direct" else 100,
                "tibetan_ratio": 0.6,
                "chinese_tolerance": 0,
            },
        }

        # Save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{task_code}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pool, f, ensure_ascii=False, indent=2)

        logger.info("Saved prompt pool to %s", output_path)
        logger.info("Summary: %d hints, %d instructions, %d styles, mode=%s",
                     len(hints), len(instructions), len(system_prompts), mode)

    finally:
        await client.close()


@click.command()
@click.argument("task_code")
@click.option("--hints", "-h", default=500, help="Number of Chinese hints to generate")
@click.option("--instructions", "-i", default=200, help="Number of Tibetan instructions")
def main(task_code, hints, instructions):
    """Generate prompt pool for a L3 task. Example: python scripts/generate_prompt_pool.py 2.1.2"""
    asyncio.run(generate_pool(task_code, hints, instructions))


if __name__ == "__main__":
    main()
