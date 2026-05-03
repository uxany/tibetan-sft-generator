#!/usr/bin/env python3
"""
Stage 2: Generate SFT training data for any L3 task using a prompt pool.

This version keeps task-specific quality rules outside this script:

- pipeline/config/task_quality_rules.json
- pipeline/quality/task_rules.py
- pipeline/quality/text_quality.py
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import date
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv(Path("pipeline/.env"))

from pipeline.llm.providers.openai_client import OpenAIClient
from pipeline.quality.task_rules import get_task_rule, load_quality_rules
from pipeline.quality.text_quality import (
    clean_text_for_rule,
    content_hash,
    estimate_quality_score,
    infer_script,
    parse_json_response,
    quality_reason,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("generate_data")

POOL_DIR = Path("output/prompt_pools")
DATA_DIR = Path("output/data")


def build_prompt_notes(rule: dict) -> str:
    notes = rule.get("prompt_notes", [])

    if not notes:
        return ""

    return "\n".join(f"{idx + 1}. {note}" for idx, note in enumerate(notes))


def build_direct_prompt(
    instruction: str,
    hint: str,
    difficulty_guidance: str,
    task_name: str,
    rule: dict,
) -> str:
    notes = build_prompt_notes(rule)

    return f"""请完成以下藏文SFT任务。

任务类型：{task_name}
任务场景：{hint}
写作指令：{instruction}
难度要求：{difficulty_guidance}

任务规则：
{notes}

通用输出要求：
1. 直接输出最终答案
2. 不要输出额外说明
3. 不要输出 Markdown
4. 不要输出代码块
"""


def build_paired_prompt(
    hint: str,
    task_name: str,
    difficulty_guidance: str,
    rule: dict,
) -> str:
    notes = build_prompt_notes(rule)

    return f"""请生成一条SFT训练数据。

任务类型：{task_name}
场景描述：{hint}
难度要求：{difficulty_guidance}

任务规则：
{notes}

生成要求：
1. instruction 字段：用户输入
2. output 字段：助手回复
3. instruction 和 output 必须语义相关，output 必须直接回应 instruction
4. 不要输出任何额外说明
5. 不要输出 Markdown
6. 不要输出代码块

你的输出必须是且仅是一个合法 JSON 对象：
{{
  "instruction": "<用户输入>",
  "output": "<助手回复>"
}}"""


def load_pool(task_code: str) -> dict:
    path = POOL_DIR / f"{task_code}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt pool not found: {path}\n"
            f"Run: PYTHONPATH=. python3 scripts/generate_prompt_pool.py {task_code}"
        )

    with path.open("r", encoding="utf-8") as f:
        pool = json.load(f)

    missing = [
        key
        for key in ("system_prompts", "hints", "difficulty", "mode")
        if not pool.get(key)
    ]

    if pool.get("mode") == "direct" and not pool.get("instructions"):
        missing.append("instructions")

    if missing:
        raise ValueError(
            f"Prompt pool {path} is missing required content: {', '.join(missing)}"
        )

    return pool


def build_task_list(
    pool: dict,
    sample_numbers: list[int],
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)

    instructions = pool.get("instructions", [])
    hints = pool.get("hints", [])
    styles = pool.get("system_prompts", [])
    difficulty_config = pool["difficulty"]
    mode = pool["mode"]

    difficulties = []
    total = len(sample_numbers)

    for diff_name, diff_cfg in difficulty_config.items():
        n = int(total * diff_cfg.get("ratio", 0.33))
        difficulties.extend([(diff_name, diff_cfg.get("guidance", ""))] * n)

    while len(difficulties) < total:
        diff_name = rng.choice(list(difficulty_config.keys()))
        difficulties.append(
            (diff_name, difficulty_config[diff_name].get("guidance", ""))
        )

    rng.shuffle(difficulties)

    tasks = []

    for i, sample_no in enumerate(sample_numbers):
        instruction = rng.choice(instructions) if instructions else ""
        hint = rng.choice(hints) if hints else ""
        style = rng.choice(styles) if styles else {"prompt": ""}
        diff_name, diff_guidance = difficulties[i]

        tasks.append(
            {
                "idx": sample_no,
                "sample_id": f"bo_sft_{sample_no:06d}",
                "instruction": instruction,
                "hint": hint,
                "style": style,
                "difficulty": diff_name,
                "difficulty_guidance": diff_guidance,
                "mode": mode,
            }
        )

    rng.shuffle(tasks)
    return tasks


def build_sample_direct(task: dict, pool: dict, output: str, rule: dict) -> dict:
    messages = [
        {"role": "system", "content": task["style"]["prompt"]},
        {"role": "user", "content": task["instruction"]},
        {"role": "assistant", "content": output},
    ]

    return _build_sample_common(task, pool, messages, has_system=True, rule=rule)


def build_sample_paired(
    task: dict,
    pool: dict,
    instruction: str,
    output: str,
    rule: dict,
) -> dict:
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]

    return _build_sample_common(task, pool, messages, has_system=False, rule=rule)


def _build_sample_common(
    task: dict,
    pool: dict,
    messages: list,
    has_system: bool,
    rule: dict,
) -> dict:
    assistant_output = ""

    for msg in messages:
        if msg.get("role") == "assistant":
            assistant_output = msg.get("content", "")
            break

    return {
        "id": task["sample_id"],
        "messages": messages,
        "metadata": {
            "L1_domain": pool["L1_domain"],
            "L2_task_type": pool["L2_task_type"],
            "L3_task": pool["L3_task"],
            "task_code": pool["task_code"],
            "language": pool["language"],
            "difficulty": task["difficulty"],
            "turns": 1,
            "has_system": has_system,
            "source": "synthetic",
            "method": "M4",
            "generator_model": os.environ.get("MODEL_NAME") or "unknown",
            "quality_score": None,
            "quality_tier": None,
            "dialect": "standard",
            "script": infer_script(assistant_output, rule),
            "domain_tags": rule.get("domain_tags", []),
            "requires_cultural_knowledge": bool(
                rule.get("requires_cultural_knowledge", False)
            ),
            "quality_rule_prefixes": rule.get("matched_prefixes", []),
            "created_at": date.today().isoformat(),
            "version": "2.0",
        },
    }


class DataGenerator:
    def __init__(self, concurrency: int = 3, max_rounds: int = 5):
        self.concurrency = concurrency
        self.max_rounds = max_rounds

        self.generated_count = 0
        self.failed_count = 0
        self.quality_rejected = 0
        self.parse_rejected = 0
        self.duplicate_rejected = 0

        self._lock = asyncio.Lock()
        self._checkpoint: set[str] = set()
        self._content_hashes: set[str] = set()

        self._start_time = 0.0
        self._file_handle = None
        self._reject_file_handle = None

        self._all_rules = load_quality_rules()

    @staticmethod
    def _assistant_content(sample: dict) -> str:
        for msg in sample.get("messages", []):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def _completed_target_count(self, count: int) -> int:
        return sum(
            1
            for i in range(1, count + 1)
            if f"bo_sft_{i:06d}" in self._checkpoint
        )

    def _load_checkpoint(self, output_file: Path):
        if output_file.exists():
            with output_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                        self._checkpoint.add(obj["id"])

                        assistant = self._assistant_content(obj)

                        if assistant:
                            self._content_hashes.add(content_hash(assistant))

                    except Exception:
                        pass

        logger.info("Checkpoint: %d samples already generated", len(self._checkpoint))

    def _save_sample(self, sample: dict):
        self._file_handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self._file_handle.flush()

    def _save_reject(self, item: dict):
        self._reject_file_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._reject_file_handle.flush()

    async def _record_reject(
        self,
        task: dict,
        pool: dict,
        reason,
        stage: str,
        raw_output: str | None = None,
        parsed: dict | None = None,
        cleaned: dict | None = None,
        attempt: int | None = None,
    ):
        async with self._lock:
            if stage == "parse":
                self.parse_rejected += 1
            elif stage == "duplicate":
                self.duplicate_rejected += 1
            else:
                self.quality_rejected += 1

            self._save_reject(
                {
                    "sample_id": task.get("sample_id"),
                    "task_code": pool.get("task_code"),
                    "L3_task": pool.get("L3_task"),
                    "stage": stage,
                    "reason": reason,
                    "attempt": attempt,
                    "difficulty": task.get("difficulty"),
                    "hint": task.get("hint"),
                    "raw_output": raw_output,
                    "parsed": parsed,
                    "cleaned": cleaned,
                    "created_at": date.today().isoformat(),
                }
            )

    async def _try_save_sample(self, sample: dict, task: dict, pool: dict) -> bool:
        assistant = self._assistant_content(sample)
        sample_hash = content_hash(assistant)

        async with self._lock:
            if sample["id"] in self._checkpoint:
                return True

            if sample_hash in self._content_hashes:
                self.duplicate_rejected += 1
                self._save_reject(
                    {
                        "sample_id": task.get("sample_id"),
                        "task_code": pool.get("task_code"),
                        "L3_task": pool.get("L3_task"),
                        "stage": "duplicate",
                        "reason": "duplicate_assistant_content",
                        "assistant": assistant,
                        "created_at": date.today().isoformat(),
                    }
                )
                return False

            self._save_sample(sample)
            self._checkpoint.add(sample["id"])
            self._content_hashes.add(sample_hash)
            self.generated_count += 1
            self._log_progress()

            return True

    async def _generate_one_direct(
        self,
        client: OpenAIClient,
        task: dict,
        pool: dict,
        rule: dict,
        semaphore: asyncio.Semaphore,
    ):
        if task["sample_id"] in self._checkpoint:
            return True

        prompt = build_direct_prompt(
            instruction=task["instruction"],
            hint=task["hint"],
            difficulty_guidance=task["difficulty_guidance"],
            task_name=pool.get("L3_task", ""),
            rule=rule,
        )

        system_prompt = task["style"]["prompt"]

        async with semaphore:
            for attempt in range(3):
                try:
                    resp = await client.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=float(rule.get("temperature", 0.9)),
                        max_tokens=int(rule.get("max_tokens", 4096)),
                    )

                    raw = resp.content.strip()
                    output = clean_text_for_rule(raw, rule)

                    reason = quality_reason(output, rule, field_name="output")

                    if reason is None:
                        sample = build_sample_direct(task, pool, output, rule)

                        score = estimate_quality_score(output, rule, field_name="output")
                        sample["metadata"]["quality_score"] = score
                        sample["metadata"]["quality_tier"] = (
                            "gold" if score >= 0.9 else "silver"
                        )

                        if await self._try_save_sample(sample, task, pool):
                            return True

                    else:
                        await self._record_reject(
                            task=task,
                            pool=pool,
                            reason=reason,
                            stage="quality",
                            raw_output=raw,
                            cleaned={"output": output},
                            attempt=attempt + 1,
                        )

                except Exception as e:
                    wait = 5 * (attempt + 1)
                    logger.warning(
                        "%s attempt %d failed: %s. Retry in %ds...",
                        task["sample_id"],
                        attempt + 1,
                        str(e)[:120],
                        wait,
                    )
                    await asyncio.sleep(wait)

        async with self._lock:
            self.failed_count += 1

        return False

    async def _generate_one_paired(
        self,
        client: OpenAIClient,
        task: dict,
        pool: dict,
        rule: dict,
        semaphore: asyncio.Semaphore,
    ):
        if task["sample_id"] in self._checkpoint:
            return True

        prompt = build_paired_prompt(
            hint=task["hint"],
            task_name=pool["L3_task"],
            difficulty_guidance=task["difficulty_guidance"],
            rule=rule,
        )

        system_prompt = task["style"]["prompt"]

        async with semaphore:
            for attempt in range(3):
                try:
                    resp = await client.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=float(rule.get("temperature", 0.9)),
                        max_tokens=int(rule.get("max_tokens", 4096)),
                    )

                    raw = resp.content
                    parsed = parse_json_response(raw)

                    if not parsed:
                        await self._record_reject(
                            task=task,
                            pool=pool,
                            reason="parse_json_failed",
                            stage="parse",
                            raw_output=raw,
                            attempt=attempt + 1,
                        )
                        continue

                    instruction = clean_text_for_rule(parsed.get("instruction", ""), rule)
                    output = clean_text_for_rule(parsed.get("output", ""), rule)

                    instr_reason = quality_reason(
                        instruction,
                        rule,
                        field_name="instruction",
                    )

                    out_reason = quality_reason(
                        output,
                        rule,
                        field_name="output",
                    )

                    if instr_reason is None and out_reason is None:
                        sample = build_sample_paired(
                            task,
                            pool,
                            instruction,
                            output,
                            rule,
                        )

                        score = estimate_quality_score(output, rule, field_name="output")
                        sample["metadata"]["quality_score"] = score
                        sample["metadata"]["quality_tier"] = (
                            "gold" if score >= 0.9 else "silver"
                        )

                        if await self._try_save_sample(sample, task, pool):
                            return True

                    else:
                        await self._record_reject(
                            task=task,
                            pool=pool,
                            reason={
                                "instruction": instr_reason,
                                "output": out_reason,
                            },
                            stage="quality",
                            raw_output=raw,
                            parsed=parsed,
                            cleaned={
                                "instruction": instruction,
                                "output": output,
                            },
                            attempt=attempt + 1,
                        )

                except Exception as e:
                    wait = 5 * (attempt + 1)
                    logger.warning(
                        "%s attempt %d failed: %s. Retry in %ds...",
                        task["sample_id"],
                        attempt + 1,
                        str(e)[:120],
                        wait,
                    )
                    await asyncio.sleep(wait)

        async with self._lock:
            self.failed_count += 1

        return False

    def _log_progress(self):
        if self.generated_count % 10 == 0:
            elapsed = time.time() - self._start_time
            rate = self.generated_count / max(elapsed, 1) * 3600

            logger.info(
                "Progress: %d generated | %d failed | %d quality rejected | "
                "%d parse rejected | %d duplicates | %.0f/hr",
                self.generated_count,
                self.failed_count,
                self.quality_rejected,
                self.parse_rejected,
                self.duplicate_rejected,
                rate,
            )

    async def run(self, task_code: str, count: int = 10000):
        pool = load_pool(task_code)
        mode = pool["mode"]
        rule = get_task_rule(task_code, self._all_rules)

        logger.info(
            "Task: %s (%s) | Mode: %s | Target: %d | Rules: %s",
            task_code,
            pool["L3_task"],
            mode,
            count,
            ",".join(rule.get("matched_prefixes", [])) or "default",
        )

        output_dir = DATA_DIR / task_code
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "data.jsonl"
        reject_file = output_dir / "rejects.jsonl"

        self._load_checkpoint(output_file)

        already_done = self._completed_target_count(count)

        if already_done >= count:
            logger.info("Already have %d samples. Done!", already_done)
            return

        client = OpenAIClient(
            api_key=os.environ.get("API_KEY"),
            model=os.environ.get("MODEL_NAME"),
            base_url=os.environ.get("API_BASE_URL"),
        )

        semaphore = asyncio.Semaphore(self.concurrency)

        self._start_time = time.time()
        self._file_handle = output_file.open("a", encoding="utf-8")
        self._reject_file_handle = reject_file.open("a", encoding="utf-8")

        gen_fn = (
            self._generate_one_direct
            if mode == "direct"
            else self._generate_one_paired
        )

        try:
            batch_size = 100

            for round_no in range(1, self.max_rounds + 1):
                missing_numbers = [
                    i
                    for i in range(1, count + 1)
                    if f"bo_sft_{i:06d}" not in self._checkpoint
                ]

                if not missing_numbers:
                    break

                tasks = build_task_list(pool, missing_numbers, seed=42 + round_no)

                logger.info(
                    "Round %d/%d: %d samples still missing",
                    round_no,
                    self.max_rounds,
                    len(tasks),
                )

                before_round = len(self._checkpoint)

                for batch_start in range(0, len(tasks), batch_size):
                    batch = tasks[batch_start : batch_start + batch_size]
                    coros = [
                        gen_fn(client, task, pool, rule, semaphore)
                        for task in batch
                    ]

                    await asyncio.gather(*coros)

                    total = self._completed_target_count(count)

                    logger.info(
                        "Batch %d-%d done. Total: %d/%d (%.1f%%)",
                        batch_start + 1,
                        min(batch_start + batch_size, len(tasks)),
                        total,
                        count,
                        total / count * 100,
                    )

                    if total >= count:
                        break

                if len(self._checkpoint) == before_round:
                    logger.warning(
                        "Round %d produced no new samples; stopping early to avoid a stuck run",
                        round_no,
                    )
                    break

        finally:
            if self._file_handle:
                self._file_handle.close()

            if self._reject_file_handle:
                self._reject_file_handle.close()

            await client.close()

        elapsed = time.time() - self._start_time

        logger.info(
            "Complete! Total: %d/%d, New: %d, Failed: %d, "
            "Quality rejected: %d, Parse rejected: %d, Duplicates: %d, Time: %.0f min",
            self._completed_target_count(count),
            count,
            self.generated_count,
            self.failed_count,
            self.quality_rejected,
            self.parse_rejected,
            self.duplicate_rejected,
            elapsed / 60,
        )

        logger.info("Reject log written to: %s", reject_file)


@click.command()
@click.argument("task_code")
@click.option("--count", "-n", default=10000, help="Target number of samples")
@click.option("--concurrency", "-c", default=3, help="Max concurrent API calls")
@click.option("--max-rounds", default=5, help="Retry rounds for missing/rejected samples")
def main(task_code, count, concurrency, max_rounds):
    gen = DataGenerator(concurrency=concurrency, max_rounds=max_rounds)
    asyncio.run(gen.run(task_code, count))


if __name__ == "__main__":
    main()