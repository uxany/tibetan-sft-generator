import copy
import json
from pathlib import Path


DEFAULT_RULE_PATH = Path(__file__).resolve().parents[1] / "config" / "task_quality_rules.json"


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def load_quality_rules(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_RULE_PATH

    if not path.exists():
        raise FileNotFoundError(f"Quality rules file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_task_rule(task_code: str, rules: dict | None = None) -> dict:
    """
    Return merged rule for task_code.

    Matching strategy:
    - Start from rules["default"]
    - Find all matching prefixes in rules["prefix_rules"]
    - Apply from shortest to longest, so 6.2.1 can override 6.2 later if needed
    """
    rules = rules or load_quality_rules()

    default = rules.get("default", {})
    prefix_rules = rules.get("prefix_rules", {})

    task_code = str(task_code or "")

    matched_prefixes = [
        prefix
        for prefix in prefix_rules
        if task_code == prefix or task_code.startswith(prefix + ".")
    ]

    matched_prefixes.sort(key=lambda x: len(x.split(".")))

    merged = copy.deepcopy(default)

    for prefix in matched_prefixes:
        merged = _deep_merge(merged, prefix_rules[prefix])

    merged["matched_prefixes"] = matched_prefixes
    return merged