import hashlib
import json
import re
from typing import Any


TIBETAN_RE = re.compile(r"[\u0f00-\u0fff]")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
ENGLISH_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")
LONG_ENGLISH_WORD_RE = re.compile(r"\b[a-zA-Z]{5,}\b")
WYLIE_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z'\-\.]{1,}\b")
WYLIE_CHAR_RE = re.compile(r"[A-Za-z'\-\. ]")
NEUTRAL_SYMBOL_RE = re.compile(r"[0-9０-９༠-༩\s\.,;:!?()\[\]{}<>《》“”‘’\"'།༎༏༐༑་\-_/]+")


def tibetan_ratio_of(text: str) -> float:
    text = text or ""
    return len(TIBETAN_RE.findall(text)) / max(len(text), 1)


def wylie_ratio_of(text: str) -> float:
    text = text or ""
    return len(WYLIE_CHAR_RE.findall(text)) / max(len(text), 1)


def has_tibetan(text: str) -> bool:
    return bool(TIBETAN_RE.search(text or ""))


def has_wylie_like(text: str) -> bool:
    text = text or ""
    tokens = WYLIE_TOKEN_RE.findall(text)
    return any(len(t) >= 2 for t in tokens)


def clean_text_strict(text: str) -> str:
    """
    Strict cleaner for ordinary Tibetan long-form tasks.
    """
    if text is None:
        return ""

    lines = str(text).split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        tibetan_chars = len(TIBETAN_RE.findall(line))

        if tibetan_chars < len(line) * 0.3 and len(line) > 10:
            continue

        if line.startswith("**") or line.startswith("#") or line.startswith("---"):
            continue

        if any(
            line.lower().startswith(w)
            for w in [
                "note:",
                "wait",
                "let me",
                "okay",
                "text:",
                "ready",
                "(note",
                "here",
                "i ",
                "the ",
                "this ",
                "story",
                "answer:",
                "output:",
                "response:",
            ]
        ):
            continue

        cleaned.append(line)

    result = " ".join(cleaned)
    result = ENGLISH_WORD_RE.sub("", result)
    result = re.sub(r" +", " ", result).strip()

    return result


def clean_text_light(text: str) -> str:
    """
    Light cleaner for short, structured, legal/admin, culture/art, and transliteration tasks.
    """
    if text is None:
        return ""

    text = str(text).strip()

    text = re.sub(r"^```(?:json|JSON)?\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    text = re.sub(r"^\s*(answer|output|response|assistant)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(instruction|user)\s*[:：]\s*", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text_for_rule(text: str, rule: dict) -> str:
    cleaner = rule.get("cleaner", "strict")

    if cleaner == "light":
        return clean_text_light(text)

    return clean_text_strict(text)


def normalize_for_dedup(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[།༎༏༐༑་\u0f0b\W_]+", "", text)
    return text


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_dedup(text).encode("utf-8")).hexdigest()


def quality_reason(
    text: str,
    rule: dict,
    field_name: str = "output",
) -> str | None:
    """
    Return None if text passes quality checks. Otherwise return reason string.
    """
    text = text or ""

    min_length = int(rule.get("min_length", 50))
    tibetan_ratio = float(rule.get("tibetan_ratio", 0.6))

    if field_name == "instruction":
        min_length = int(rule.get("instruction_min_length", min_length))
        tibetan_ratio = float(rule.get("instruction_tibetan_ratio", tibetan_ratio))

    english_policy = rule.get("english_policy", "forbid")
    script_policy = rule.get("script_policy", "tibetan")
    score_mode = rule.get("quality_score_mode", "tibetan")

    if CHINESE_RE.search(text):
        return "chinese_contamination"

    if len(text) < min_length:
        return f"too_short:{len(text)}<{min_length}"

    tib_ratio = tibetan_ratio_of(text)
    wy_ratio = wylie_ratio_of(text)

    if script_policy == "tibetan_or_wylie":
        if not has_tibetan(text) and not has_wylie_like(text):
            return "no_tibetan_or_wylie_content"

        if tib_ratio < tibetan_ratio and wy_ratio < 0.45:
            return f"low_script_ratio:tibetan={tib_ratio:.3f},wylie={wy_ratio:.3f}"

        return None

    if english_policy == "forbid":
        if ENGLISH_WORD_RE.search(text):
            return "english_contamination"

    elif english_policy == "allow_short_acronyms":
        tokens = LONG_ENGLISH_WORD_RE.findall(text)
        if len(tokens) >= 3:
            return f"english_contamination:{','.join(tokens[:5])}"

    elif english_policy in {"allow_wylie", "allow"}:
        pass

    else:
        return f"unknown_english_policy:{english_policy}"

    if score_mode == "legal_admin":
        stripped = NEUTRAL_SYMBOL_RE.sub("", text)
        adjusted_ratio = len(TIBETAN_RE.findall(stripped)) / max(len(stripped), 1)

        if adjusted_ratio < tibetan_ratio:
            return f"low_tibetan_ratio_adjusted:{adjusted_ratio:.3f}<{tibetan_ratio}"

        return None

    if tib_ratio < tibetan_ratio:
        return f"low_tibetan_ratio:{tib_ratio:.3f}<{tibetan_ratio}"

    return None


def estimate_quality_score(text: str, rule: dict, field_name: str = "output") -> float:
    text = text or ""

    min_length = int(rule.get("min_length", 50))
    tibetan_ratio = float(rule.get("tibetan_ratio", 0.6))

    if field_name == "instruction":
        min_length = int(rule.get("instruction_min_length", min_length))
        tibetan_ratio = float(rule.get("instruction_tibetan_ratio", tibetan_ratio))

    score_mode = rule.get("quality_score_mode", "tibetan")

    length_score = min(len(text) / max(min_length * 2, 1), 1.0)

    if score_mode == "tibetan_or_wylie":
        script_score = max(
            min(tibetan_ratio_of(text) / max(tibetan_ratio, 0.01), 1.0),
            min(wylie_ratio_of(text) / 0.45, 1.0),
        )
        return round(length_score * 0.35 + script_score * 0.65, 4)

    if score_mode == "legal_admin":
        stripped = NEUTRAL_SYMBOL_RE.sub("", text)
        ratio = len(TIBETAN_RE.findall(stripped)) / max(len(stripped), 1)
        purity_score = min(ratio / max(tibetan_ratio, 0.01), 1.0)
        return round(length_score * 0.35 + purity_score * 0.65, 4)

    ratio = tibetan_ratio_of(text)
    purity_score = min(ratio / max(tibetan_ratio, 0.01), 1.0)

    return round(length_score * 0.35 + purity_score * 0.65, 4)


def parse_json_response(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()

    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "instruction" in data and "output" in data:
            return data
    except json.JSONDecodeError:
        pass

    fences = re.findall(r"```(?:json|JSON)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    for fence in fences:
        try:
            data = json.loads(fence.strip())
            if isinstance(data, dict) and "instruction" in data and "output" in data:
                return data
        except json.JSONDecodeError:
            pass

    match = re.search(r'\{[^{}]*"instruction"[^{}]*"output"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "instruction" in data and "output" in data:
                return data
        except json.JSONDecodeError:
            pass

    return None


def infer_script(text: str, rule: dict) -> str:
    script_policy = rule.get("script_policy", "tibetan")

    if script_policy == "tibetan_or_wylie":
        if has_tibetan(text) and has_wylie_like(text):
            return "mixed"
        if has_wylie_like(text) and not has_tibetan(text):
            return "wylie"
        return "tibetan"

    return "tibetan"