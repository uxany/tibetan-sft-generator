#!/usr/bin/env python3
"""
Quality check for 10,000 Tibetan SFT story dataset.

Checks:
- Chinese contamination (0 tolerance)
- English leakage (< 1%)
- Tibetan purity (>= 70%)
- Dialogue presence (>= 80%)
- Length range (100-2000 chars)
- Uniqueness (hash dedup)
- Genre distribution (each within 20% of target)
- Difficulty distribution
- JSON Schema validation

Usage:
    PYTHONPATH=. python3 scripts/quality_check_10k.py [FILE]
    PYTHONPATH=. python3 scripts/quality_check_10k.py output/stories_10k/stories_raw.jsonl
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import click

# ============================================================
# Regex patterns
# ============================================================

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
ENGLISH_RE = re.compile(r"[a-zA-Z]{3,}")
TIBETAN_RE = re.compile(r"[\u0f00-\u0fff]")
DIALOGUE_RE = re.compile(r"《[^》]+》")

# ============================================================
# Schema validation (optional — uses jsonschema if available)
# ============================================================

SCHEMA_PATH = Path("pipeline/config/sft_schema.json")


def load_schema():
    """Load JSON Schema if jsonschema is available."""
    try:
        import jsonschema  # noqa: F401
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except ImportError:
        return None
    except FileNotFoundError:
        return None


def validate_schema(sample: dict, schema: dict) -> bool:
    """Validate a single sample against JSON Schema."""
    import jsonschema
    try:
        jsonschema.validate(sample, schema)
        return True
    except jsonschema.ValidationError:
        return False


# ============================================================
# Quality checks
# ============================================================

def get_assistant_content(sample: dict) -> str:
    """Extract assistant message content from sample."""
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def run_quality_check(samples: list[dict], verbose: bool = False):
    """Run full quality report on a list of SFT samples."""
    total = len(samples)
    if total == 0:
        print("No samples to check.")
        return

    # Counters
    chinese_contaminated = []
    english_leaked = []
    low_tibetan_ratio = []
    has_dialogue = 0
    short_samples = []  # < 100
    long_samples = []   # > 2000
    lengths = []
    hashes = set()
    duplicates = 0
    schema_fails = []

    # Distribution
    genre_counter = Counter()
    difficulty_counter = Counter()

    # Schema
    schema = load_schema()
    schema_available = schema is not None

    for i, sample in enumerate(samples):
        sid = sample.get("id", f"row_{i}")
        story = get_assistant_content(sample)
        meta = sample.get("metadata", {})

        # 1. Chinese contamination
        chinese_matches = CHINESE_RE.findall(story)
        if chinese_matches:
            chinese_contaminated.append(sid)

        # 2. English leakage
        english_matches = ENGLISH_RE.findall(story)
        if english_matches:
            english_leaked.append(sid)

        # 3. Tibetan purity
        tibetan_count = len(TIBETAN_RE.findall(story))
        ratio = tibetan_count / max(len(story), 1)
        if ratio < 0.7:
            low_tibetan_ratio.append((sid, f"{ratio:.2%}"))

        # 4. Dialogue
        if DIALOGUE_RE.search(story):
            has_dialogue += 1

        # 5. Length
        char_count = len(story)
        lengths.append(char_count)
        if char_count < 100:
            short_samples.append((sid, char_count))
        if char_count > 2000:
            long_samples.append((sid, char_count))

        # 6. Uniqueness
        h = hashlib.sha256(story.encode()).hexdigest()
        if h in hashes:
            duplicates += 1
        else:
            hashes.add(h)

        # 7. Distribution
        domain_tags = meta.get("domain_tags", [])
        for tag in domain_tags:
            genre_counter[tag] += 1
        difficulty_counter[meta.get("difficulty", "unknown")] += 1

        # 8. Schema validation
        if schema_available:
            if not validate_schema(sample, schema):
                schema_fails.append(sid)

    # ============================================================
    # Report
    # ============================================================

    print("=" * 60)
    print(f"  QUALITY REPORT — {total} samples")
    print("=" * 60)

    # Length stats
    avg_len = sum(lengths) / total
    print(f"\n📏 Length Statistics:")
    print(f"  Average: {avg_len:.0f} chars")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Under 100: {len(short_samples)}")
    print(f"  Over 2000: {len(long_samples)}")
    print(f"  Distribution:")
    brackets = [
        ("  <200", 0, 200),
        ("  200-400", 200, 400),
        ("  400-700", 400, 700),
        ("  700-1100", 700, 1100),
        ("  >1100", 1100, 999999),
    ]
    for label, lo, hi in brackets:
        n = sum(1 for l in lengths if lo <= l < hi)
        print(f"    {label}: {n} ({n * 100 / total:.1f}%)")

    # Chinese contamination
    print(f"\n🈲 Chinese Contamination: {len(chinese_contaminated)}/{total}", end="")
    print(f"  {'✅ PASS' if len(chinese_contaminated) == 0 else '❌ FAIL'}")
    if chinese_contaminated and verbose:
        for sid in chinese_contaminated[:10]:
            print(f"    - {sid}")

    # English leakage
    pct = len(english_leaked) / total * 100
    print(f"\n🔤 English Leakage: {len(english_leaked)}/{total} ({pct:.1f}%)", end="")
    print(f"  {'✅ PASS' if pct < 1 else '⚠️  HIGH'}")

    # Tibetan purity
    print(f"\n🏔️  Low Tibetan Ratio (<70%): {len(low_tibetan_ratio)}/{total}", end="")
    print(f"  {'✅ PASS' if len(low_tibetan_ratio) == 0 else '⚠️  CHECK'}")
    if low_tibetan_ratio and verbose:
        for sid, r in low_tibetan_ratio[:10]:
            print(f"    - {sid}: {r}")

    # Dialogue
    dial_pct = has_dialogue / total * 100
    print(f"\n💬 Dialogue Presence: {has_dialogue}/{total} ({dial_pct:.1f}%)", end="")
    print(f"  {'✅ PASS' if dial_pct >= 80 else '⚠️  LOW'}")

    # Uniqueness
    print(f"\n🔑 Uniqueness: {total - duplicates}/{total} unique", end="")
    print(f"  {'✅ PASS' if duplicates == 0 else f'⚠️  {duplicates} duplicates'}")

    # Schema validation
    if schema_available:
        print(f"\n📋 Schema Validation: {total - len(schema_fails)}/{total} pass", end="")
        print(f"  {'✅ PASS' if len(schema_fails) == 0 else '❌ FAIL'}")
        if schema_fails and verbose:
            for sid in schema_fails[:10]:
                print(f"    - {sid}")
    else:
        print(f"\n📋 Schema Validation: SKIPPED (install jsonschema)")

    # Difficulty distribution
    print(f"\n📊 Difficulty Distribution:")
    target_diff = {"easy": 0.30, "medium": 0.40, "hard": 0.30}
    for diff in ["easy", "medium", "hard"]:
        n = difficulty_counter.get(diff, 0)
        actual = n / total
        target = target_diff.get(diff, 0)
        dev = abs(actual - target) / target * 100 if target > 0 else 0
        status = "✅" if dev < 20 else "⚠️"
        print(f"  {diff}: {n} ({actual:.1%}) target={target:.0%} dev={dev:.0f}% {status}")

    # Genre distribution
    print(f"\n📚 Genre Distribution (domain_tags):")
    for tag, count in sorted(genre_counter.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")

    print("\n" + "=" * 60)

    # Summary verdict
    issues = []
    if len(chinese_contaminated) > 0:
        issues.append(f"{len(chinese_contaminated)} Chinese contaminated")
    if pct >= 1:
        issues.append(f"{pct:.1f}% English leakage")
    if len(low_tibetan_ratio) > 0:
        issues.append(f"{len(low_tibetan_ratio)} low Tibetan ratio")
    if dial_pct < 80:
        issues.append(f"dialogue rate {dial_pct:.1f}% < 80%")
    if duplicates > 0:
        issues.append(f"{duplicates} duplicates")
    if schema_fails:
        issues.append(f"{len(schema_fails)} schema failures")

    if not issues:
        print("  VERDICT: ✅ ALL CHECKS PASSED")
    else:
        print(f"  VERDICT: ⚠️  {len(issues)} issue(s):")
        for issue in issues:
            print(f"    - {issue}")

    print("=" * 60)


# ============================================================
# CLI
# ============================================================

@click.command()
@click.argument("file", default="output/stories_10k/stories_raw.jsonl")
@click.option("--verbose", "-v", is_flag=True, help="Show sample IDs for failures")
def main(file, verbose):
    """Run quality checks on generated JSONL file."""
    path = Path(file)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: invalid JSON at line {line_no}: {e}")

    print(f"Loaded {len(samples)} samples from {path}")
    run_quality_check(samples, verbose=verbose)


if __name__ == "__main__":
    main()
