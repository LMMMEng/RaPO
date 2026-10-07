import re
from typing import Any

REWARD_NAME = "cls"
REWARD_TYPE = "sequential"


def _normalize(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ").replace(".", " ")


def _extract_tag_content(response: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", response, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def format_reward(response: str) -> float:
    # Response must be exactly <think>...</think><answer>...</answer>, and the
    # <think> block must not merely echo the prompt's placeholder text.
    lazy_think_patterns = [
        "step-by-step reasoning here",
        "reasoning here",
        "thinking process here",
        "detailed reasoning here",
    ]
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    if not re.fullmatch(pattern, response):
        return 0.0
    think_text = _extract_tag_content(response, "think")
    if any(p in think_text.lower() for p in lazy_think_patterns):
        return 0.0
    return 1.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    content_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    pred = content_match.group(1).strip() if content_match else response.strip()
    return 1.0 if _normalize(pred) == _normalize(ground_truth.strip()) else 0.0


def compute_score(reward_input: dict[str, Any], format_weight: float = 1) -> dict[str, float]:
    format_score = format_reward(reward_input["response"])
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    overall = accuracy_score + format_weight * format_score
    return {
        "overall": overall,
        "format": format_score,
        "accuracy": accuracy_score,
    }
