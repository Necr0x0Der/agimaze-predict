"""Zero- and few-shot MAP + ACT -> POS probes through OpenAI-compatible APIs.

This deliberately has no dependency on PyTorch, Transformers, or a trained
spatial-memory checkpoint.  It measures an API model's out-of-the-box ability
to interpret the ASCII maze and return the dataset's structured position target.
"""

from __future__ import annotations

import json
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from agimaze_predict.data.prepared import PreparedExample, read_prepared_jsonl


SYSTEM_PROMPT = """You solve a deterministic grid-maze state-tracking task.

The user supplies an ASCII maze inside <MAP>...</MAP>, followed by one or more
completed actions in chronological order inside <ACT>...</ACT>. The `S` symbol
marks the agent's initial cell. The maze has zero-based coordinates: row 0 is
the top cell row and column 0 is the left cell column. `+`, `-`, and `|` draw
cell borders/walls. Apply each action (up, down, left, right) in order. A move
through a wall or outside the maze leaves the agent in its current cell. Other
symbols (for example K and T) are map objects, not the agent's position.

Return exactly one answer and no explanation: <POS>(row, col)</POS>."""
_POS_RE = re.compile(r"<POS>\([0-9]+, [0-9]+\)</POS>")


@dataclass(frozen=True)
class IclPrediction:
    expected: str
    completion: str
    extracted: str | None
    correct: bool


def build_messages(
    example: PreparedExample, *, demonstrations: Sequence[PreparedExample] = ()
) -> list[dict[str, str]]:
    """Build a portable Chat Completions request without provider-specific tools."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for demonstration in demonstrations:
        messages.extend((
            {"role": "user", "content": demonstration.input},
            {"role": "assistant", "content": demonstration.target},
        ))
    messages.append({"role": "user", "content": example.input})
    return messages


def extract_pos(completion: str) -> str | None:
    """Extract one canonical POS answer, rejecting prose with multiple answers."""

    matches = _POS_RE.findall(completion)
    return matches[0] if len(matches) == 1 else None


def select_demonstrations(
    examples: Sequence[PreparedExample], *, count: int, seed: int
) -> list[PreparedExample]:
    if count < 0:
        raise ValueError("few-shot count must be non-negative")
    if count > len(examples):
        raise ValueError(f"few-shot count {count} exceeds demonstration dataset size {len(examples)}")
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    return [examples[index] for index in indices[:count]]


def resolve_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("set OPENAI_API_KEY or OPENROUTER_API_KEY")
    return key


def default_base_url() -> str:
    return (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )


def chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    retries: int,
) -> str:
    """Call the portable OpenAI Chat Completions subset, retrying transient faults."""

    if max_tokens <= 0 or timeout_seconds <= 0 or retries < 1:
        raise ValueError("max_tokens, timeout_seconds, and retries must be positive")
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"].get("content")
            if not isinstance(content, str):
                raise ValueError("API response has no textual choices[0].message.content")
            return content
        except urllib.error.HTTPError as exc:
            error = exc
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"API HTTP {exc.code}: {body}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 0.8 * (2 ** attempt)
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            error = exc
            if attempt == retries - 1:
                raise RuntimeError(f"API request failed after {retries} attempts: {exc}") from exc
            delay = 0.8 * (2 ** attempt)
        time.sleep(delay)
    raise RuntimeError(f"API request failed after {retries} attempts: {error}")


def evaluate_examples(
    examples: Sequence[PreparedExample], *, demonstrations: Sequence[PreparedExample], complete: Callable[[list[dict[str, str]]], str]
) -> list[IclPrediction]:
    """Run an injected completion function, making the metric testable offline."""

    predictions: list[IclPrediction] = []
    for example in examples:
        completion = complete(build_messages(example, demonstrations=demonstrations))
        extracted = extract_pos(completion)
        predictions.append(IclPrediction(
            expected=example.target,
            completion=completion,
            extracted=extracted,
            correct=extracted == example.target,
        ))
    return predictions


def run(
    *,
    dataset: Path,
    model: str,
    base_url: str,
    few_shot_dataset: Path | None,
    few_shot_count: int,
    seed: int,
    max_examples: int | None,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    retries: int,
    output: Path | None,
) -> dict[str, object]:
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when supplied")
    examples = read_prepared_jsonl(dataset)
    if max_examples is not None:
        examples = examples[:max_examples]
    if not examples:
        raise ValueError("no evaluation examples selected")
    demonstration_pool = [] if few_shot_dataset is None else read_prepared_jsonl(few_shot_dataset)
    demonstrations = select_demonstrations(demonstration_pool, count=few_shot_count, seed=seed)
    api_key = resolve_api_key()

    def complete(messages: list[dict[str, str]]) -> str:
        return chat_completion(
            base_url=base_url, api_key=api_key, model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            timeout_seconds=timeout_seconds, retries=retries,
        )

    predictions = evaluate_examples(examples, demonstrations=demonstrations, complete=complete)
    correct = sum(prediction.correct for prediction in predictions)
    result: dict[str, object] = {
        "model": model,
        "base_url": base_url,
        "dataset": str(dataset.resolve()),
        "examples": len(predictions),
        "few_shot_examples": len(demonstrations),
        "temperature": temperature,
        "exact_target_accuracy": correct / len(predictions),
        "format_valid_accuracy": sum(prediction.extracted is not None for prediction in predictions) / len(predictions),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            for example, prediction in zip(examples, predictions):
                stream.write(json.dumps({
                    "input": example.input,
                    "expected": prediction.expected,
                    "completion": prediction.completion,
                    "extracted": prediction.extracted,
                    "correct": prediction.correct,
                }, ensure_ascii=False) + "\n")
        result["output"] = str(output)
    return result
