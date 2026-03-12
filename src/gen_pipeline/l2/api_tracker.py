"""
api_tracker.py — OpenAI API usage tracker and dummy client.

Usage in generate_content.py:
    from api_tracker import TrackedOpenAI, DummyOpenAI

    # Real API with token/cost logging:
    client = TrackedOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Free dummy for smoke-testing pipeline logic:
    client = DummyOpenAI()

Both clients expose the same interface as openai.OpenAI so no other code
needs to change.

Log file: api_usage_log.csv (next to this file by default).
Each run appends rows — open directly in Excel or paste into a sheet.
"""

from __future__ import annotations

import csv
import json
import random
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Pricing table  (USD per 1 000 000 tokens)
# Update these whenever OpenAI changes pricing.
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    "gpt-4.1-mini":   {"input": 0.40,  "output": 1.60},
    "gpt-4.1-nano":   {"input": 0.10,  "output": 0.40},
    "gpt-4.1":        {"input": 2.00,  "output": 8.00},
    "gpt-4o":         {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":    {"input": 0.15,  "output": 0.60},
    "gpt-3.5-turbo":  {"input": 0.50,  "output": 1.50},
    # Add more as needed; unknown models fall back to 0.0 cost with a warning.
}

# Default log path: same directory as this file.
DEFAULT_LOG_PATH = Path(__file__).parent / "api_usage_log.csv"

CSV_HEADER = [
    "timestamp_utc",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cost_usd",
    "completion_cost_usd",
    "total_cost_usd",
    "note",
]


# ---------------------------------------------------------------------------
# Shared logging logic
# ---------------------------------------------------------------------------

def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, float]:
    """Return (prompt_cost, completion_cost) in USD."""
    prices = PRICING.get(model)
    if prices is None:
        print(f"  [api_tracker] WARNING: no pricing data for model '{model}'. Logging $0.")
        return 0.0, 0.0
    prompt_cost = prompt_tokens * prices["input"] / 1_000_000
    completion_cost = completion_tokens * prices["output"] / 1_000_000
    return prompt_cost, completion_cost


def _append_log_row(
    log_path: Path,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    note: str = "",
) -> None:
    """Append one usage row to the CSV log."""
    prompt_cost, completion_cost = _compute_cost(model, prompt_tokens, completion_tokens)
    total_cost = prompt_cost + completion_cost
    total_tokens = prompt_tokens + completion_tokens
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerow([
            ts,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            f"{prompt_cost:.6f}",
            f"{completion_cost:.6f}",
            f"{total_cost:.6f}",
            note,
        ])

    print(
        f"  [api_tracker] {model} | "
        f"tokens: {prompt_tokens}+{completion_tokens}={total_tokens} | "
        f"cost: ${total_cost:.4f}"
    )


# ---------------------------------------------------------------------------
# Thin wrappers that mimic openai.OpenAI's chat.completions interface
# ---------------------------------------------------------------------------

@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _FakeMessage:
    content: str
    role: str = "assistant"


@dataclass
class _FakeChoice:
    message: _FakeMessage
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _FakeCompletion:
    choices: list[_FakeChoice]
    usage: _FakeUsage
    model: str = "dummy"
    id: str = "dummy-0"


class _TrackedCompletions:
    """Wraps openai.resources.chat.completions.Completions to intercept usage."""

    def __init__(self, real_completions: Any, log_path: Path, note: str) -> None:
        self._real = real_completions
        self._log_path = log_path
        self._note = note

    def create(self, **kwargs) -> Any:
        response = self._real.create(**kwargs)
        usage = response.usage
        if usage:
            _append_log_row(
                self._log_path,
                model=kwargs.get("model", "unknown"),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                note=self._note,
            )
        return response


class _TrackedChat:
    def __init__(self, real_chat: Any, log_path: Path, note: str) -> None:
        self.completions = _TrackedCompletions(real_chat.completions, log_path, note)


class TrackedOpenAI:
    """
    Drop-in replacement for openai.OpenAI that logs every API call's token
    usage and cost to a CSV file.

    Parameters
    ----------
    api_key:
        OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.
    log_path:
        Where to append usage rows. Defaults to api_usage_log.csv next to
        this file.
    note:
        Optional label added to every row (e.g. "generate_content_pools").
        Useful for telling different scripts apart in the log.
    """

    def __init__(
        self,
        api_key: str | None = None,
        log_path: Path | str = DEFAULT_LOG_PATH,
        note: str = "",
        **openai_kwargs: Any,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self._client = OpenAI(api_key=api_key, **openai_kwargs)
        self._log_path = Path(log_path)
        self.chat = _TrackedChat(self._client.chat, self._log_path, note)

    # Pass through any other attribute access to the real client.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Dummy client — zero cost, instant, returns parseable gibberish JSON
# ---------------------------------------------------------------------------

def _random_word(n: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=n))


def _gibberish_content_set(roles: list[str]) -> dict[str, str]:
    """Return a dict with plausible-looking but nonsense values."""
    return {role: _random_word(random.randint(4, 10)).upper() for role in roles}


def _dummy_batch(messages: list[dict], batch_size: int = 5) -> str:
    """
    Produce a JSON array that looks like what generate_content.py expects.
    We guess the roles from the user message; if we can't parse them we
    return generic two-key objects.
    """
    # Try to extract role names from the prompt text.
    import re
    user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
    role_matches = re.findall(r'"(\w+)":', user_text)
    roles = list(dict.fromkeys(role_matches)) or ["field_a", "field_b"]

    batch = [_gibberish_content_set(roles) for _ in range(batch_size)]
    return json.dumps(batch)


class _DummyCompletions:
    def create(self, *, model: str = "dummy", messages: list, max_tokens: int = 1024, **_) -> _FakeCompletion:
        # Guess batch_size from the prompt (look for a number after "Generate exactly")
        import re
        user_text = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        match = re.search(r"Generate exactly (\d+)", user_text)
        batch_size = int(match.group(1)) if match else 5

        content = _dummy_batch(messages, batch_size)

        # Fake token counts (rough heuristic: 1 token ≈ 4 chars)
        prompt_chars = sum(len(msg.get("content", "")) for msg in messages)
        prompt_tokens = max(1, prompt_chars // 4)
        completion_tokens = max(1, len(content) // 4)

        return _FakeCompletion(
            choices=[_FakeChoice(message=_FakeMessage(content=content))],
            usage=_FakeUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=model,
        )


class _DummyChat:
    completions = _DummyCompletions()


class DummyOpenAI:
    """
    Zero-cost mock that mimics openai.OpenAI for pipeline smoke-testing.

    Returns syntactically valid JSON arrays with nonsense content so that
    the validation / deduplication logic in generate_content.py can be
    exercised without spending any money.

    Usage:
        client = DummyOpenAI()
        # Then use exactly as you would TrackedOpenAI / OpenAI.
    """

    chat = _DummyChat()

    def __init__(self, **_: Any) -> None:
        print("  [api_tracker] DummyOpenAI active — no real API calls will be made.")


# ---------------------------------------------------------------------------
# Protocol — type-hint for any compatible client (OpenAI / Tracked / Dummy)
# ---------------------------------------------------------------------------

@runtime_checkable
class OpenAIClient(Protocol):
    """Structural type for any client with a .chat.completions.create interface."""
    chat: Any
