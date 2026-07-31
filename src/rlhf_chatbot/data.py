"""Preference-dataset adapters with no import-time Spark side effects."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str | None = None


T = TypeVar("T")


def batched(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    if hasattr(value, "asDict"):
        return value.asDict(recursive=True).get(key)
    return getattr(value, key, None)


def assistant_response(messages: Any) -> str:
    """Extract the final assistant response from common chat-message schemas."""

    if isinstance(messages, str):
        return messages
    if not messages:
        raise ValueError("A preference response is empty.")

    for message in reversed(list(messages)):
        role = _field(message, "role")
        content = _field(message, "content")
        if role == "assistant" and content:
            return str(content)

    content = _field(list(messages)[-1], "content")
    if content:
        return str(content)
    raise ValueError("No assistant message was found in the preference response.")


def iter_parquet_preferences(
    dataset_path: str | Path,
    *,
    limit: int | None = None,
) -> Iterator[PreferenceExample]:
    """Stream UltraFeedback-style rows through Spark.

    PySpark is imported lazily so inference does not require Java or start a Spark
    session merely because the package was imported.
    """

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("rlhf-preference-loader").getOrCreate()
    frame = spark.read.parquet(str(dataset_path)).select("prompt", "chosen", "rejected")
    if limit is not None:
        frame = frame.limit(limit)

    for row in frame.toLocalIterator():
        yield PreferenceExample(
            prompt=str(row["prompt"]),
            chosen=assistant_response(row["chosen"]),
            rejected=assistant_response(row["rejected"]),
        )
