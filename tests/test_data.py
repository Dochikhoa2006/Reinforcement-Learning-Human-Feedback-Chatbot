from dataclasses import dataclass

import pytest

from rlhf_chatbot.data import assistant_response, batched


@dataclass
class Message:
    role: str
    content: str


def test_assistant_response_supports_dictionary_messages() -> None:
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Preferred answer"},
    ]

    assert assistant_response(messages) == "Preferred answer"


def test_assistant_response_supports_object_messages() -> None:
    messages = [Message("user", "Question"), Message("assistant", "Answer")]

    assert assistant_response(messages) == "Answer"


def test_assistant_response_rejects_empty_conversation() -> None:
    with pytest.raises(ValueError, match="empty"):
        assistant_response([])


def test_batched_retains_final_partial_batch() -> None:
    assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]
