"""Interactive Streamlit demo for the PPO-aligned policy."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from rlhf_chatbot.config import load_settings
from rlhf_chatbot.models.policy import PolicyValueModel

st.set_page_config(
    page_title="RLHF Research Chatbot",
    page_icon="🤖",
    layout="centered",
)

settings = load_settings()
default_checkpoint = settings.paths.resolve(settings.paths.ppo_checkpoint)
checkpoint = Path(os.getenv("RLHF_MODEL_DIR", str(default_checkpoint))).expanduser()


@st.cache_resource(show_spinner=False)
def load_policy(checkpoint_path: str) -> PolicyValueModel:
    return PolicyValueModel.from_checkpoint(checkpoint_path)


st.title("RLHF Research Chatbot")
st.caption("GPT-2 aligned with pairwise preferences, a RoBERTa reward model, and PPO")

with st.sidebar:
    st.header("Generation")
    deterministic = st.toggle("Deterministic", value=True)
    max_new_tokens = st.slider("Maximum new tokens", 16, 256, 128, 16)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.1)
    top_k = st.slider("Top-k", 1, 100, 50)
    st.divider()
    st.caption(f"Checkpoint: `{checkpoint}`")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

try:
    with st.spinner("Loading the PPO checkpoint…"):
        policy = load_policy(str(checkpoint))
except Exception as error:
    st.error(f"Unable to load the model checkpoint: {error}")
    st.info("Set `RLHF_MODEL_DIR` to a valid PPO checkpoint directory.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask the aligned GPT-2 policy a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating response…"):
            answer = policy.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=not deterministic,
            )
        answer = answer or "The model produced an empty response."
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption(
    "Research demo only. GPT-2 is a small legacy language model and may produce inaccurate, "
    "biased, or unsafe text."
)
