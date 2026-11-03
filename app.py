"""Web page for the assistant. Run with:  streamlit run app.py

Optional settings (environment variables, or Streamlit secrets when hosted):
  APP_PASSWORD           if set, visitors must enter it before asking questions
  MAX_QUESTIONS          questions allowed per browser session (default 15)
"""
import os

import streamlit as st

# Copy Streamlit secrets (API key, settings) into environment variables BEFORE
# importing the project modules, because they read their settings on import.
try:
    for key, value in st.secrets.items():
        if isinstance(value, (str, int, float)):
            os.environ.setdefault(key, str(value))
except Exception:
    pass  # no secrets file locally; that's fine

from common import CHAT_BACKEND, CHAT_MODEL, EMBED_MODEL, REFUSAL  # noqa: E402
from rag import answer  # noqa: E402

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "15"))
MAX_CHARS = 400

st.set_page_config(page_title="Regulatory Guidance Assistant", page_icon="📄")
st.title("Regulatory Guidance Assistant")
where = "running locally" if CHAT_BACKEND == "ollama" else "answers by Claude via the Anthropic API"
st.caption(f"Answers only from five public documents on model risk, AI risk and third-party risk, "
           f"with citations · {CHAT_MODEL} · {where}")

with st.expander("About this demo"):
    st.markdown(
        "A personal proof of concept using public documents only: SR 26-2, SR 11-7 (superseded), "
        "the NIST AI Risk Management Framework, the NIST Generative AI Profile, and the 2023 "
        "interagency third-party risk guidance. It is not legal or regulatory advice, and answers "
        "can be incomplete or wrong. Always check the cited passage.")
    if CHAT_BACKEND == "anthropic":
        st.markdown("Questions are sent to Anthropic's API to generate answers. "
                    "Please don't enter confidential information.")

# Optional password gate.
if APP_PASSWORD and not st.session_state.get("unlocked"):
    pw = st.text_input("Password", type="password")
    if pw == APP_PASSWORD:
        st.session_state.unlocked = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    st.stop()

st.session_state.setdefault("asked", 0)


@st.cache_data(show_spinner=False, max_entries=500)
def cached_answer(question, mode):
    """Identical questions reuse the earlier answer instead of calling the model again."""
    return answer(question, mode=mode)


# Example questions: each shows a different kind of question, including one the
# documents can't answer, so visitors see the assistant decline as well as answer.
EXAMPLES = [
    ("Scope", "Does SR 26-2 apply to generative AI and agentic AI?"),
    ("A list", "What are the stages of the third-party relationship life cycle?"),
    ("A comparison", "What changed between SR 11-7 and SR 26-2 on effective challenge?"),
    ("Out of scope", "What are the capital requirements under Basel III?"),
]


def use_example(text):
    st.session_state.question = text
    st.session_state.run_now = True  # examples run immediately


st.caption("Try an example, or type your own question below.")
cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES):
    col.button(label, help=text, on_click=use_example, args=(text,), width="stretch")

# A form, so a question runs when the visitor clicks Ask (or presses Enter).
with st.form("ask_form", border=False):
    st.text_input("Ask a question about the documents", max_chars=MAX_CHARS, key="question")
    mode = st.radio("Retrieval", ["hybrid", "vector", "keyword"], horizontal=True, key="mode")
    submitted = st.form_submit_button("Ask", type="primary")

question = (st.session_state.get("question") or "").strip()
if (submitted or st.session_state.pop("run_now", False)) and question:
    # Count a question only when it's new, not when the same one is asked again.
    current = (question, mode)
    is_new = st.session_state.get("last_asked") != current
    if is_new and st.session_state.asked >= MAX_QUESTIONS:
        st.warning(f"This demo allows {MAX_QUESTIONS} questions per visit. Thanks for trying it.")
        st.stop()
    with st.spinner("Searching and drafting an answer..."):
        try:
            st.session_state.result = cached_answer(question, mode)
        except Exception:
            st.error("Something went wrong generating the answer. Please try again shortly.")
            st.stop()
    if is_new:
        st.session_state.asked += 1
        st.session_state.last_asked = current

result = st.session_state.get("result")
if result:
    st.markdown(result["answer"])
    if REFUSAL.lower().rstrip(".") in result["answer"].lower():
        st.caption("No sources shown, because the documents don't answer this question.")
    else:
        st.subheader("Sources")
        for i, s in enumerate(result["sources"], start=1):
            with st.expander(f"[{i}] {s['doc']}, page {s['page']}"):
                st.write(s["text"].replace(f"Document: {s['doc']}. ", ""))
    st.caption(f"Question {st.session_state.asked} of {MAX_QUESTIONS} this visit.")
