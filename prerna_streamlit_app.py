import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from prerna_adaptive_engine import AdaptiveEngine


BASE_DIR = Path(__file__).resolve().parent
BANK_PATH = BASE_DIR / "prerna_Q27_Q95_canonical_question_bank.csv"


st.set_page_config(
    page_title="Prerna Adaptive Learning",
    page_icon="📘",
    layout="wide",
)


@st.cache_data
def load_bank() -> pd.DataFrame:
    return pd.read_csv(BANK_PATH)


def create_engine(seed: int | None = None) -> AdaptiveEngine:
    bank = load_bank()
    if seed is None:
        seed = int(uuid.uuid4().int % (2**32))
    return AdaptiveEngine(bank, seed=seed, initial_ability=3.0)


def bootstrap() -> None:
    if "run_id" not in st.session_state:
        st.session_state.run_id = str(uuid.uuid4())

    if "engine" not in st.session_state:
        st.session_state.engine = create_engine()

    if "current_question" not in st.session_state:
        st.session_state.current_question = None

    if "feedback" not in st.session_state:
        st.session_state.feedback = None


def get_current_question():
    engine = st.session_state.engine

    if st.session_state.current_question is None:
        # Preview is side-effect free. The question is only recorded as seen
        # when the student commits to starting it.
        st.session_state.current_question = engine.preview_question()

    return st.session_state.current_question


def commit_current_question():
    engine = st.session_state.engine
    q = st.session_state.current_question

    if q is None:
        q = engine.select_question()
        st.session_state.current_question = q
    else:
        # The preview did not modify engine state. Select again now that the
        # student has committed. RNG restoration makes this deterministic.
        q = engine.select_question()
        st.session_state.current_question = q

    return q


def reset_demo() -> None:
    st.session_state.run_id = str(uuid.uuid4())
    st.session_state.engine = create_engine()
    st.session_state.current_question = None
    st.session_state.feedback = None


def render_dashboard(engine: AdaptiveEngine) -> None:
    state = engine.state

    col1, col2, col3 = st.columns(3)
    col1.metric("Ability", f"{state.ability:.2f} / 5")
    col2.metric("Accuracy", f"{state.accuracy:.0%}")
    col3.metric("Attempted", state.attempted)

    st.progress(max(0.0, min(1.0, state.ability / 5.0)))

    st.markdown("### Topic mastery")

    # Keep the source's subtopic names exactly as represented in the bank.
    topics = sorted(state.topic_mastery.items(), key=lambda item: item[0])

    for topic, mastery in topics:
        st.write(f"**{topic}** — {mastery:.0%}")
        st.progress(mastery)


def render_feedback(feedback: dict) -> None:
    if feedback["is_correct"]:
        st.success("Correct ✓")
    else:
        st.error(
            f"Incorrect. Correct option: {feedback['correct_option']}"
        )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Ability",
        f"{feedback['ability_after']:.2f}",
        delta=f"{feedback['ability_after'] - feedback['ability_before']:+.2f}",
    )
    c2.metric("Topic mastery", f"{feedback['topic_mastery_after']:.0%}")
    c3.metric("Overall accuracy", f"{feedback['accuracy']:.0%}")


def main() -> None:
    bootstrap()
    engine = st.session_state.engine

    st.title("📘 Prerna Adaptive Learning")
    st.caption("Adaptive demo using the Prerna Classes Q27–Q95 question bank")

    left, right = st.columns([2.1, 1])

    with right:
        with st.container(border=True):
            st.markdown("### Student dashboard")
            render_dashboard(engine)

            if st.button("🔄 Reset student", use_container_width=True):
                reset_demo()
                st.rerun()

    with left:
        q = get_current_question()

        if st.session_state.feedback is None:
            st.markdown("### Next question")

            st.caption(
                f"Question Q{int(q['question_id'])}  •  "
                f"Difficulty {int(q['difficulty'])}"
            )

            st.markdown(str(q["question_text"]))

            with st.form("answer_form"):
                option_labels = {
                    "A": str(q["option_A"]),
                    "B": str(q["option_B"]),
                    "C": str(q["option_C"]),
                    "D": str(q["option_D"]),
                }

                selected = st.radio(
                    "Choose an answer:",
                    options=list(option_labels.keys()),
                    format_func=lambda x: f"{x}. {option_labels[x]}",
                )

                submitted = st.form_submit_button(
                    "Submit answer",
                    use_container_width=True,
                )

            if submitted:
                try:
                    # Commit the previewed question to the engine first.
                    committed = commit_current_question()

                    # Then process exactly one answer.
                    feedback = engine.answer(
                        int(committed["question_id"]),
                        selected,
                    )

                    st.session_state.feedback = feedback
                    st.rerun()

                except ValueError as exc:
                    st.error(str(exc))

        else:
            feedback = st.session_state.feedback

            st.markdown(
                f"### Q{feedback['question_id']} — Result"
            )

            render_feedback(feedback)

            if st.button("Next adaptive question", use_container_width=True):
                st.session_state.feedback = None
                st.session_state.current_question = None
                st.rerun()

    st.divider()
    st.caption(
        f"Run ID: {st.session_state.run_id[:8]}  •  "
        "Baseline heuristic engine • Q27–Q95"
    )


if __name__ == "__main__":
    main()
