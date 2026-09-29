import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from prerna_adaptive_engine import AdaptiveEngine

BASE_DIR = Path(__file__).resolve().parent
BANK_PATH = BASE_DIR / "prerna_Q27_Q95_enhanced_question_bank.csv"
THEORY_PATH = BASE_DIR / "prerna_Q27_Q95_theory.csv"

st.set_page_config(page_title="Prerna Adaptive Learning", page_icon="📘", layout="wide")


@st.cache_data
def load_bank() -> pd.DataFrame:
    return pd.read_csv(BANK_PATH, encoding="utf-8-sig")


@st.cache_data
def load_theory() -> pd.DataFrame:
    return pd.read_csv(THEORY_PATH, encoding="utf-8-sig")


def create_engine(seed=None):
    bank = load_bank()
    if seed is None:
        seed = int(uuid.uuid4().int % (2**32))
    return AdaptiveEngine(bank, seed=seed, initial_ability=3.0)


def bootstrap():
    defaults = {
        "run_id": str(uuid.uuid4()),
        "engine": None,
        "current_question": None,
        "feedback": None,
        "show_hint_1": False,
        "show_hint_2": False,
        "show_solution": False,
        "question_attempt_id": str(uuid.uuid4()),
        "mode": "📚 Theory",
        "bank_topic": "All topics",
        "bank_difficulty": "All difficulties",
        "bank_sort": "Question Number → Q27 → Q95",
        "bank_status": "All questions",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.engine is None:
        st.session_state.engine = create_engine()


def reset_question_state():
    st.session_state.current_question = None
    st.session_state.feedback = None
    st.session_state.show_hint_1 = False
    st.session_state.show_hint_2 = False
    st.session_state.show_solution = False
    st.session_state.question_attempt_id = str(uuid.uuid4())


def reset_demo():
    st.session_state.run_id = str(uuid.uuid4())
    st.session_state.engine = create_engine()
    reset_question_state()


def get_current_question():
    if st.session_state.current_question is None:
        st.session_state.current_question = st.session_state.engine.preview_question()
    return st.session_state.current_question


def commit_current_question():
    q = st.session_state.current_question
    if q is None:
        q = st.session_state.engine.preview_question()
        st.session_state.current_question = q
    st.session_state.engine.commit_selection(int(q["question_id"]))
    return q


def render_theory():
    theory = load_theory().copy()
    st.markdown("## 📚 Theory")
    st.caption("Study the concept first, then move to Adaptive Practice or the Question Bank.")

    if theory.empty:
        st.warning("No theory content is available.")
        return

    display_topics = theory["topic"].astype(str).tolist()
    selected = st.selectbox("Choose a topic", display_topics, key="theory_topic_control")
    row = theory[theory["topic"].astype(str) == selected].iloc[0]

    st.markdown(f"### {row['topic']}")
    st.markdown(f"**Core idea:** {row['core_idea']}")
    st.markdown("**Key formulas:**")
    st.info(str(row["key_formulas"]))
    st.markdown("**How to use:**")
    st.write(str(row["how_to_use"]))


def render_feedback(feedback):
    if feedback["is_correct"]:
        st.success("Correct ✓")
    else:
        st.error(f"Incorrect. Correct option: {feedback['correct_option']}")
        if feedback.get("is_retry"):
            st.info("This question stays in your review pool and will come back again until you solve it correctly.")
        else:
            st.info("This question has been added to your review pool and will come back again.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ability", f"{feedback['ability_after']:.2f}", delta=f"{feedback['ability_after'] - feedback['ability_before']:+.2f}")
    c2.metric("Topic mastery", f"{feedback['topic_mastery_after']:.0%}")
    c3.metric("Overall accuracy", f"{feedback['accuracy']:.0%}")


def _clean(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def render_hints(q):
    """Show Hint 1 and Hint 2 directly on the question page."""
    hint_1 = _clean(q.get("hint_level_1", ""))
    hint_2 = _clean(q.get("hint_level_2", ""))

    st.markdown("#### 💡 Hints")
    h1, h2 = st.columns(2)

    with h1:
        if hint_1:
            if not st.session_state.show_hint_1:
                if st.button("💡 Hint 1", key=f"hint1_{int(q['question_id'])}_{st.session_state.question_attempt_id}", use_container_width=True):
                    st.session_state.show_hint_1 = True
                    st.rerun()
            else:
                st.success("**Hint 1**")
                st.info(hint_1)

    with h2:
        if hint_2:
            if not st.session_state.show_hint_2:
                if st.button("💡 Hint 2", key=f"hint2_{int(q['question_id'])}_{st.session_state.question_attempt_id}", use_container_width=True):
                    st.session_state.show_hint_2 = True
                    st.rerun()
            else:
                st.success("**Hint 2**")
                st.info(hint_2)


def render_learning_support(q):
    """Show hints on the question page and reveal the full solution only after submission."""
    render_hints(q)

    solution = _clean(q.get("solution", ""))
    if solution:
        st.markdown("### 📖 Solution")
        if not st.session_state.show_solution:
            if st.button("📖 Show Solution", key=f"solution_{int(q['question_id'])}_{st.session_state.question_attempt_id}", use_container_width=True):
                st.session_state.show_solution = True
                st.rerun()
        else:
            st.markdown("**Solution:**")
            st.code(solution, language=None)

    verification = _clean(q.get("verification_status", ""))
    if "inconsistency" in verification.lower():
        st.warning("Source note: this question retains a source-inconsistency flag. The demo does not silently alter the source.")


def render_question(q):
    qid = int(q["question_id"])
    st.caption(f"Question Q{qid}  •  Difficulty {int(q['difficulty'])}  •  {q['subtopic']}")
    st.markdown(str(q["question_text"]))

    engine = st.session_state.engine
    already_correct = (
        qid in engine.state.ever_answered
        and qid not in engine.state.wrong_questions
    )

    if already_correct:
        st.info("🔎 Review mode — you have already answered this question correctly. A new submission will not be counted again.")
        option_labels = {k: str(q[f"option_{k}"]) for k in "ABCD"}
        st.markdown("**Options:**")
        for option, label in option_labels.items():
            st.markdown(f"**{option}.** {label}")
        st.markdown(f"**Correct option: {q["correct_option"]}**")
        submitted = False
    else:
        with st.form(f"answer_form_{qid}_{st.session_state.question_attempt_id}"):
            option_labels = {k: str(q[f"option_{k}"]) for k in "ABCD"}
            selected = st.radio(
                "Choose an answer:",
                list(option_labels),
                format_func=lambda x: f"{x}. {option_labels[x]}",
                key=f"answer_{qid}_{st.session_state.question_attempt_id}",
            )
            submitted = st.form_submit_button("Submit answer", use_container_width=True)

    # Hints are part of the question page and are available before submission.
    if already_correct:
        render_learning_support(q)
    else:
        render_hints(q)

    if submitted:
        try:
            committed = commit_current_question()
            feedback = st.session_state.engine.answer(int(committed["question_id"]), selected)
            st.session_state.feedback = feedback
            st.session_state.show_solution = False
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

def render_question_bank():
    bank = load_bank().copy()
    engine = st.session_state.engine

    st.markdown("## 📝 Question Bank")
    st.caption("Choose a manual ordering, then filter by topic, difficulty, or status. Adaptive Practice remains separate and is controlled by the adaptive engine.")

    st.markdown("### 🎛️ Practice controls")
    sort_options = [
        "Question Number → Q27 → Q95",
        "Question Number → Q95 → Q27",
        "Easy → Hard",
        "Hard → Easy",
        "Random",
    ]
    status_options = [
        "All questions",
        "Unseen",
        "Attempted",
        "Correct",
        "Incorrect / Wrong",
        "Due for Retry",
    ]

    c1, c2 = st.columns(2)
    with c1:
        sort_order = st.selectbox(
            "Sort questions by",
            sort_options,
            index=sort_options.index(st.session_state.bank_sort)
            if st.session_state.bank_sort in sort_options else 0,
            key="question_bank_sort_control",
        )
    with c2:
        status = st.selectbox(
            "Question status",
            status_options,
            index=status_options.index(st.session_state.bank_status)
            if st.session_state.bank_status in status_options else 0,
            key="question_bank_status_control",
        )

    st.session_state.bank_sort = sort_order
    st.session_state.bank_status = status

    topics = ["All topics"] + sorted(bank["subtopic"].dropna().unique().tolist())
    difficulties = ["All difficulties", "1", "2", "3", "4", "5"]

    c1, c2 = st.columns(2)
    with c1:
        topic = st.selectbox(
            "Topic",
            topics,
            index=topics.index(st.session_state.bank_topic)
            if st.session_state.bank_topic in topics else 0,
            key="question_bank_topic_control",
        )
    with c2:
        difficulty = st.selectbox(
            "Difficulty",
            difficulties,
            index=difficulties.index(st.session_state.bank_difficulty)
            if st.session_state.bank_difficulty in difficulties else 0,
            key="question_bank_difficulty_control",
        )

    st.session_state.bank_topic = topic
    st.session_state.bank_difficulty = difficulty

    filtered = bank.copy()

    if topic != "All topics":
        filtered = filtered[filtered["subtopic"] == topic]

    if difficulty != "All difficulties":
        filtered = filtered[filtered["difficulty"].astype(int) == int(difficulty)]

    seen = set(engine.state.seen)
    answered = set(engine.state.ever_answered)
    wrong = set(engine.state.wrong_questions)
    due = {
        qid for qid, retry_at in engine.state.retry_after_attempt.items()
        if qid in wrong and engine.state.attempted >= retry_at
    }

    if status == "Unseen":
        filtered = filtered[~filtered["question_id"].isin(seen)]
    elif status == "Attempted":
        filtered = filtered[filtered["question_id"].isin(answered)]
    elif status == "Correct":
        correct_ids = answered - wrong
        filtered = filtered[filtered["question_id"].isin(correct_ids)]
    elif status == "Incorrect / Wrong":
        filtered = filtered[filtered["question_id"].isin(wrong)]
    elif status == "Due for Retry":
        filtered = filtered[filtered["question_id"].isin(due)]

    if filtered.empty:
        st.warning("No questions match these filters.")
        return

    if sort_order == "Question Number → Q27 → Q95":
        filtered = filtered.sort_values("question_id", ascending=True)
    elif sort_order == "Question Number → Q95 → Q27":
        filtered = filtered.sort_values("question_id", ascending=False)
    elif sort_order == "Easy → Hard":
        filtered = filtered.sort_values(["difficulty", "question_id"], ascending=[True, True])
    elif sort_order == "Hard → Easy":
        filtered = filtered.sort_values(["difficulty", "question_id"], ascending=[False, True])
    else:  # Random
        filtered = filtered.sample(frac=1, random_state=int(uuid.uuid4().int % (2**32)))

    st.write(f"**{len(filtered)} questions available**")

    display_rows = filtered[["question_id", "subtopic", "difficulty"]].copy()
    display_rows["question_id"] = display_rows["question_id"].astype(int)
    display_rows["difficulty"] = display_rows["difficulty"].astype(int)
    display_rows = display_rows.rename(
        columns={"question_id": "Question", "subtopic": "Topic", "difficulty": "Difficulty"}
    )
    st.dataframe(display_rows, use_container_width=True, hide_index=True)

    ids = filtered["question_id"].astype(int).tolist()
    labels = [
        f"Q{qid} — Difficulty {int(row['difficulty'])} — {row['subtopic']}"
        for qid, (_, row) in zip(ids, filtered.iterrows())
    ]
    selected_label = st.selectbox("Choose question", labels, key="question_bank_question_control")
    selected_qid = ids[labels.index(selected_label)]

    if st.button("▶️ Start this question", use_container_width=True):
        st.session_state.current_question = engine.select_specific(selected_qid)
        st.session_state.feedback = None
        st.session_state.show_hint_1 = False
        st.session_state.show_hint_2 = False
        st.session_state.show_solution = False
        st.session_state.question_attempt_id = str(uuid.uuid4())
        st.rerun()

    if st.session_state.current_question is not None:
        q = st.session_state.current_question
        st.markdown("---")
        if st.session_state.feedback is None:
            render_question(q)
        else:
            st.markdown(f"### Q{st.session_state.feedback['question_id']} — Result")
            render_feedback(st.session_state.feedback)
            render_learning_support(q)

        if st.button("Next question in this order", use_container_width=True):
            next_ids = ids
            current_id = int(q["question_id"])
            try:
                current_index = next_ids.index(current_id)
                next_id = next_ids[(current_index + 1) % len(next_ids)]
            except ValueError:
                next_id = next_ids[0]
            st.session_state.current_question = engine.select_specific(next_id)
            st.session_state.feedback = None
            st.session_state.show_hint_1 = False
            st.session_state.show_hint_2 = False
            st.session_state.show_solution = False
            st.session_state.question_attempt_id = str(uuid.uuid4())
            st.rerun()

def render_wrong_questions():
    engine = st.session_state.engine
    wrong = sorted(engine.state.wrong_questions)
    st.markdown("## 🔁 Wrong Questions Review")
    st.caption("Questions you got wrong are kept here and can be retried until you solve them correctly.")
    if not wrong:
        st.success("No wrong questions right now. Keep going — this list will populate automatically when you make a mistake.")
        return

    bank = load_bank()
    rows = bank[bank["question_id"].isin(wrong)].copy()
    rows = rows.sort_values("question_id")
    st.write(f"**{len(rows)} questions in your review pool**")
    st.dataframe(rows[["question_id", "subtopic", "difficulty"]].rename(columns={"question_id": "Question", "subtopic": "Topic", "difficulty": "Difficulty"}), use_container_width=True, hide_index=True)

    qid = st.selectbox("Choose a wrong question to retry", rows["question_id"].astype(int).tolist())
    if st.button("🔁 Retry this question", use_container_width=True):
        st.session_state.current_question = engine.select_specific(qid)
        st.session_state.feedback = None
        st.session_state.show_hint_1 = False
        st.session_state.show_hint_2 = False
        st.session_state.show_solution = False
        st.rerun()

    if st.session_state.current_question is not None:
        q = st.session_state.current_question
        if st.session_state.feedback is None:
            st.markdown("---")
            render_question(q)
        else:
            st.markdown("---")
            render_feedback(st.session_state.feedback)
            render_learning_support(q)
            if st.button("Back to wrong-question list", use_container_width=True):
                reset_question_state()
                st.rerun()


def render_adaptive():
    st.markdown("## 🧠 Adaptive Practice")
    st.caption("The engine adjusts difficulty and topic focus based on your performance, while also bringing back questions you got wrong.")
    if st.session_state.feedback is None:
        try:
            q = get_current_question()
            render_question(q)
        except ValueError as exc:
            st.success("🎉 Adaptive cycle complete")
            st.info(str(exc))
            st.markdown("Use **Question Bank** for manual practice or **Wrong Questions Review** to continue targeted revision.")
    else:
        q = st.session_state.current_question
        st.markdown(f"### Q{st.session_state.feedback['question_id']} — Result")
        render_feedback(st.session_state.feedback)
        render_learning_support(q)
        if st.button("Next adaptive / review question", use_container_width=True):
            reset_question_state()
            st.rerun()


def main():
    bootstrap()
    engine = st.session_state.engine

    st.title("📘 Prerna Adaptive Learning")
    st.caption("Q27–Q95 • Theory → Practice → Review")

    st.sidebar.markdown("## Navigation")
    modes = ["📚 Theory", "🧠 Adaptive Practice", "📝 Question Bank", "🔁 Wrong Questions"]
    previous_mode = st.session_state.mode
    mode = st.sidebar.radio("Go to", modes, index=modes.index(previous_mode))
    if mode != previous_mode:
        reset_question_state()
    st.session_state.mode = mode

    with st.sidebar:
        st.divider()
        st.markdown("### Student dashboard")
        st.metric("Ability", f"{engine.state.ability:.2f}/5")
        st.metric("Accuracy", f"{engine.state.accuracy:.0%}")
        st.metric("Attempted", engine.state.attempted)
        st.metric("Wrong to review", len(engine.state.wrong_questions))
        st.divider()
        if st.button("🔄 Reset student", use_container_width=True):
            reset_demo()
            st.rerun()

    if mode == "📚 Theory":
        render_theory()
    elif mode == "🧠 Adaptive Practice":
        render_adaptive()
    elif mode == "📝 Question Bank":
        render_question_bank()
    else:
        render_wrong_questions()

    st.divider()
    st.caption(f"Run ID: {st.session_state.run_id[:8]} • Heuristic adaptive engine • Q27–Q95")


if __name__ == "__main__":
    main()
