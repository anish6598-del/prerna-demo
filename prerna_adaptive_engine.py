from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict
import random
import pandas as pd


@dataclass
class StudentState:
    """State maintained by the baseline adaptive engine."""

    ability: float = 3.0
    topic_mastery: Dict[str, float] = field(default_factory=dict)
    asked: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    seen: set[int] = field(default_factory=set)
    # Questions that have been answered at least once. Never discarded.
    ever_answered: set[int] = field(default_factory=set)

    # Questions whose current on-screen attempt has already been submitted.
    answered: set[int] = field(default_factory=set)
    wrong_questions: set[int] = field(default_factory=set)
    retry_after_attempt: Dict[int, int] = field(default_factory=dict)
    correct: int = 0
    attempted: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempted if self.attempted else 0.0


class AdaptiveEngine:
    """
    Heuristic adaptive engine for Prerna Classes Q27-Q95.

    It adapts using ability, topic weakness, novelty and a retry priority for
    questions answered incorrectly. Wrong questions are brought back later;
    once answered correctly on the retry, they leave the retry pool.
    """

    VALID_OPTIONS = {"A", "B", "C", "D"}

    def __init__(self, question_bank: pd.DataFrame, seed: int = 42, initial_ability: float = 3.0):
        required = {
            "question_id", "question_text", "option_A", "option_B", "option_C",
            "option_D", "correct_option", "subtopic", "difficulty",
        }
        missing = required - set(question_bank.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        if question_bank.empty:
            raise ValueError("Question bank cannot be empty.")

        self.bank = question_bank.copy().reset_index(drop=True)
        try:
            self.bank["question_id"] = self.bank["question_id"].astype(int)
        except (TypeError, ValueError) as exc:
            raise ValueError("question_id must contain integer-like values.") from exc
        if not self.bank["question_id"].is_unique:
            raise ValueError("question_id values must be unique.")

        self.bank["correct_option"] = self.bank["correct_option"].astype(str).str.strip().str.upper()
        invalid_answers = set(self.bank["correct_option"]) - self.VALID_OPTIONS
        if invalid_answers:
            raise ValueError(f"Invalid correct_option values in question bank: {sorted(invalid_answers)}")

        try:
            self.bank["difficulty"] = pd.to_numeric(self.bank["difficulty"], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("difficulty must be numeric.") from exc
        if self.bank["difficulty"].isna().any():
            raise ValueError("difficulty cannot contain NaN.")
        if not self.bank["difficulty"].between(1, 5).all():
            raise ValueError("difficulty must be between 1 and 5.")

        if self.bank["subtopic"].isna().any():
            raise ValueError("subtopic cannot contain NaN.")
        self.bank["subtopic"] = self.bank["subtopic"].astype(str).str.strip()
        if (self.bank["subtopic"] == "").any():
            raise ValueError("subtopic cannot be empty.")

        if isinstance(initial_ability, bool):
            raise ValueError("initial_ability must be numeric, not bool.")
        try:
            initial_ability = float(initial_ability)
        except (TypeError, ValueError) as exc:
            raise ValueError("initial_ability must be numeric.") from exc
        if pd.isna(initial_ability):
            raise ValueError("initial_ability cannot be NaN.")

        self.state = StudentState(ability=max(1.0, min(5.0, initial_ability)))
        self.rng = random.Random(seed)
        for topic in self.bank["subtopic"].unique():
            self.state.topic_mastery[str(topic)] = 0.50

    def _topic_mastery(self, topic: str) -> float:
        return self.state.topic_mastery.get(topic, 0.50)

    def _question_score(self, row: pd.Series) -> float:
        difficulty = float(row["difficulty"])
        topic = str(row["subtopic"])
        mastery = self._topic_mastery(topic)
        difficulty_match = 1.0 - min(abs(difficulty - self.state.ability) / 4.0, 1.0)
        weakness = 1.0 - mastery
        qid = int(row["question_id"])
        novelty = 0.18 if qid not in self.state.seen else -0.30
        retry_due = (
            qid in self.state.wrong_questions
            and self.state.attempted >= self.state.retry_after_attempt.get(qid, 0)
        )
        retry_bonus = 0.72 if retry_due else 0.0
        exploration = self.rng.random() * 0.08
        return 0.62 * difficulty_match + 0.30 * weakness + novelty + retry_bonus + exploration

    def _record_selection(self, qid: int) -> None:
        self.state.asked.append(qid)
        self.state.seen.add(qid)
        # Reopening a wrong question makes it answerable again, while a
        # question that is currently on screen still cannot be double-submitted.
        # Explicitly selecting a question for practice (adaptive retry or
        # question-bank browse) opens a fresh attempt for that question.
        self.state.answered.discard(qid)

    def select_question(self) -> pd.Series:
        """Select and record the next adaptive question."""
        unseen = self.bank[~self.bank["question_id"].isin(self.state.seen)].copy()
        due_retry_ids = {
            qid for qid in self.state.wrong_questions
            if self.state.attempted >= self.state.retry_after_attempt.get(qid, float("inf"))
        }
        retry = self.bank[self.bank["question_id"].isin(due_retry_ids)].copy()

        # Combine genuinely new questions with retries that are actually due.
        # A wrong question that is not yet due is not selected merely because it
        # is in the wrong pool.
        if unseen.empty:
            recent = set(list(dict.fromkeys(reversed(self.state.asked)))[:3])
            candidates = retry.copy()

            if candidates.empty:
                # The student has exhausted all unseen questions and has no
                # retry currently due. Do not fall back to an already-correct
                # question: answer() intentionally refuses to count such a
                # question again. Instead, surface a clean completion state
                # to the UI so the student can use manual review.
                completed_correct = self.state.ever_answered - self.state.wrong_questions
                candidates = self.bank[
                    ~self.bank["question_id"].isin(recent)
                    & ~self.bank["question_id"].isin(completed_correct)
                ].copy()

            if candidates.empty:
                raise ValueError(
                    "No adaptive question is currently available. "
                    "You have completed the available questions; use the Question Bank "
                    "or Wrong Questions Review for further practice."
                )
        else:
            candidates = pd.concat([unseen, retry], ignore_index=True)
            candidates = candidates.drop_duplicates(subset=["question_id"], keep="first")

        candidates["_score"] = candidates.apply(self._question_score, axis=1)
        candidates["_ability_distance"] = (candidates["difficulty"] - self.state.ability).abs()
        best = candidates.sort_values(["_score", "_ability_distance"], ascending=[False, True]).iloc[0]
        qid = int(best["question_id"])
        self._record_selection(qid)
        return best.drop(labels=["_score", "_ability_distance"], errors="ignore")

    def select_specific(self, question_id: int) -> pd.Series:
        """Select a requested question for browse/filter practice."""
        qid = int(question_id)
        matches = self.bank[self.bank["question_id"] == qid]
        if matches.empty:
            raise ValueError(f"Question Q{qid} does not exist in the question bank.")
        self._record_selection(qid)
        return matches.iloc[0].copy()

    def commit_selection(self, question_id: int) -> None:
        """Commit a question that was already displayed via preview_question()."""
        qid = int(question_id)
        matches = self.bank[self.bank["question_id"] == qid]
        if matches.empty:
            raise ValueError(f"Question Q{qid} does not exist in the question bank.")
        self._record_selection(qid)

    def preview_question(self) -> pd.Series:
        asked_backup = self.state.asked.copy()
        seen_backup = self.state.seen.copy()
        answered_backup = self.state.answered.copy()
        ever_answered_backup = self.state.ever_answered.copy()
        retry_after_backup = self.state.retry_after_attempt.copy()
        rng_state = self.rng.getstate()
        try:
            return self.select_question()
        finally:
            self.state.asked.clear()
            self.state.asked.extend(asked_backup)
            self.state.seen.clear()
            self.state.seen.update(seen_backup)
            self.state.answered.clear()
            self.state.answered.update(answered_backup)
            self.state.ever_answered.clear()
            self.state.ever_answered.update(ever_answered_backup)
            self.state.retry_after_attempt.clear()
            self.state.retry_after_attempt.update(retry_after_backup)
            self.rng.setstate(rng_state)

    def answer(self, question_id: int, selected_option: str) -> Dict[str, object]:
        qid = int(question_id)
        selected = str(selected_option).strip().upper()
        if selected not in self.VALID_OPTIONS:
            raise ValueError(f"Invalid option '{selected_option}'. Expected one of {sorted(self.VALID_OPTIONS)}.")
        if qid not in self.state.seen:
            raise ValueError("Question was not selected by the engine.")
        if qid in self.state.answered:
            raise ValueError(f"Question {qid} has already been answered in the current attempt.")

        # Previously correct questions are not silently re-counted when a
        # student merely browses/selects them again. Wrong questions are the
        # explicit exception because they are part of the retry workflow.
        if qid in self.state.ever_answered and qid not in self.state.wrong_questions:
            raise ValueError(f"Question {qid} has already been answered correctly. It is available for review, but will not be counted again.")

        row = self.bank[self.bank["question_id"] == qid].iloc[0]
        correct_option = str(row["correct_option"]).strip().upper()
        is_correct = selected == correct_option
        difficulty = float(row["difficulty"])
        topic = str(row["subtopic"])
        is_retry = qid in self.state.wrong_questions

        old_ability = self.state.ability
        gap = difficulty - old_ability
        learning_rate = 0.22
        if is_correct:
            delta = learning_rate * (0.75 + 0.35 * max(gap, 0.0))
            self.state.correct += 1
            self.state.wrong_questions.discard(qid)
        else:
            delta = -learning_rate * (0.75 + 0.25 * max(-gap, 0.0))
            self.state.wrong_questions.add(qid)

        new_ability = max(1.0, min(5.0, old_ability + delta))
        self.state.ability = new_ability

        old_mastery = self._topic_mastery(topic)
        topic_lr = 0.18 if is_correct else 0.22
        target = 1.0 if is_correct else 0.0
        new_mastery = max(0.0, min(1.0, old_mastery + topic_lr * (target - old_mastery)))
        self.state.topic_mastery[topic] = new_mastery
        self.state.attempted += 1
        self.state.ever_answered.add(qid)
        self.state.answered.add(qid)
        if not is_correct:
            # Bring a wrong question back after two further attempts rather
            # than immediately repeating it. This creates a simple spaced retry.
            self.state.retry_after_attempt[qid] = self.state.attempted + 2
        else:
            self.state.retry_after_attempt.pop(qid, None)

        return {
            "question_id": qid,
            "selected_option": selected,
            "correct_option": correct_option,
            "is_correct": is_correct,
            "difficulty": difficulty,
            "subtopic": topic,
            "ability_before": round(old_ability, 3),
            "ability_after": round(new_ability, 3),
            "topic_mastery_after": round(new_mastery, 3),
            "accuracy": round(self.state.accuracy, 3),
            "is_retry": is_retry,
        }

    def next_question(self) -> pd.Series:
        return self.select_question()


def load_engine(csv_path: str, seed: int = 42, initial_ability: float = 3.0) -> AdaptiveEngine:
    bank = pd.read_csv(csv_path, encoding="utf-8-sig")
    return AdaptiveEngine(bank, seed=seed, initial_ability=initial_ability)
