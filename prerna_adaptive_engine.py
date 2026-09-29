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

    # Full presentation history is not needed for adaptive fallback.
    # Keep only a bounded recent history for "avoid recent" behavior.
    asked: deque[int] = field(default_factory=lambda: deque(maxlen=10))

    # Questions ever presented to the student.
    seen: set[int] = field(default_factory=set)

    # Questions whose answers have already been processed.
    answered: set[int] = field(default_factory=set)

    correct: int = 0
    attempted: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempted if self.attempted else 0.0


class AdaptiveEngine:
    """
    Baseline adaptive engine for Prerna Classes Q27-Q95.

    Difficulty is on a 1-5 scale. The engine is intentionally simple:
      1. Prefer questions close to current ability.
      2. Give extra weight to weak subtopics.
      3. Prefer unseen questions.
      4. Avoid recently attempted questions after exhaustion.
      5. Update ability and subtopic mastery after every response.

    This is a heuristic baseline, not an IRT model.
    """

    VALID_OPTIONS = {"A", "B", "C", "D"}

    def __init__(
        self,
        question_bank: pd.DataFrame,
        seed: int = 42,
        initial_ability: float = 3.0,
    ):
        required = {
            "question_id",
            "question_text",
            "option_A",
            "option_B",
            "option_C",
            "option_D",
            "correct_option",
            "subtopic",
            "difficulty",
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

        self.bank["correct_option"] = (
            self.bank["correct_option"].astype(str).str.strip().str.upper()
        )
        invalid_answers = set(self.bank["correct_option"]) - self.VALID_OPTIONS
        if invalid_answers:
            raise ValueError(
                f"Invalid correct_option values in question bank: "
                f"{sorted(invalid_answers)}"
            )

        try:
            self.bank["difficulty"] = pd.to_numeric(
                self.bank["difficulty"], errors="raise"
            )
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

        self.state = StudentState(
            ability=max(1.0, min(5.0, initial_ability))
        )
        self.rng = random.Random(seed)

        for topic in self.bank["subtopic"].unique():
            self.state.topic_mastery[str(topic)] = 0.50

    def _topic_mastery(self, topic: str) -> float:
        return self.state.topic_mastery.get(topic, 0.50)

    def _question_score(self, row: pd.Series) -> float:
        difficulty = float(row["difficulty"])
        topic = str(row["subtopic"])
        mastery = self._topic_mastery(topic)

        difficulty_match = 1.0 - min(
            abs(difficulty - self.state.ability) / 4.0, 1.0
        )
        weakness = 1.0 - mastery

        exploration = self.rng.random() * 0.08

        qid = int(row["question_id"])
        novelty = 0.18 if qid not in self.state.seen else -0.30

        return (
            0.62 * difficulty_match
            + 0.30 * weakness
            + novelty
            + exploration
        )

    def select_question(self) -> pd.Series:
        """Select and record the next question."""
        available = self.bank[
            ~self.bank["question_id"].isin(self.state.seen)
        ].copy()

        # Once every question has been seen, reuse questions while avoiding
        # the most recent three distinct IDs where possible.
        if available.empty:
            recent = list(dict.fromkeys(reversed(self.state.asked)))[:3]
            recent = set(recent)

            available = self.bank[
                ~self.bank["question_id"].isin(recent)
            ].copy()

            if available.empty:
                available = self.bank.copy()

        available["_score"] = available.apply(self._question_score, axis=1)

        # Primary ordering is adaptive score. For exact score ties, prefer
        # difficulty closest to current ability rather than automatically
        # preferring easier questions.
        available["_ability_distance"] = (
            available["difficulty"] - self.state.ability
        ).abs()

        best = available.sort_values(
            ["_score", "_ability_distance"],
            ascending=[False, True],
        ).iloc[0]

        qid = int(best["question_id"])
        self.state.asked.append(qid)
        self.state.seen.add(qid)

        return best.drop(
            labels=["_score", "_ability_distance"],
            errors="ignore",
        )

    def preview_question(self) -> pd.Series:
        """
        Preview the next question without changing engine state.

        History, seen IDs, and RNG state are restored.
        """
        asked_backup = self.state.asked.copy()
        seen_backup = self.state.seen.copy()
        rng_state = self.rng.getstate()

        try:
            return self.select_question()
        finally:
            self.state.asked.clear()
            self.state.asked.extend(asked_backup)

            self.state.seen.clear()
            self.state.seen.update(seen_backup)

            self.rng.setstate(rng_state)

    def answer(self, question_id: int, selected_option: str) -> Dict[str, object]:
        """Record an answer and update ability/mastery exactly once."""
        try:
            question_id = int(question_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("question_id must be integer-like.") from exc

        selected = str(selected_option).strip().upper()
        if selected not in self.VALID_OPTIONS:
            raise ValueError(
                f"Invalid option '{selected_option}'. "
                f"Expected one of {sorted(self.VALID_OPTIONS)}."
            )

        if question_id not in self.state.seen:
            raise ValueError(
                "Question was not selected by the engine."
            )

        if question_id in self.state.answered:
            raise ValueError(
                f"Question {question_id} has already been answered."
            )

        row = self.bank[
            self.bank["question_id"] == question_id
        ].iloc[0]

        correct_option = str(row["correct_option"]).strip().upper()
        is_correct = selected == correct_option

        difficulty = float(row["difficulty"])
        topic = str(row["subtopic"])

        old_ability = self.state.ability
        gap = difficulty - old_ability
        learning_rate = 0.22

        if is_correct:
            delta = learning_rate * (0.75 + 0.35 * max(gap, 0.0))
            self.state.correct += 1
        else:
            delta = -learning_rate * (0.75 + 0.25 * max(-gap, 0.0))

        new_ability = max(1.0, min(5.0, old_ability + delta))
        self.state.ability = new_ability

        old_mastery = self._topic_mastery(topic)
        topic_lr = 0.18 if is_correct else 0.22
        target = 1.0 if is_correct else 0.0

        new_mastery = old_mastery + topic_lr * (target - old_mastery)
        new_mastery = max(0.0, min(1.0, new_mastery))
        self.state.topic_mastery[topic] = new_mastery

        self.state.attempted += 1
        self.state.answered.add(question_id)

        return {
            "question_id": question_id,
            "selected_option": selected,
            "correct_option": correct_option,
            "is_correct": is_correct,
            "difficulty": difficulty,
            "subtopic": topic,
            "ability_before": round(old_ability, 3),
            "ability_after": round(new_ability, 3),
            "topic_mastery_after": round(new_mastery, 3),
            "accuracy": round(self.state.accuracy, 3),
        }

    def next_question(self) -> pd.Series:
        return self.select_question()


def load_engine(
    csv_path: str,
    seed: int = 42,
    initial_ability: float = 3.0,
) -> AdaptiveEngine:
    bank = pd.read_csv(csv_path, encoding="utf-8-sig")
    return AdaptiveEngine(
        bank,
        seed=seed,
        initial_ability=initial_ability,
    )
