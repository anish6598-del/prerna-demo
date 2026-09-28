from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import random
import pandas as pd


@dataclass
class StudentState:
    """State maintained by the baseline adaptive engine."""

    ability: float = 3.0
    topic_mastery: Dict[str, float] = field(default_factory=dict)
    asked: List[int] = field(default_factory=list)
    correct: int = 0
    attempted: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempted if self.attempted else 0.0


class AdaptiveEngine:
    """
    Baseline adaptive engine for Prerna Classes Q27-Q95.

    Difficulty is on a 1-5 scale. The engine is intentionally simple and
    interpretable for the demo:
      1. Prefer questions close to current ability.
      2. Give extra weight to weak subtopics.
      3. Avoid immediate/repeated questions.
      4. Update ability and subtopic mastery after every response.

    This is a heuristic baseline, not an IRT model.
    """

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

        self.bank = question_bank.copy().reset_index(drop=True)
        self.state = StudentState(ability=initial_ability)
        self.rng = random.Random(seed)

        for topic in self.bank["subtopic"].unique():
            self.state.topic_mastery[topic] = 0.50

    def _topic_mastery(self, topic: str) -> float:
        return self.state.topic_mastery.get(topic, 0.50)

    def _question_score(self, row: pd.Series) -> float:
        difficulty = float(row["difficulty"])
        topic = row["subtopic"]
        mastery = self._topic_mastery(topic)

        # Main signal: match question difficulty to current ability.
        difficulty_match = 1.0 - min(abs(difficulty - self.state.ability) / 4.0, 1.0)

        # Weak-topic priority.
        weakness = 1.0 - mastery

        # Small exploration term prevents the same narrow pattern.
        exploration = self.rng.random() * 0.08

        # Slightly prefer unseen questions.
        novelty = 0.18 if int(row["question_id"]) not in self.state.asked else -0.30

        return (
            0.62 * difficulty_match
            + 0.30 * weakness
            + novelty
            + exploration
        )

    def select_question(self) -> pd.Series:
        available = self.bank[
            ~self.bank["question_id"].isin(self.state.asked)
        ].copy()

        # If all questions have been seen, allow reuse, but avoid the most
        # recently attempted question.
        if available.empty:
            recent = set(self.state.asked[-3:])
            available = self.bank[~self.bank["question_id"].isin(recent)].copy()

        available["_score"] = available.apply(self._question_score, axis=1)
        best = available.sort_values(
            ["_score", "difficulty"], ascending=[False, True]
        ).iloc[0]

        qid = int(best["question_id"])
        if not self.state.asked or self.state.asked[-1] != qid:
            self.state.asked.append(qid)

        return best.drop(labels=["_score"], errors="ignore")

    def answer(
        self,
        question_id: int,
        selected_option: str,
    ) -> Dict[str, object]:
        rows = self.bank[self.bank["question_id"] == question_id]
        if rows.empty:
            raise ValueError(f"Unknown question_id: {question_id}")

        row = rows.iloc[0]
        selected = selected_option.strip().upper()
        correct_option = str(row["correct_option"]).strip().upper()
        is_correct = selected == correct_option

        difficulty = float(row["difficulty"])
        topic = str(row["subtopic"])

        # Adaptive ability update:
        # harder correct answers give a larger positive update;
        # harder incorrect answers give a larger negative update.
        gap = difficulty - self.state.ability
        learning_rate = 0.22

        if is_correct:
            delta = learning_rate * (0.75 + 0.35 * max(gap, 0.0))
            self.state.correct += 1
        else:
            delta = -learning_rate * (0.75 + 0.25 * max(-gap, 0.0))

        self.state.ability = max(1.0, min(5.0, self.state.ability + delta))

        # Topic mastery update.
        old_mastery = self._topic_mastery(topic)
        topic_lr = 0.18 if is_correct else 0.22
        target = 1.0 if is_correct else 0.0
        new_mastery = old_mastery + topic_lr * (target - old_mastery)
        self.state.topic_mastery[topic] = max(0.0, min(1.0, new_mastery))

        self.state.attempted += 1

        return {
            "question_id": question_id,
            "selected_option": selected,
            "correct_option": correct_option,
            "is_correct": is_correct,
            "difficulty": difficulty,
            "subtopic": topic,
            "ability_before": round(
                self.state.ability - delta, 3
            ),
            "ability_after": round(self.state.ability, 3),
            "topic_mastery_after": round(new_mastery, 3),
            "accuracy": round(self.state.accuracy, 3),
        }

    def next_question(self) -> pd.Series:
        return self.select_question()


def load_engine(csv_path: str, seed: int = 42, initial_ability: float = 3.0):
    bank = pd.read_csv(csv_path)
    return AdaptiveEngine(bank, seed=seed, initial_ability=initial_ability)
