"""LLM-as-judge: faithfulness + correctness scoring.

Two orthogonal questions about a generated answer:

* **Faithfulness** — is every claim grounded in the retrieved contexts? This is
  the hallucination check, and it's judged *only* against the contexts the model
  was given (not the world, not the reference answer). An ungrounded-but-true
  statement still fails.
* **Correctness** — does the answer actually answer the question, matching the
  hand-written reference? This is judged against the gold reference answer.

Both are scored 1-5 by the same local Qwen that generates answers. Using the
generator as its own judge is cheap and fully local but introduces a self-
preference bias; the report discloses this, and the judge is a drop-in seam for
a stronger external model later. We force a low temperature for stable scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import GenerationConfig
from ..generate.llm import LLM
from ..generate.prompt import format_context
from .parsing import extract_json

if TYPE_CHECKING:
    from ..retrieve.dense import SearchHit

_FAITHFULNESS_SYSTEM = (
    "You are a strict grader checking whether an ANSWER is fully supported by the "
    "provided SOURCE EXCERPTS. Judge only against the excerpts — outside knowledge, "
    "even if true, does not count as support. Score 1-5:\n"
    "5 = every claim is directly supported by the excerpts.\n"
    "3 = partially supported; some claims lack support.\n"
    "1 = largely unsupported or contradicted by the excerpts.\n"
    'Reply with JSON only: {"score": <1-5>, "rationale": "<one sentence>"}.'
)

_CORRECTNESS_SYSTEM = (
    "You are grading whether an ANSWER correctly answers a QUESTION, using the "
    "REFERENCE ANSWER as ground truth. Reward factual agreement with the reference; "
    "ignore differences in wording or extra correct detail. Score 1-5:\n"
    "5 = fully correct and addresses the question.\n"
    "3 = partially correct or incomplete.\n"
    "1 = incorrect or off-topic.\n"
    'Reply with JSON only: {"score": <1-5>, "rationale": "<one sentence>"}.'
)


@dataclass
class Judgement:
    score: float  # 1-5; 0.0 means the judge reply couldn't be parsed
    rationale: str

    @property
    def parsed(self) -> bool:
        return self.score > 0.0


def _judge(llm: LLM, system: str, user: str) -> Judgement:
    completion = llm.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    data = extract_json(completion.text)
    if not data or "score" not in data:
        return Judgement(score=0.0, rationale=f"unparseable: {completion.text[:120]}")
    try:
        score = float(data["score"])
    except (TypeError, ValueError):
        return Judgement(score=0.0, rationale=f"non-numeric score: {data!r}")
    score = max(1.0, min(5.0, score))  # clamp to the 1-5 rubric
    return Judgement(score=score, rationale=str(data.get("rationale", "")).strip())


def _judge_config(cfg: GenerationConfig) -> GenerationConfig:
    """A copy of the generation config pinned to temperature 0 for stable grading."""
    return cfg.model_copy(update={"temperature": 0.0, "max_tokens": 200})


def judge_faithfulness(
    answer_text: str, contexts: list["SearchHit"], cfg: GenerationConfig
) -> Judgement:
    """Score how well the answer is grounded in exactly the retrieved contexts."""
    llm = LLM(_judge_config(cfg))
    user = (
        f"SOURCE EXCERPTS:\n\n{format_context(contexts)}\n\n"
        f"ANSWER:\n{answer_text}\n\n"
        "Is every claim in the answer supported by the excerpts? Grade per the rubric."
    )
    return _judge(llm, _FAITHFULNESS_SYSTEM, user)


def judge_correctness(
    question: str, answer_text: str, reference_answer: str, cfg: GenerationConfig
) -> Judgement:
    """Score the answer against the hand-written reference answer."""
    llm = LLM(_judge_config(cfg))
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference_answer}\n\n"
        f"ANSWER:\n{answer_text}\n\n"
        "Does the answer match the reference? Grade per the rubric."
    )
    return _judge(llm, _CORRECTNESS_SYSTEM, user)
