"""Prompt templates for the WebShop Experience Generator rollout pipeline.

Three roles share the same underlying model weights:
  1. Executor (no experience)  – plays WebShop episode on source task (Step 1)
  2. Generator               – extracts one short policy-facing skill from trajectory (Step 2)
  3. Executor (with skill)     – plays downstream WebShop tasks using a retrieved skill (Step 4)
"""

import json
import os
import re

# ---------------------------------------------------------------------------
# 1. Executor prompt (no experience)  ← Step 1
# ---------------------------------------------------------------------------

EXECUTOR_SYSTEM_PROMPT = (
    "You are an intelligent shopping agent in the WebShop text environment. "
    "Your goal is to buy one product that satisfies every requirement in the user's instruction.\n\n"
    "At each step, you will be given the current observation and a list of admissible actions. "
    "You MUST first reason step-by-step about the current situation inside <reasoning> and </reasoning> tags. "
    "Then choose exactly ONE admissible action and output it inside <action> and </action> tags.\n\n"
    "For a search, replace the placeholder in search[<keywords>] with concise keywords. "
    "Every click action MUST be copied verbatim from the admissible actions list. "
    "Use the observation returned after every action as feedback. Compare title, attributes, options, and price "
    "against the instruction before clicking Buy Now; select all required options first.\n\n"
    "Example output format:\n"
    "<reasoning>\n"
    "The user needs a waterproof hiking shoe under the stated budget. "
    "I should search with the product type and the most discriminative attribute.\n"
    "</reasoning>\n"
    "<action>search[waterproof hiking shoe]</action>"
)



def format_executor_messages(task_description: str) -> list[dict]:
    """Chat messages for the Executor without any experience context (Step 1).

    The initial observation and admissible commands are NOT included here —
    they are prepended as the first observation in the episode response.

    Args:
        task_description: The raw task description (e.g. "put a fork in a drawer").

    Returns:
        A list of chat message dicts compatible with tokenizer.apply_chat_template().
    """
    return [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Your task is to: {task_description}"},
    ]


# ---------------------------------------------------------------------------
# 2. Generator prompt  ← Step 2
# ---------------------------------------------------------------------------

SUCCESSFUL_SCOUT_GENERATOR_SYSTEM_PROMPT = """You are the Skill Generator in a reinforcement-learning WebShop agent.
Given one trajectory that the environment confirms succeeded, extract exactly
one reusable policy skill grounded only in the supplied evidence. Success does
not imply that every action in the trajectory was useful or optimal."""


SUCCESSFUL_SCOUT_GENERATOR_USER_PROMPT_TEMPLATE = """Task:
{task_description}

Environment-confirmed successful trajectory:
<<<BEGIN_SUCCESSFUL_TRAJECTORY>>>
{trajectory}
<<<END_SUCCESSFUL_TRAJECTORY>>>

Extract one reusable, non-trivial decision rule supported by the task and the
observed action-observation sequence.

Requirements:
- Focus on exactly one mechanism: search formulation, candidate screening,
  hard-constraint verification, option/price verification, or recovery from a
  dead end.
- Express an observable trigger, the concrete action or action ordering to use,
  and the verification or stopping condition relevant to that mechanism.
- The rule must be applicable to the current task and reusable for a different
  task whenever the same observable condition or page affordance appears.
- The applicability condition must be checkable from the task description and
  page observations alone. Do not claim that the rule is universally optimal,
  causally necessary, or guaranteed to succeed.
- The rule must add information beyond generic WebShop advice. Do not merely say
  to use relevant keywords, inspect candidates, compare products, or verify all
  requirements before buying. Specify what observable ambiguity, mismatch, or
  missing evidence should trigger a change in action.
- Ground every condition, action, and verification step in information or
  behavior explicitly present in the supplied task and trajectory.
- Attribute types such as size, color, material, compatibility, and price may be
  mentioned when relevant, but do not copy product IDs, brands, exact prices,
  exact sizes, exact colors, or task-specific query phrases. Do not paraphrase
  this task's particular attribute values.
- Do not invent filters, controls, product facts, or actions absent from the
  trajectory.
- Success does not prove that every action was useful, necessary, causal, or
  optimal.
- Do not mention “the trajectory”, “the example”, or the reward.
- Output only one JSON object with exactly one key:
  {json_schema}
- The skill must contain 2–4 action-oriented sentences and 50–150 English words.
- Do not output Markdown, analysis, headings, lists, or code fences."""


GENERAL_CARD_GENERATOR_USER_PROMPT_TEMPLATE = """You are the Skill Generator in a reinforcement-learning WebShop agent. Extract the single most reusable decision mechanism supported by the private successful evidence below.

Private task: {task_description}
Private trajectory: {trajectory}

STRICT SERIALIZATION CONTRACT — violation makes the entire output invalid:
1. The first character of your response must be `{{` and the last character must be `}}`.
2. Emit exactly one valid JSON object with exactly one key named `skill`.
3. Never emit backticks, a code fence, analysis, preface, or trailing text.
4. Encode newlines inside the JSON string normally; ensure `json.loads` can parse the complete response.

The `skill` value is a 50-150 word Markdown card with headings in this exact order: `# Skill:`, `## When to Use`, `## Inputs to Identify`, `## Workflow`, `## Decision Rules`. Use no more than six complete sentences; list fragments are allowed.

Choose exactly one mechanism name: `query-reformulation-after-weak-retrieval`, `explicit-contradiction-screening`, `page-evidence-constraint-verification`, `required-option-verification-before-purchase`, `compatibility-evidence-verification`, or `budget-and-constraint-gating`.

Use only these semantic role labels for request-specific concepts: product type, hard constraint, contradictory evidence, required option, compatibility requirement, page evidence, budget, candidate, search result. Keep them verbatim and never add a concrete example or subtype. Invalid: `device compatibility`, `device family`, `color or size`, `fragrance`, `power rating`, `running use`. Valid replacements: `compatibility requirement`, `required option`, or `hard constraint`. Do not output any product/category name, brand, identifier, query phrase, number, currency amount, size, color, material, audience, device, power rating, scent, skin property, sport/use case, or requested attribute. Do not mention private evidence, task, source, trajectory, example, outcome, reward, or success.

Describe only the selected mechanism: its observable trigger, ordered action, and verification or stopping condition. Do not summarize the full episode, invent controls or facts, or claim universal optimality. Silently verify that the card would remain valid for a different product category without editing.

Your response must begin exactly like this (with the selected name replacing the placeholder):
{{"skill":"# Skill: mechanism-name\\n"""


def format_successful_scout_generator_messages(
    task_description: str,
    trajectory: str,
) -> list[dict]:
    """Build the reviewed Feature-2 prompt without candidate-specific text."""

    successful_trajectory = str(trajectory or "").strip()
    if not successful_trajectory:
        raise ValueError("Successful scout extraction requires a non-empty trajectory")
    prompt_variant = os.environ.get("DELTA_GENERATOR_PROMPT_VARIANT", "default").strip()
    if prompt_variant == "card_general_v1":
        user_prompt = GENERAL_CARD_GENERATOR_USER_PROMPT_TEMPLATE.format(
            task_description=str(task_description or "").strip(),
            trajectory=successful_trajectory,
        )
    elif prompt_variant == "default":
        user_prompt = SUCCESSFUL_SCOUT_GENERATOR_USER_PROMPT_TEMPLATE.format(
            task_description=str(task_description or "").strip(),
            trajectory=successful_trajectory,
            json_schema='{"skill": "one reusable WebShop skill"}',
        )
    else:
        raise ValueError(f"Unknown DELTA_GENERATOR_PROMPT_VARIANT={prompt_variant!r}")
    return [
        {"role": "system", "content": SUCCESSFUL_SCOUT_GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

SIMPLE_SKILL_OUTPUT_PROMPT = """Return ONLY one valid JSON object using exactly this schema:

{"skill": "one transferable policy-facing WebShop skill"}

Requirements:
- The object must contain exactly one key: "skill".
- The skill must be exactly two action-oriented sentences totaling 60-80 English words.
- Sentence 1 must state a reusable trigger, decision rule or action ordering, and verification step.
- Sentence 2 must state a safe recovery action for missing or contradictory evidence.
- Evidence has priority over plausibility: use only facts explicitly present in the task, observations,
  admissible actions, model actions, or environment outcome fields.
- High entropy means the model was uncertain; it does not prove correctness, error, importance, or causation.
- Abstract away product IDs, brands, exact prices, sizes, colors, and copied query phrases.
- Use only actions supported by the supplied WebShop evidence. Never invent filters, sorting controls,
  sliders, carts, shipping rules, tabs, or product facts that are not shown.
- Never relax a hard user constraint or recommend the nearest partial match.
- Do not output analysis, Markdown, headings, lists, code fences, or any text outside the JSON object.
"""


def format_generator_messages(
    task_description: str, trajectory: str, won_score: float
) -> list[dict]:
    """Chat messages for the Generator agent (Step 2).

    Args:
        task_description: The shopping instruction the Executor attempted.
        trajectory: The full episode transcript (obs + reasoning + action per step).
        won_score: 1.0 if the episode succeeded, 0.0 if failed.

    Returns:
        A list of chat message dicts compatible with tokenizer.apply_chat_template().
    """

    success = won_score >= 0.9
    generator_prompt = (
        f"Task: {task_description}\n\n"
        "Agent's Trajectory:\n"
        "<<<BEGIN_TRAJECTORY>>>\n"
        f"{trajectory}\n"
        "<<<END_TRAJECTORY>>>\n\n"
        f"True Environment Success: {success}\n\n"
        f"{SIMPLE_SKILL_OUTPUT_PROMPT}"
    )
    return [
        {"role": "user", "content": generator_prompt},
    ]


def _format_high_entropy_turn_evidence(
    high_entropy_turns: list[dict],
) -> str:
    """Serialize only the source-turn evidence selected for the generator."""

    evidence = []
    for entropy_rank, turn in enumerate(high_entropy_turns, start=1):
        evidence.append(
            {
                "entropy_rank": entropy_rank,
                "turn_index": int(turn["turn_index"]),
                "turn_entropy": float(turn["turn_entropy"]),
                "topk_probability_mass_mean": turn.get("topk_probability_mass_mean"),
                "observation": str(turn.get("observation") or ""),
                "admissible_actions": list(turn.get("admissible_actions") or []),
                "model_response": str(turn.get("response_text") or ""),
                "parsed_action": str(turn.get("action") or ""),
                "model_action_valid": turn.get("model_action_valid"),
                "environment_action_valid": turn.get("environment_action_valid"),
                "done_after_turn": turn.get("done_after_turn"),
                "won_after_turn": turn.get("won_after_turn"),
                "task_score_after_turn": turn.get("task_score_after_turn"),
            }
        )
    return json.dumps(evidence, ensure_ascii=False, indent=2)


def format_outcome_conditioned_generator_messages(
    task_description: str,
    won_score: float,
    *,
    full_trajectory: str | None,
    high_entropy_turns: list[dict],
    candidate_index: int,
    candidate_count: int,
) -> list[dict]:
    """Use a full successful trajectory or only high-entropy failed turns."""

    if candidate_count <= 0 or not 0 <= candidate_index < candidate_count:
        raise ValueError(
            f"invalid candidate position: index={candidate_index}, count={candidate_count}"
        )

    success = won_score >= 0.9
    if success:
        trajectory = str(full_trajectory or "").strip()
        if not trajectory:
            raise ValueError("Successful source episode requires its full trajectory")
        evidence_mode = "FULL_SUCCESS_TRAJECTORY"
        evidence_block = (
            "Full Successful Scout Trajectory:\n"
            "<<<BEGIN_FULL_TRAJECTORY>>>\n"
            f"{trajectory}\n"
            "<<<END_FULL_TRAJECTORY>>>"
        )
        outcome_instruction = (
            "The scout succeeded, so the complete trajectory is provided. Trace backward "
            "from completion and extract one evidence-supported reusable decision rule or "
            "action ordering. Success does not prove that every action was useful or optimal, "
            "and selecting the first result once does not justify an always-select-first rule."
        )
    else:
        if not high_entropy_turns:
            raise ValueError(
                "Failed source episode requires at least one high-entropy response turn"
            )
        evidence_mode = "HIGH_ENTROPY_FAILURE_TURNS"
        evidence = _format_high_entropy_turn_evidence(high_entropy_turns)
        evidence_block = (
            "High-Entropy Turns From Failed Scout (highest entropy first):\n"
            "<<<BEGIN_HIGH_ENTROPY_TURNS>>>\n"
            f"{evidence}\n"
            "<<<END_HIGH_ENTROPY_TURNS>>>"
        )
        outcome_instruction = (
            "The scout failed, so only its most uncertain response turns are provided. "
            "Call something a mistake only when the evidence explicitly shows an invalid "
            "action, unmet hard constraint, product mismatch, or premature purchase. If the "
            "failure cause is not observable, give a conservative verification and recovery "
            "policy instead of guessing what happened in omitted turns."
        )

    generator_prompt = (
        f"Task: {task_description}\n\n"
        f"Scout Outcome: {'SUCCESS' if success else 'FAILURE'}\n"
        f"Generator Evidence Mode: {evidence_mode}\n\n"
        f"{outcome_instruction}\n\n"
        f"{evidence_block}\n\n"
        "Produce exactly ONE reusable skill, not a list of skills. Choose exactly one "
        "evidence-supported mechanism: search/refinement, candidate screening, hard-constraint "
        "or hidden-specification verification, variant/price verification, or recovery from "
        "a contradiction or dead end. Do not combine unrelated advice merely to fill space.\n\n"
        f"{SIMPLE_SKILL_OUTPUT_PROMPT}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the Generator role in a reinforcement-learning agent. "
                "Convert outcome-conditioned executor evidence into one evidence-grounded, "
                "reusable WebShop skill."
            ),
        },
        {"role": "user", "content": generator_prompt},
    ]


# ---------------------------------------------------------------------------
# Short skill JSON parsing
# ---------------------------------------------------------------------------

_TRAILING_MODEL_TERMINATORS = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "</s>",
)


def _strip_trailing_model_terminators(text: str) -> str:
    """Remove only known model terminators that follow the JSON response."""

    source = text.strip()
    while True:
        for terminator in _TRAILING_MODEL_TERMINATORS:
            if source.endswith(terminator):
                source = source[: -len(terminator)].rstrip()
                break
        else:
            return source


def _extract_skill_object(text: str) -> tuple[dict | None, str]:
    """Parse a response that must consist only of one JSON object."""

    source = _strip_trailing_model_terminators(str(text or ""))
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return None, "Skill output must be one JSON object"
    return payload, ""


def skill_json_to_text(
    json_str: str,
    *,
    min_words: int = 20,
    max_words: int = 100,
    min_sentences: int = 1,
    max_sentences: int = 2,
) -> tuple[str | None, str, int]:
    """Parse strict skill JSON with a wider safety range than the prompt target."""

    payload, error = _extract_skill_object(json_str)
    if payload is None:
        return None, error, 0
    if set(payload) != {"skill"}:
        return None, "Skill JSON must contain exactly one key: skill", 0

    skill_text = str(payload.get("skill") or "").strip()
    word_count = len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9'-]*\b", skill_text))
    if not skill_text:
        return None, "Skill text is empty", 0
    if not min_words <= word_count <= max_words:
        return (
            None,
            f"Skill must contain {min_words}-{max_words} words, got {word_count}",
            word_count,
        )

    # Count only punctuation that behaves like an actual sentence boundary.
    # This keeps decimal points (``6.6``) and abbreviation points followed by
    # punctuation/lowercase text (``e.g., phone``) inside the same sentence.
    # The generator is asked to begin each action sentence with a capitalized
    # word, so a terminator followed by whitespace + uppercase/digit is the
    # useful boundary here; final punctuation is also a boundary.
    sentences = [
        part
        for part in re.split(
            r"(?:[!?]+|(?<!\d)\.(?!\d))"
            r"(?=(?:[\"')\]]*)?(?:\s+[A-Z0-9]|\s*$))",
            skill_text,
        )
        if part.strip()
    ]
    if not min_sentences <= len(sentences) <= max_sentences:
        return (
            None,
            "Skill must contain "
            f"{min_sentences}-{max_sentences} sentences, got {len(sentences)}",
            word_count,
        )
    return skill_text, "", word_count



# ---------------------------------------------------------------------------
# 3. Executor prompt (with skill)  ← Step 4
# ---------------------------------------------------------------------------

EXECUTOR_WITH_SKILL_SYSTEM_PROMPT_TEMPLATE = (
    "You are an intelligent shopping agent in the WebShop text environment. "
    "Your goal is to buy one product that satisfies every requirement in the user's instruction.\n\n"
    "At each step, you will be given the current observation and a list of admissible actions. "
    "You MUST first reason step-by-step about the current situation inside <reasoning> and </reasoning> tags. "
    "Then choose exactly ONE admissible action and output it inside <action> and </action> tags.\n\n"
    "For a search, replace the placeholder in search[<keywords>] with concise keywords. "
    "Every click action MUST be copied verbatim from the admissible actions list. "
    "Use the observation returned after every action as feedback. Compare title, attributes, options, and price "
    "against the instruction before clicking Buy Now; select all required options first.\n\n"
    "Example output format:\n"
    "<reasoning>\n"
    "The user needs a waterproof hiking shoe under the stated budget. "
    "I should search with the product type and the most discriminative attribute.\n"
    "</reasoning>\n"
    "<action>search[waterproof hiking shoe]</action>\n\n"
    "A candidate reusable skill from another trajectory is provided below. Silently decide whether "
    "its decision rule is relevant before each action. Apply it only when relevant; otherwise ignore it. "
    "The current user instruction, observations, and admissible actions always take priority. Treat the "
    "skill as optional strategy guidance, never as evidence that a product satisfies a requirement.\n\n"
    "Reference Skill:\n{skill}"
)




def format_executor_with_skill_messages(
    task_description: str, skill: str
) -> list[dict]:
    """Chat messages for the Executor conditioned on a retrieved skill (Step 4).

    Args:
        task_description: The downstream task the Executor should attempt.
        skill: One short policy-facing skill extracted by the Generator.

    Returns:
        A list of chat message dicts compatible with tokenizer.apply_chat_template().
    """
    system = EXECUTOR_WITH_SKILL_SYSTEM_PROMPT_TEMPLATE.format(skill=skill)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Your task is to: {task_description}"},
    ]
