"""Canonical four-scout Delta-card training rollout for ALFWorld."""

from __future__ import annotations

import asyncio
import json
import math
import re
from copy import deepcopy
from functools import partial

from slime.utils.types import Sample

from src.alfworld.generate.delta_credit import score_cards
from src.alfworld.generate.episode import run_episode
from src.alfworld.generate.generator import generate_generator_sample
from src.alfworld.reward.functions import grpo_normalize
from src.alfworld.utils.common import ensure_metadata, require_arg


SCOUT_COUNT = 4
POST_SCOUT_COUNT = 4
EXECUTOR_GROUP_SIZE = 8

CARD_SYSTEM_PROMPT = """You are a Skill Generator for an ALFWorld reinforcement-learning agent.

The current task query was sampled from the training data buffer. Your job is
to extract one compact decision aid for that same query from an
environment-confirmed successful trajectory.

A useful skill must increase the likelihood of one demonstrated model action
when its observable trigger holds, while leaving unrelated turns unchanged.
Delta credit is computed only on model-generated reasoning/action tokens for
that action phase; environment observations are never credited. Generic task
restatements such as explore, find, take, place, or complete the task are not
decision aids.

Extract exactly one mechanism. If the trajectory contains no non-trivial,
evidence-supported mechanism, abstain instead of inventing one.

Return only one valid JSON object. Do not output analysis, Markdown fences, or
text outside the JSON object."""

LEGACY_CARD_USER_TEMPLATE = """Task: {task_description}

Environment-confirmed successful trajectory:
<<<BEGIN_SUCCESSFUL_TRAJECTORY>>>
{trajectory}
<<<END_SUCCESSFUL_TRAJECTORY>>>

Extract one reusable, non-trivial decision rule supported by the task and the
observed action-observation sequence.

Requirements:
- Focus on exactly one mechanism: observation-driven search ordering, container
  state handling, prerequisite state transition, admissible-action verification,
  goal-state verification, or recovery from a dead end.
- Express an observable trigger, the concrete action or action ordering to use,
  and the verification or stopping condition relevant to that mechanism.
- The rule must be applicable to the current task and reusable for a different
  household task whenever the same observable condition or action affordance appears.
- The applicability condition must be checkable from the task description,
  observations, inventory, and admissible actions alone. Do not claim that the
  rule is universally optimal, causally necessary, or guaranteed to succeed.
- The rule must add information beyond generic ALFWorld advice. Do not merely say
  to explore rooms, search locations, take the object, or complete the task.
  Specify what observable ambiguity, failed action, missing prerequisite, or
  state mismatch should trigger a change in action.
- Ground every condition, action, and verification step in information or
  behavior explicitly present in the supplied task and trajectory.
- Use request-independent semantic roles such as target object, target receptacle,
  required state, candidate location, inventory, observation, and admissible action.
  Do not copy concrete object or receptacle names, numbered entity identifiers,
  exact task wording, instance paths, or a room-by-room route from this episode.
- Generic affordances such as container, receptacle, state-changing appliance,
  inventory, and admissible action may be named when supported by the evidence.
- Do not invent locations, controls, object properties, or actions absent from
  the trajectory.
- Success does not prove that every action was useful, necessary, causal, or optimal.
- Do not mention the trajectory, example, evidence, reward, outcome, or success.
- Output only one valid JSON object with exactly one key named \"skill\".
- Encode all newlines inside the JSON string as \\n so json.loads can parse it.
- The skill value must be a 50-150 word Markdown card in exactly this shape:
  # Skill: <hyphenated-mechanism-name>
  1. Trigger: <the observable condition that activates this mechanism>
  2. Action: <the concrete ordered operation using ALFWorld admissible actions>
  3. Verify: <what observation, inventory state, or affordance to check>
  4. Stop: <when to abandon this branch or choose a recovery action>
- Coin a descriptive hyphenated mechanism name of at most five words; lowercase
  kebab-case is preferred. Each line must be self-contained and describe the
  same single mechanism.
- Do not output analysis, code fences, or any text outside the JSON object.

Your response must begin exactly with {{\"skill\":\"# Skill: and end with \"}}."""


CARD_USER_TEMPLATE = """Task:
{task_description}

Successful trajectory:
<<<BEGIN_TRAJECTORY>>>
{trajectory}
<<<END_TRAJECTORY>>>

Silently identify:
1. The task family and target state transition.
2. One point where an observable condition determines the next useful action.
3. The action or action ordering supported by the trajectory.
4. How to verify that the action produced the intended state.
5. The exact observable state that completes this mechanism.

Counterfactual usefulness test:
- Imagine the same buffered query is solved once without the card and once with
  the card. Emit only a rule that should make one demonstrated next action more
  likely when its trigger is visible.
- Emit only a demonstrated prerequisite or state-changing decision (open/close,
  clean, heat, cool, or toggle). Navigation, acquisition, inspection, and final
  placement are not eligible for local Delta credit.
- If the rule merely repeats the task goal or an action already forced by the
  admissible-action list, abstain.
- The card is available from the start of the episode, but it must explicitly
  remain inactive until its observable trigger holds.

Extraction rules:
- Extract exactly one decision mechanism, not the entire episode workflow.
- The mechanism must be capable of changing the agent's next action.
- Draw a hard boundary around one decision point. A sequence such as locate the
  object, take it, open the receptacle, and place it is the task workflow, not
  one mechanism, and MUST NOT be emitted. Select only the non-generic decision
  inside that sequence (for example, opening an observed closed receptacle
  before placement), or abstain if no such decision is supported.
- Emit exactly one policy item. Its single "then" value may describe only one
  ALFWorld action phase; do not encode ordered actions or successive task phases
  either inside that string or as separate policy items.
- Every condition must be observable from the task, current observation, inventory, or admissible actions.
- Every proposed action must be supported by an action or state transition appearing in the trajectory.
- stop_when MUST contain exactly one item copied verbatim from the single
  policy item's "verify" value. This makes the mechanism stop at its own
  completion state rather than at a later phase of the task.
- Use semantic roles such as target object, target receptacle, required state, candidate location, inventory, and admissible action.
- Do not copy concrete object names, numbered entity IDs, exact task wording, instance paths, or room-by-room routes.
- Names of observed objects, rooms, containers, appliances, and surfaces are
  private entity labels. Replace all of them with semantic roles, including in
  the name, summary, conditions, action, verification, and stop field.
- The single policy "then" field must contain one ALFWorld action phase only.
  Do not join navigation, object acquisition, state change, or placement into
  the same action field.
- Before emitting, silently run this rejection check over every string field:
  (1) if any concrete noun from Task or Successful trajectory remains, replace
  it with a semantic role; (2) if the name contains an underscore, rewrite it
  in kebab-case; (3) if the policy "then" contains more than one action verb,
  an "and", a comma joining actions, or a second "then", reduce it to the one
  state-changing decision phase. If any check cannot be satisfied without
  guessing, emit the abstain form instead.
- The action phase is one of: inspect/check, search/locate, take, open/close,
  clean, heat/cool, toggle, or place/put. Never combine two of these phases.
- For an emitted card, the policy "then" string MUST exactly use one of these
  single-phase templates (substitute semantic roles only; do not use concrete
  episode names):
  "inspect the target receptacle"; "go to the candidate location"; "take the
  target object from the source"; "open the target receptacle"; "close the
  target receptacle"; "clean the target object using the state-changing
  appliance"; "heat the target object using the state-changing appliance";
  "cool the target object using the state-changing appliance"; "toggle the
  target appliance"; "place the target object in the target receptacle"; or
  "place the target object on the target surface". Do not use "move", and do
  not add a second verb, an "and", a comma, or a route to any template.
- Decision gate: emit only when one of those templates is directly supported
  and every field can use semantic roles. If the trajectory needs a compound
  search-clean-place workflow, a route, a concrete entity name, or a later
  task-goal verification to explain the candidate rule, abstain. Do not try to
  salvage a multi-step draft by describing its first and last steps.
- Use this exact all-empty form when abstaining (replace only the reason):
  {{"decision":"abstain","abstain_reason":"No single reusable action phase is supported.","name":"","task_family":"","category":"","summary":"","use_when":[],"do_not_use_when":[],"policy":[],"recovery":[],"stop_when":[]}}.
- Do not copy this example's wording; it illustrates the required granularity:
  "if": "the target receptacle is closed", "then": "open the target
  receptacle", "verify": "the target receptacle is open".
- Success does not prove that every trajectory action was useful or optimal.
- Do not describe generic ALFWorld behavior such as simply exploring, taking the target object, or completing the task.
- Recovery is evidence-gated: each recovery item MUST correspond to a failed action
  or observable mismatch that actually occurs in the supplied trajectory and the
  concrete next action that recovers from it. A merely possible future failure,
  an imagined fallback, or a restatement of the main policy does not qualify.
  If that failure-then-recovery sequence is absent, "recovery" MUST be [].
- Keep the complete response concise: at most 90 English words. Prefer exactly
  one use_when item, at most one do_not_use_when item, and recovery: [] unless
  a recovery transition is directly demonstrated.
- Use "decision": "abstain" if no non-trivial reusable rule is supported.

Return exactly this schema:
{{
  "decision": "emit or abstain",
  "abstain_reason": "empty when decision is emit",
  "name": "kebab-case mechanism name, empty when abstaining",
  "task_family": "pick-place, pick-two, clean-place, heat-place, cool-place, look-at, or cross-task",
  "category": "search-order, container-state, prerequisite, affordance-check, goal-verification, or dead-end-recovery",
  "summary": "one sentence describing what the mechanism does",
  "use_when": ["observable activation condition"],
  "do_not_use_when": ["observable non-applicability condition"],
  "policy": [
    {{"if": "observable state or affordance", "then": "concrete ordered operation", "verify": "observable expected result"}}
  ],
  "recovery": [
    {{"if": "observable failure or mismatch", "then": "concrete recovery operation"}}
  ],
  "stop_when": ["exact copy of the policy verify value"]
}}

Cardinality rules:
- use_when: 1-3 items when emitting
- do_not_use_when: 0-2 items
- policy: exactly 1 conditional operation describing one decision point
- recovery: 0-2 recovery rules; use [] unless each failure and recovery operation is directly demonstrated in the trajectory
- stop_when: exactly 1 item, identical to the single policy item's verify value
- When abstaining, use empty strings or arrays for all skill-content fields."""


CARD_KEYS = {
    "decision",
    "abstain_reason",
    "name",
    "task_family",
    "category",
    "summary",
    "use_when",
    "do_not_use_when",
    "policy",
    "recovery",
    "stop_when",
}
TASK_FAMILIES = {
    "pick-place",
    "pick-two",
    "clean-place",
    "heat-place",
    "cool-place",
    "look-at",
    "cross-task",
}
CARD_CATEGORIES = {
    "search-order",
    "container-state",
    "prerequisite",
    "affordance-check",
    "goal-verification",
    "dead-end-recovery",
}

# The generator is sampled at the same temperature as the r9 baseline.  Make
# the semantic card contract executable at decoding time so exploration cannot
# turn a one-phase rule into a multi-phase action string.  The parser below
# remains responsible for episode-specific entity leakage and evidence audits.
CARD_THEN_TEMPLATES = (
    "inspect the target receptacle",
    "go to the candidate location",
    "take the target object from the source",
    "open the target receptacle",
    "close the target receptacle",
    "clean the target object using the state-changing appliance",
    "heat the target object using the state-changing appliance",
    "cool the target object using the state-changing appliance",
    "toggle the target appliance",
    "place the target object in the target receptacle",
    "place the target object on the target surface",
)

_NONEMPTY_STRING_SCHEMA = {"type": "string", "minLength": 1}
_CARD_POLICY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["if", "then", "verify"],
    "properties": {
        "if": _NONEMPTY_STRING_SCHEMA,
        "then": {"type": "string", "enum": list(CARD_THEN_TEMPLATES)},
        "verify": _NONEMPTY_STRING_SCHEMA,
    },
}
_CARD_RECOVERY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["if", "then"],
    "properties": {"if": _NONEMPTY_STRING_SCHEMA, "then": _NONEMPTY_STRING_SCHEMA},
}
_CARD_EMIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(CARD_KEYS),
    "properties": {
        "decision": {"type": "string", "enum": ["emit"]},
        "abstain_reason": {"type": "string", "enum": [""]},
        "name": _NONEMPTY_STRING_SCHEMA,
        "task_family": {"type": "string", "enum": sorted(TASK_FAMILIES)},
        "category": {"type": "string", "enum": sorted(CARD_CATEGORIES)},
        "summary": _NONEMPTY_STRING_SCHEMA,
        "use_when": {"type": "array", "minItems": 1, "maxItems": 3, "items": _NONEMPTY_STRING_SCHEMA},
        "do_not_use_when": {"type": "array", "maxItems": 2, "items": _NONEMPTY_STRING_SCHEMA},
        "policy": {"type": "array", "minItems": 1, "maxItems": 1, "items": _CARD_POLICY_SCHEMA},
        "recovery": {"type": "array", "maxItems": 2, "items": _CARD_RECOVERY_SCHEMA},
        "stop_when": {"type": "array", "minItems": 1, "maxItems": 1, "items": _NONEMPTY_STRING_SCHEMA},
    },
}
_CARD_ABSTAIN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(CARD_KEYS),
    "properties": {
        "decision": {"type": "string", "enum": ["abstain"]},
        "abstain_reason": _NONEMPTY_STRING_SCHEMA,
        "name": {"type": "string", "enum": [""]},
        "task_family": {"type": "string", "enum": [""]},
        "category": {"type": "string", "enum": [""]},
        "summary": {"type": "string", "enum": [""]},
        "use_when": {"type": "array", "maxItems": 0},
        "do_not_use_when": {"type": "array", "maxItems": 0},
        "policy": {"type": "array", "maxItems": 0},
        "recovery": {"type": "array", "maxItems": 0},
        "stop_when": {"type": "array", "maxItems": 0},
    },
}
CARD_DECODING_JSON_SCHEMA = json.dumps(
    {"oneOf": [_CARD_EMIT_SCHEMA, _CARD_ABSTAIN_SCHEMA]}, separators=(",", ":")
)

EPISODE_GENERIC_TERMS = {
    "a",
    "an",
    "and",
    "at",
    "clean",
    "cleaned",
    "cold",
    "container",
    "cool",
    "cooled",
    "find",
    "from",
    "heat",
    "heated",
    "hot",
    "in",
    "inside",
    "into",
    "inventory",
    "is",
    "it",
    "look",
    "object",
    "of",
    "on",
    "open",
    "opened",
    "pick",
    "place",
    "put",
    "receptacle",
    "required",
    "state",
    "target",
    "task",
    "the",
    "then",
    "to",
    "two",
    "with",
}

ACTION_PHASE_PATTERNS = {
    "check": r"check(?:s|ed|ing)?",
    "clean": r"clean(?:s|ed|ing)?|wash(?:es|ed|ing)?",
    "close": r"clos(?:e|es|ed|ing)",
    "cool": r"cool(?:s|ed|ing)?",
    "examine": r"examin(?:e|es|ed|ing)|inspect(?:s|ed|ing)?",
    "go": r"go|goes|going|went|navigat(?:e|es|ed|ing)|mov(?:e|es|ed|ing)",
    "heat": r"heat(?:s|ed|ing)?",
    "locate": r"locat(?:e|es|ed|ing)|find|found|search(?:es|ed|ing)?",
    "open": r"open(?:s|ed|ing)?",
    "put": r"put|puts|putting|plac(?:e|es|ed|ing)",
    "take": r"tak(?:e|es|en|ing)|pick(?:s|ed|ing)?|retriev(?:e|es|ed|ing)",
    "toggle": r"toggl(?:e|es|ed|ing)",
}


def _sampling_params_for_position(params: dict, position: int) -> dict:
    result = dict(params)
    if result.get("sampling_seed") is not None:
        result["sampling_seed"] = int(result["sampling_seed"]) + int(position)
    return result


def _is_trainable(sample: Sample) -> bool:
    return (
        isinstance(sample.tokens, list)
        and bool(sample.tokens)
        and int(sample.response_length or 0) > 0
        and isinstance(sample.loss_mask, list)
        and any(bool(value) for value in sample.loss_mask)
    )


def _parse_legacy_card(args, response_text: str) -> dict:
    del args
    text = str(response_text or "").strip()
    # SGLang's decoded Qwen response may retain the generated chat EOS token.
    # It is framing rather than assistant-authored trailing prose; remove only
    # these exact terminal markers and keep every other extra byte invalid.
    for terminal_marker in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(terminal_marker):
            text = text[: -len(terminal_marker)].rstrip()
            break
    try:
        payload = json.loads(text)
    except Exception as exc:
        return {"skill": "", "format_score": 0.0, "format_penalty": -1.0, "skill_parse_error": str(exc)}
    if not isinstance(payload, dict) or set(payload) != {"skill"}:
        return {"skill": "", "format_score": 0.0, "format_penalty": -1.0, "skill_parse_error": "expected exactly one skill key"}
    skill = str(payload.get("skill") or "").strip()
    word_count = len(re.findall(r"[A-Za-z][A-Za-z'-]*", skill))
    card_match = re.fullmatch(
        r"# Skill: ([A-Za-z0-9]+(?:-[A-Za-z0-9]+){0,4})\n"
        r"1\. Trigger: (\S.+)\n"
        r"2\. Action: (\S.+)\n"
        r"3\. Verify: (\S.+)\n"
        r"4\. Stop: (\S.+)",
        skill,
    )
    forbidden = re.compile(
        r"```|<<<|>>>|(?:^|\s)/(?:mnt|home|wl_intelligent)/|\b(?:trajectory|interaction|reward|evidence)\b",
        re.IGNORECASE,
    )
    # Keep the word-count tolerance used by the reviewed WebShop pipeline, but
    # require the card fields so a generic episode summary cannot reach Executor.
    valid = (
        20 <= word_count <= 200
        and card_match is not None
        and not forbidden.search(skill)
    )
    return {
        "skill": skill if valid else "",
        "raw_skill_output": text,
        "skill_word_count": word_count,
        "skill_mechanism_name": card_match.group(1).lower() if card_match else "",
        "skill_card_field_count": 4 if card_match else 0,
        "skill_prompt_contract": "named_trigger_action_verify_stop_50-150_words",
        "skill_parser_contract": "exact_card_fields_20-200_words",
        "format_score": 1.0 if valid else 0.0,
        "format_penalty": 0.0 if valid else -1.0,
        "skill_parse_error": "" if valid else "rule_based_card_audit_failed",
        "skill_language_audit_applied": 1.0,
        "skill_language_audit_pass": 1.0 if valid else 0.0,
    }


def _invalid_card(text: str, reason: str) -> dict:
    return {
        "skill": "",
        "raw_skill_output": text,
        "skill_decision": "invalid",
        "skill_prompt_contract": "alfworld_buffer_query_local_delta_json_v3_90_words",
        "skill_parser_contract": "strict_json_schema_contextual_leakage_audit_v3",
        "format_score": 0.0,
        "format_penalty": -1.0,
        "skill_parse_error": reason,
        "skill_language_audit_applied": 1.0,
        "skill_language_audit_pass": 0.0,
    }


def _strict_json_loads(text: str) -> dict:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    payload = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON value must be an object")
    return payload


def _single_line_string(value, *, allow_empty: bool = False) -> bool:
    return (
        isinstance(value, str)
        and "\n" not in value
        and "\r" not in value
        and (allow_empty or bool(value.strip()))
        and value == value.strip()
    )


def _string_list(payload: dict, key: str, minimum: int, maximum: int) -> bool:
    values = payload.get(key)
    return (
        isinstance(values, list)
        and minimum <= len(values) <= maximum
        and all(_single_line_string(value) for value in values)
    )


def _episode_specific_terms(task_description: str, trajectory: str) -> set[str]:
    """Return concrete episode labels that must not be copied into a card."""
    task_terms = set(re.findall(r"[a-z][a-z-]*", str(task_description).lower()))
    numbered_terms = set(
        re.findall(r"\b([a-z][a-z_-]*)\s+\d+\b", str(trajectory).lower())
    )
    return {
        term.replace("_", "-")
        for term in task_terms | numbered_terms
        if len(term) > 1 and term not in EPISODE_GENERIC_TERMS
    }


def _action_phases(text: str) -> set[str]:
    return {
        name
        for name, pattern in ACTION_PHASE_PATTERNS.items()
        if re.search(rf"\b(?:{pattern})\b", str(text), re.IGNORECASE)
    }


def _emitted_entity_terms(text: str) -> set[str]:
    """Token terms plus compact two/three-token aliases such as sink-basin."""
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    terms = set(tokens)
    for width in (2, 3):
        terms.update("".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1))
    return terms


def _render_skill_card(payload: dict) -> str:
    lines = [f"# Skill: {payload['name']}", str(payload["summary"]), "## Use When"]
    lines.extend(f"- {item}" for item in payload["use_when"])
    if payload["do_not_use_when"]:
        lines.append("## Do Not Use When")
        lines.extend(f"- {item}" for item in payload["do_not_use_when"])
    lines.append("## Policy")
    lines.extend(
        f"- IF {item['if']}, THEN {item['then']}; VERIFY {item['verify']}"
        for item in payload["policy"]
    )
    if payload["recovery"]:
        lines.append("## Recovery")
        lines.extend(
            f"- IF {item['if']}, THEN {item['then']}" for item in payload["recovery"]
        )
    lines.append("## Stop")
    lines.extend(f"- {item}" for item in payload["stop_when"])
    return "\n".join(lines)


def _parse_card(
    args,
    response_text: str,
    *,
    source_task_description: str = "",
    source_trajectory: str = "",
) -> dict:
    del args
    text = str(response_text or "").strip()
    # SGLang may preserve the generated chat EOS marker. Remove only this exact
    # framing marker; any assistant-authored prefix or suffix remains invalid.
    for terminal_marker in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(terminal_marker):
            text = text[: -len(terminal_marker)].rstrip()
            break
    try:
        payload = _strict_json_loads(text)
    except Exception as exc:
        return _invalid_card(text, str(exc))
    if set(payload) != CARD_KEYS:
        return _invalid_card(text, "top-level keys do not exactly match the V2 schema")

    decision = payload.get("decision")
    if decision not in {"emit", "abstain"}:
        return _invalid_card(text, "decision must be emit or abstain")

    content_word_count = len(
        re.findall(
            r"[A-Za-z][A-Za-z'-]*",
            " ".join(
                value
                for key in CARD_KEYS
                for value in (
                    [payload.get(key)]
                    if isinstance(payload.get(key), str)
                    else []
                )
                if isinstance(value, str)
            )
            + " "
            + " ".join(
                value
                for key in ("use_when", "do_not_use_when", "stop_when")
                for value in (payload.get(key) if isinstance(payload.get(key), list) else [])
                if isinstance(value, str)
            )
            + " "
            + " ".join(
                value
                for key in ("policy", "recovery")
                for item in (payload.get(key) if isinstance(payload.get(key), list) else [])
                if isinstance(item, dict)
                for value in item.values()
                if isinstance(value, str)
            ),
        )
    )
    if content_word_count > 180:
        return _invalid_card(text, "skill content exceeds 180 English words")

    if decision == "abstain":
        empty_fields = (
            payload.get("name") == ""
            and payload.get("task_family") == ""
            and payload.get("category") == ""
            and payload.get("summary") == ""
            and payload.get("use_when") == []
            and payload.get("do_not_use_when") == []
            and payload.get("policy") == []
            and payload.get("recovery") == []
            and payload.get("stop_when") == []
        )
        reason = payload.get("abstain_reason")
        if not _single_line_string(reason) or not empty_fields:
            return _invalid_card(text, "abstention fields violate the V2 schema")
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return {
            "skill": "",
            "raw_skill_output": text,
            "skill_structured_json": canonical,
            "skill_decision": "abstain",
            "skill_abstain_reason": reason,
            "skill_word_count": content_word_count,
            "skill_mechanism_name": "",
            "skill_card_field_count": 0,
            "skill_prompt_contract": "alfworld_buffer_query_local_delta_json_v3_90_words",
            "skill_parser_contract": "strict_json_schema_contextual_leakage_audit_v3",
            "format_score": 1.0,
            "format_penalty": 0.0,
            "skill_parse_error": "",
            "skill_language_audit_applied": 1.0,
            "skill_language_audit_pass": 1.0,
        }

    if payload.get("abstain_reason") != "":
        return _invalid_card(text, "abstain_reason must be empty when emitting")
    name = payload.get("name")
    if not isinstance(name, str) or re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", name) is None:
        return _invalid_card(text, "name must use lowercase kebab-case or underscores")
    name_was_normalized = "_" in name
    if name_was_normalized:
        payload = dict(payload)
        payload["name"] = name.replace("_", "-")
    name = payload["name"]
    if payload.get("task_family") not in TASK_FAMILIES:
        return _invalid_card(text, "unknown task_family")
    if payload.get("category") not in CARD_CATEGORIES:
        return _invalid_card(text, "unknown category")
    if not _single_line_string(payload.get("summary")):
        return _invalid_card(text, "summary must be one non-empty line")
    if not _string_list(payload, "use_when", 1, 3):
        return _invalid_card(text, "use_when must contain 1-3 non-empty strings")
    if not _string_list(payload, "do_not_use_when", 0, 2):
        return _invalid_card(text, "do_not_use_when must contain 0-2 non-empty strings")
    if not _string_list(payload, "stop_when", 1, 1):
        return _invalid_card(text, "stop_when must contain exactly one non-empty string")

    policy = payload.get("policy")
    if not isinstance(policy, list) or len(policy) != 1:
        return _invalid_card(text, "policy must contain exactly one decision operation")
    if any(
        not isinstance(item, dict)
        or set(item) != {"if", "then", "verify"}
        or not all(_single_line_string(item.get(key)) for key in ("if", "then", "verify"))
        for item in policy
    ):
        return _invalid_card(text, "policy operations do not match the exact schema")
    if payload["stop_when"] != [policy[0]["verify"]]:
        return _invalid_card(text, "stop_when must exactly match the policy verify value")
    if len(_action_phases(policy[0]["then"])) != 1:
        return _invalid_card(text, "policy then must contain exactly one ALFWorld action phase")

    recovery = payload.get("recovery")
    if not isinstance(recovery, list) or len(recovery) > 2:
        return _invalid_card(text, "recovery must contain 0-2 operations")
    if any(
        not isinstance(item, dict)
        or set(item) != {"if", "then"}
        or not all(_single_line_string(item.get(key)) for key in ("if", "then"))
        for item in recovery
    ):
        return _invalid_card(text, "recovery operations do not match the exact schema")

    emitted_text = " ".join(
        [payload["name"], payload["summary"], *payload["use_when"], *payload["do_not_use_when"], *payload["stop_when"]]
        + [value for item in policy for value in item.values()]
        + [value for item in recovery for value in item.values()]
    )
    forbidden = re.compile(
        r"```|<<<|>>>|(?:^|\s)/(?:mnt|home|wl_intelligent)/|\b(?:trajectory|interaction|reward|evidence)\b",
        re.IGNORECASE,
    )
    if forbidden.search(emitted_text):
        return _invalid_card(text, "skill content failed the leakage audit")
    episode_terms = _episode_specific_terms(source_task_description, source_trajectory)
    emitted_terms = _emitted_entity_terms(emitted_text)
    compact_episode_terms = {re.sub(r"[^a-z0-9]", "", term) for term in episode_terms}
    if episode_terms & emitted_terms or compact_episode_terms & emitted_terms:
        return _invalid_card(text, "skill content copied episode-specific entity terms")

    skill = _render_skill_card(payload)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "skill": skill,
        "raw_skill_output": text,
        "skill_structured_json": canonical,
        "skill_decision": "emit",
        "skill_abstain_reason": "",
        "skill_word_count": content_word_count,
        "skill_mechanism_name": name,
        "skill_task_family": payload["task_family"],
        "skill_category": payload["category"],
        "skill_name_normalized": 1.0 if name_was_normalized else 0.0,
        "skill_episode_entity_audit_term_count": len(episode_terms),
        "skill_card_field_count": len(policy) + len(recovery),
        "skill_prompt_contract": "alfworld_buffer_query_local_delta_json_v3_90_words",
        "skill_policy_condition": policy[0]["if"],
        "skill_policy_action": policy[0]["then"],
        "skill_policy_verify": policy[0]["verify"],
        "skill_parser_contract": "strict_json_schema_contextual_leakage_audit_v3",
        "format_score": 1.0,
        "format_penalty": 0.0,
        "skill_parse_error": "",
        "skill_language_audit_applied": 1.0,
        "skill_language_audit_pass": 1.0,
    }


async def _run_executor(args, template: Sample, params: dict, position: int, skill: str | None) -> Sample:
    result = await run_episode(
        args,
        sample=deepcopy(template),
        skill=skill,
        sampling_params=_sampling_params_for_position(params, position),
    )
    if not _is_trainable(result):
        raise RuntimeError(f"executor position {position} is not trainable")
    result.index = position
    train_metadata = dict(result.train_metadata or {})
    train_metadata.update({"training_stage": "executor", "sample_role": "executor"})
    result.train_metadata = train_metadata
    return result


async def _generate_cards(args, successful: Sample, params: dict) -> list[Sample]:
    metadata = ensure_metadata(successful)
    task_description = str(metadata.get("task_description") or "")
    trajectory = str(successful.response or "")
    messages = [
        {"role": "system", "content": CARD_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": CARD_USER_TEMPLATE.format(
                task_description=task_description,
                trajectory=trajectory,
            ),
        },
    ]
    card_params = dict(params)
    card_params["max_new_tokens"] = min(int(card_params.get("max_new_tokens", 384)), 384)
    card_params["json_schema"] = CARD_DECODING_JSON_SCHEMA
    generated = await asyncio.gather(
        *[
            generate_generator_sample(
                args,
                messages=messages,
                sampling_params=_sampling_params_for_position(card_params, index),
                n_idx=index,
                requested_mode="successful_scout_delta_card",
                actual_mode="successful_scout_delta_card",
                parser_fn=partial(
                    _parse_card,
                    source_task_description=task_description,
                    source_trajectory=trajectory,
                ),
                base_metadata={"generator_candidate_index": index, "generator_candidate_count": 4},
            )
            for index in range(SCOUT_COUNT)
        ],
        return_exceptions=True,
    )
    return [item for item in generated if isinstance(item, Sample)]


def _annotate_executor_group(samples: list[Sample], group_metadata: dict) -> None:
    for position, sample in enumerate(samples):
        metadata = ensure_metadata(sample)
        raw_reward = 10.0 * float(bool(metadata.get("won", False)))
        metadata.update(group_metadata)
        metadata.update(
            {
                "trajectory_role": "executor",
                "executor_group_position": position,
                "executor_group_size": EXECUTOR_GROUP_SIZE,
                "raw_env_reward": raw_reward,
                "raw_training_reward": raw_reward,
            }
        )
        sample.reward = raw_reward


async def run_rollout(args, sample: Sample, sampling_params: dict, evaluation: bool = False) -> list[Sample]:
    if evaluation:
        raise ValueError("Use the dedicated ALFWorld evaluator for evaluation")
    if int(require_arg(args, "n_experiences")) != 4:
        raise ValueError("Canonical Delta-card requires n_experiences=4")

    executor_params = dict(sampling_params)
    executor_params["temperature"] = float(require_arg(args, "executor_temperature"))
    if getattr(args, "executor_top_p", None) is not None:
        executor_params["top_p"] = float(args.executor_top_p)
    template = deepcopy(sample)

    try:
        scouts = list(
            await asyncio.gather(
                *[_run_executor(args, template, executor_params, index, None) for index in range(SCOUT_COUNT)]
            )
        )
    except Exception as exc:
        print(f"[alfworld/delta] dropping incomplete scout group: {exc}")
        return []

    successful_position = next(
        (index for index, item in enumerate(scouts) if bool(ensure_metadata(item).get("won", False))),
        None,
    )
    cards: list[Sample] = []
    scoring: list[dict] = []
    selected_skill = None
    selected_index = None
    selected_delta = None
    if successful_position is not None:
        cards = await _generate_cards(args, scouts[successful_position], sampling_params)
        scoring = await score_cards(args, scouts[successful_position], cards)
        eligible = []
        for card, result in zip(cards, scoring):
            metadata = ensure_metadata(card)
            delta = float(result.get("diff_reward", 0.0) or 0.0)
            valid = (
                bool(result.get("available"))
                and float(metadata.get("format_score", 0.0) or 0.0) >= 1.0
                and bool(metadata.get("skill"))
                and math.isfinite(delta)
            )
            reward = max(0.0, delta) if valid else 0.0
            card.reward = reward
            metadata.update(
                {
                    "method": "delta_card",
                    "trajectory_role": "generator",
                    "skill_diff_available": 1.0 if result.get("available") else 0.0,
                    "skill_diff_skip_reason": str(result.get("skip_reason") or ""),
                    "skill_diff_base_mean_logprob": result.get("base_mean_logprob"),
                    "skill_diff_with_skill_mean_logprob": result.get("skill_mean_logprob"),
                    "skill_diff_reward": delta,
                    "skill_diff_scored_token_count": int(result.get("scored_token_count", 0) or 0),
                    "skill_diff_scored_turn_count": int(result.get("scored_turn_count", 0) or 0),
                    "skill_diff_mask_source": str(result.get("mask_source") or ""),
                    "skill_diff_credit_scope": str(result.get("credit_scope") or ""),
                    "skill_diff_policy_action_phase": str(result.get("policy_action_phase") or ""),
                    "skill_diff_model_token_count": int(result.get("model_token_count", 0) or 0),
                    "skill_diff_environment_token_count": int(result.get("environment_token_count", 0) or 0),
                    "skill_diff_scored_environment_token_count": int(
                        result.get("scored_environment_token_count", 0) or 0
                    ),
                    "skill_diff_loss_mask_mismatch_count": int(result.get("loss_mask_mismatch_count", 0) or 0),
                    "generator_reward_after_null_clip": reward,
                    "raw_training_reward": reward,
                }
            )
            if valid and delta > 0:
                eligible.append((delta, int(card.index), str(metadata["skill"])))
        if eligible:
            selected_delta, selected_index, selected_skill = max(eligible, key=lambda x: (x[0], -x[1]))

    skill_mode = "positive_delta_skill" if selected_skill else "no_skill"
    try:
        post = list(
            await asyncio.gather(
                *[
                    _run_executor(args, template, executor_params, position, selected_skill)
                    for position in range(SCOUT_COUNT, EXECUTOR_GROUP_SIZE)
                ]
            )
        )
    except Exception as exc:
        print(f"[alfworld/delta] dropping incomplete post-scout group: {exc}")
        return []

    executors = scouts + post
    group_metadata = {
        "method": "delta_card",
        "rollout_variant": "four_scout_buffer_query_local_action_delta_alfworld_v3",
        "source_query_mode": "rollout_data_buffer",
        "initial_scout_success_count": sum(bool(ensure_metadata(x).get("won", False)) for x in scouts),
        "selected_success_scout_position": successful_position,
        "generator_candidate_count": len(cards),
        "generator_positive_delta_count": sum(float(x.get("diff_reward", 0.0) or 0.0) > 0 for x in scoring),
        "selected_skill_injected": 1.0 if selected_skill else 0.0,
        "selected_skill_candidate_index": selected_index,
        "selected_skill_delta": selected_delta,
        "selected_skill": str(selected_skill or ""),
        "executor_skill_mode": skill_mode,
    }
    _annotate_executor_group(executors, group_metadata)
    grpo_normalize(executors)

    trainable_cards: list[Sample] = []
    if len(cards) == 4 and all(_is_trainable(card) for card in cards):
        source_group = sample.group_index if isinstance(sample.group_index, int) else 0
        for position, card in enumerate(sorted(cards, key=lambda x: int(x.index))):
            card.group_index = -(int(source_group) + 1)
            card.index = position
            train_metadata = dict(card.train_metadata or {})
            train_metadata.update({"training_stage": "generator", "sample_role": "generator"})
            card.train_metadata = train_metadata
        trainable_cards = sorted(cards, key=lambda x: int(x.index))
        grpo_normalize(trainable_cards)
        weight = float(require_arg(args, "generator_reward_weight"))
        for card in trainable_cards:
            card.reward = float(card.reward or 0.0) * weight
            ensure_metadata(card)["reward_weight_applied"] = weight

    return trainable_cards + executors
