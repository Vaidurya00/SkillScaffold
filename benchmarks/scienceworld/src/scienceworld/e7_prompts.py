"""ScienceWorld E7 generator contract; examples are authored, not held-out data."""

GENERATOR_SYSTEM_PROMPT = """You extract reusable ScienceWorld laboratory skills from four attempts at the same task. Learn observable decision rules from both successful and unsuccessful attempts. Use only supplied evidence. A failed or budget-limited attempt may contain valid partial progress; a score increase alone does not prove task completion. Never invent instruments, commands, causal effects, object properties, or experimental results. Output only one <skill>...</skill> card, at most 400 English words."""


def select_example_key(labels):
    return "mixed" if any(labels) and not all(labels) else "success" if all(labels) else "failure"


def build_generator_user_prompt(task, trajectories, example_keys=()):
    return f"""Task: {task}

Four observed attempts, with environment-confirmed outcomes:
{trajectories}

Produce a reusable procedural card, not a transcript or a solution tied to one variation.
Preserve verified progress and identify failure-producing decisions. Bind commands to current
observable objects and supported action templates. Distinguish measured facts from hypotheses;
check equipment, material state and prerequisites before interpreting a failed experiment.
Include observable verification, recovery and stopping criteria. Budget exhaustion is not proof
of absence or impossibility. Never invent a successful procedure when every attempt failed.
Abstract exact object names, room names and variation identifiers into roles, while retaining
scientific properties and command structure supported by the attempts. Do not mention reward,
Delta, trajectories or demonstrations inside the card.

Illustrative writing rules (authored examples, not facts about the current task):
- Circuit task: if the observations establish an incomplete circuit, check observed connections
  before inferring that a tested material is nonconductive; require the task's observable criterion.
- State-change task: separate applying a treatment from observing the requested state change;
  use only measurement or inspection commands supported by the supplied attempts.
- Failure-only task: describe the observed obstacle, a supported diagnostic check and the
  conditions required before retrying; do not fabricate a confirmed solution.

Return exactly this structure, filling each section with evidence-supported guidance:
<skill>
# Skill: short reusable mechanism name
## When to apply
Observable trigger and prerequisites.
## Success-preserving strategy
Confirmed progress and facts to preserve.
## Failure patterns to suppress
Observed errors, invalid assumptions or unproductive repetitions.
## Procedure
Ordered actions with prerequisites; no fixed private action transcript.
## Decision rules
Observable conditions and corresponding actions.
## Verification and recovery
How to check the result and recover from the observed failure modes.
## Stop conditions
Observable completion or justified stopping conditions.
</skill>"""
