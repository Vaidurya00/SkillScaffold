"""Reviewed ScienceWorld card prompt: preserve uncertainty and parameterize labels."""
from src.scienceworld.e7_prompts import (
    GENERATOR_SYSTEM_PROMPT as BASE_SYSTEM,
    build_generator_user_prompt as base_prompt,
    select_example_key,
)

GENERATOR_SYSTEM_PROMPT = BASE_SYSTEM + """
CRITICAL: An absent signal is not a negative material test until the observed test setup is
verified. Missing an object, a broken circuit, an invalid command, and budget exhaustion are
different failures. Do not convert any of them into a scientific conclusion. Never prescribe
putting an untested object into a classification box. Replace private colors, room names,
material-instance names and exact destinations with roles read from the current task.
Prefer a narrow, evidence-supported recovery skill over an invented complete solution.
Do not introduce heat, movement or other indicators unless the observed experiment used them.
Do not assume a specific instrument, wire, power source or control is available.
Cards naming a fixed box, bin or container color will fail validation. Use
"the destination specified by the current task" instead. Do not copy a color
even when it occurs in every supplied attempt.
"""

EXAMPLE = """<skill>
# Skill: Verify a test setup before classifying a material
## When to apply
A conductivity task provides an observable test setup, but an attempted test yields no clear signal.
## Success-preserving strategy
Keep confirmed object identity, accessible equipment and observed connections; bind destination roles from the current task.
## Failure patterns to suppress
Treating a missing signal as a material property before checking the setup, or classifying an object that has not been tested.
## Procedure
1. Locate the target using observed names and supported commands.
2. Inspect the available setup and correct only connection or activation problems evidenced by feedback.
3. Retest using the supported procedure and observe its stated indicator.
4. Classify only when the task's criterion is met; use its specified destination mapping.
## Decision rules
If the setup is unverified, withhold classification. If the command is rejected, inspect its target and prerequisites before retrying. If equipment is unavailable, do not invent substitutes.
## Verification and recovery
Confirm that each correction changes the intended state. Use an observed control only if available and supported by the task; otherwise retain uncertainty about a missing signal.
## Stop conditions
Stop on environment-confirmed completion. At the action limit, leave the conclusion unresolved rather than labeling an untested object.
</skill>"""


def build_generator_user_prompt(task, trajectories, example_keys=()):
    return base_prompt(task, trajectories, example_keys) + "\n\n" + (
        "The following authored example illustrates conditional wording and abstraction only. "
        "It is not evidence about this task. Do not copy its equipment or procedure unless "
        "supported by the four attempts. Your final card must use this serialization and "
        "the evidence restrictions above.\n" + EXAMPLE +
        "\nNow output only your evidence-grounded card. No fixed box colors or room names."
    )
