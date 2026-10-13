"""Domain-aware four-scout generator prompts for Tau2 E7."""

from __future__ import annotations

from typing import Iterable


COMMON_SYSTEM = """You are a strategy generator for a tool-using customer-service agent.
You receive the exact agent-visible context and four independent attempts for one task.
Infer one reusable strategy that preserves mechanisms shared by good attempts and suppresses
mechanisms associated with bad attempts. The domain policy and tool schemas always override
the strategy. Never reveal or copy task-specific entities. Return only the requested Markdown
skill card and nothing else."""


DOMAIN_GUIDANCE = {
    "airline": """AIRLINE FOCUS
- Distinguish read-only lookup from booking mutations, and obtain explicit confirmation before writes.
- Verify identity, reservation state, itinerary, eligibility, availability, fare/refund arithmetic,
  payment method, baggage, and policy time windows in the order actually required.
- State denial or escalation conditions instead of inventing unsupported exceptions.
- Keep reservation IDs, flight numbers, dates, names, prices, routes, and payment IDs abstract.""",
    "retail": """RETAIL FOCUS
- Distinguish order/item lookup from return, exchange, refund, address, payment, and cancellation writes.
- Verify identity, order status, item eligibility, delivery state, quantities, refund destination,
  replacement availability, and confirmation boundaries before irreversible actions.
- Handle partial orders and multi-item requests explicitly; stop or escalate when policy forbids an action.
- Keep order IDs, product IDs, names, addresses, dates, amounts, and payment IDs abstract.
- Do not name a product category or repeat a user's subjective product preference; say `item` and
  `requested variant` so the mechanism transfers across products.""",
    "telecom": """TELECOM FOCUS
- Separate tools the agent may call from diagnostic actions that only the user can perform.
- TELECOM FORMAT REQUIREMENT: the `## Procedure` section must contain exactly these two bullet
  roles: `- Agent:` for one account/tool check, then `- Ask user:` for exactly one device action.
  The `Ask user:` bullet must explicitly say to wait for the reported result before branching.
- Put later diagnostic alternatives under `## Decision rules` as branches conditioned on that
  reported result. Never present multiple device changes as a batch checklist.
- Verify account/device/service state before writes; include recovery, escalation, and stop conditions.
- Never claim that a user-side action happened unless the user reported it.
- Keep account IDs, phone numbers, device identifiers, names, countries/locations, addresses, and
  plan values abstract. Stop conditions must be failures or authority limits, not a success checklist.""",
}


OUTPUT_CONTRACT = """OUTPUT CONTRACT
Return the skill card directly as Markdown, beginning with `# Skill:` and containing these
sections in order: `## When to apply`, `## Policy and tool checks`, `## Procedure`,
`## Decision rules`, `## Verification and recovery`, and `## Stop or escalation conditions`.
Do not wrap it in JSON or a code fence.

The skill should be 120-240 English words, with no more than two concise sentences or bullets
per section. It must be operational rather than generic and reusable on a different query in
the same domain. Stable tool names and argument roles are allowed. Do not
copy any customer name, account/order/reservation/payment identifier, email, phone number,
address, route, flight number, date, price, exact task value, or hidden evaluator language.
Do not compare or refer to the supplied examples, their outcome labels, trajectories, rewards,
Delta, demonstrations, or grading.
Do not solve the concrete query. Extract one mechanism, not a broad customer-service handbook."""


FINAL_DOMAIN_CHECK = {
    "airline": """FINAL AIRLINE CHECK
Recheck that every eligibility rule and write action is supported by the visible policy or tools.
Do not invent a policy exception, fee, refund time, or availability requirement.""",
    "retail": """FINAL RETAIL CHECK
Recheck that identifiers and product preferences are abstract, every mutation has its required
status/eligibility/confirmation checks, and the card transfers to another item or order.""",
    "telecom": """FINAL TELECOM CHECK — THE CARD IS INVALID UNLESS ALL ITEMS PASS
First distinguish an account/network state from a device state; do not suggest an unrelated device
action when the observed account state already determines escalation. Under `## Procedure`, write
exactly two bullets with these literal prefixes and no numbered checklist:
- Agent: perform one relevant account-side customer, line, billing, or plan API check. Never put a
  SIM, APN, network-mode, permission, airplane-mode, Wi-Fi, reboot, or other device check here.
- Ask user: perform exactly one relevant device-side observation or action; wait for the user's
  reported result before continuing.
Copy the clause `; wait for the user's reported result before continuing.` verbatim at the end of the
`Ask user:` bullet. Apart from that reporting clause, do not join several device actions with `and`
or `then`.
Put every later alternative only in `## Decision rules` as an if/then branch based on that result.""",
}


def normalize_domain(value: str) -> str:
    lowered = str(value or "").lower()
    for domain in DOMAIN_GUIDANCE:
        if domain in lowered:
            return domain
    raise ValueError(f"unsupported Tau2 E7 domain: {value!r}")


def build_messages(
    *,
    domain: str,
    visible_context: str,
    attempts: Iterable[str],
    outcomes: Iterable[bool],
) -> list[dict[str, str]]:
    domain = normalize_domain(domain)
    attempt_list = list(attempts)
    outcome_list = list(outcomes)
    if len(attempt_list) != 4 or len(outcome_list) != 4:
        raise ValueError("Tau2 E7 generator requires exactly four attempts and outcomes")
    rendered = []
    for idx, (attempt, won) in enumerate(zip(attempt_list, outcome_list)):
        rendered.append(
            f"<ATTEMPT index={idx} outcome={'SUCCESS' if won else 'FAILURE'}>\n"
            f"{attempt}\n</ATTEMPT>"
        )
    user = "\n\n".join(
        [
            f"DOMAIN: {domain}",
            DOMAIN_GUIDANCE[domain],
            "<AGENT_VISIBLE_CONTEXT>\n" + visible_context + "\n</AGENT_VISIBLE_CONTEXT>",
            "\n\n".join(rendered),
            OUTPUT_CONTRACT,
            FINAL_DOMAIN_CHECK[domain],
        ]
    )
    return [
        {"role": "system", "content": COMMON_SYSTEM},
        {"role": "user", "content": user},
    ]
