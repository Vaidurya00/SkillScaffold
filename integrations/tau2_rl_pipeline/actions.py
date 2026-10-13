"""Tau2 assistant-action parsing and observation formatting.

Assistant turns follow Tau2's native two-branch protocol: plain text is sent
to the user, while tool calls use the model family's own wire format.

Qwen3 uses Hermes JSON inside ``<tool_call>``; Qwen3.5 uses the newer
``<function>/<parameter>`` XML representation.

  <tool_call>
  <function=tool_name>
  <parameter=argument_name>
  argument value
  </parameter>
  </function>
  </tool_call>
    -> environment tool
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence


_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
_FUNCTION_RE = re.compile(
    r"<function=([^>\n]+)>\s*(.*?)\s*</function>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)

ToolProtocol = Literal["qwen3_json", "qwen35_xml"]
QWEN3_JSON_PROTOCOL: ToolProtocol = "qwen3_json"
QWEN35_XML_PROTOCOL: ToolProtocol = "qwen35_xml"


@dataclass(frozen=True, slots=True)
class ParsedAction:
    name: str
    arguments: dict[str, Any]
    raw_action_call: str
    assistant_content: str | None = None


def canonical_action_key(action: ParsedAction) -> tuple[str, str]:
    """Return an order-independent key for consecutive-action detection."""
    arguments = json.dumps(
        action.arguments,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return action.name, arguments


def _to_py_literal(value: Any) -> str:
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_to_py_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            items.append(f"{_to_py_literal(k)}: {_to_py_literal(v)}")
        return "{" + ", ".join(items) + "}"
    return repr(value)


def _to_functional_call(name: str, arguments: dict[str, Any]) -> str:
    if not arguments:
        return f"{name}()"
    parts = [f"{k}={_to_py_literal(v)}" for k, v in arguments.items()]
    return f"{name}({', '.join(parts)})"


def _find_tool_call_blocks(text: str) -> list[tuple[str, int, int]]:
    blocks: list[tuple[str, int, int]] = []
    cursor = 0
    while True:
        start = text.find(_TOOL_CALL_START, cursor)
        if start == -1:
            break
        end = text.find(_TOOL_CALL_END, start + len(_TOOL_CALL_START))
        if end == -1:
            raise ValueError("Missing </tool_call> for <tool_call> block")
        content = text[start + len(_TOOL_CALL_START) : end].strip()
        blocks.append((content, start, end + len(_TOOL_CALL_END)))
        cursor = end + len(_TOOL_CALL_END)
    return blocks


def _strip_thinking_prefix(text: str) -> str:
    """Return the visible assistant output after Qwen's reasoning prefix."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


def _function_schema(
    tools_openai: list[dict[str, Any]] | None, name: str
) -> dict[str, Any] | None:
    for tool in tools_openai or []:
        function = tool.get("function", tool)
        if function.get("name") == name:
            return function
    return None


def _schema_type(schema: dict[str, Any]) -> str | None:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return schema_type
    if isinstance(schema_type, list):
        return next((item for item in schema_type if item != "null"), None)
    for branch in schema.get("anyOf", []) or schema.get("oneOf", []):
        branch_type = _schema_type(branch)
        if branch_type and branch_type != "null":
            return branch_type
    return None


def _parse_parameter_value(raw_value: str, schema: dict[str, Any]) -> Any:
    value = raw_value.strip()
    value_type = _schema_type(schema)

    # Match Qwen's qwen3_coder parser: null is accepted for every schema type.
    if value.lower() == "null":
        return None
    if not schema:
        return value
    if value_type == "string":
        return value
    if value_type == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    if value_type == "number":
        try:
            parsed = float(value)
            return int(parsed) if parsed.is_integer() else parsed
        except ValueError:
            return value
    if value_type == "boolean":
        return value.lower() == "true"
    if value_type == "object":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _parse_qwen35_xml_action(
    text: str,
    tools_openai: list[dict[str, Any]] | None = None,
) -> ParsedAction:
    """Parse one Qwen3.5-native XML tool call.

    Text before the <tool_call> block is preserved as assistant content
    (models naturally emit transitions like "Let me check that for you."
    before a call). Text after the block and multiple calls are rejected
    because native structured history cannot reproduce them losslessly.
    Parameter values are converted with the corresponding OpenAI tool schema,
    matching the typed arguments that Tau2's native function-calling harness
    passes to the environment.

    Use :func:`parse_assistant_action` for model output because plain text is a
    valid user-facing action in Tau2.
    """
    text = _strip_thinking_prefix(text)

    blocks = _find_tool_call_blocks(text)
    if not blocks:
        raise ValueError("Missing <tool_call>...</tool_call> block")
    if len(blocks) != 1:
        raise ValueError("Expected exactly one <tool_call> block")
    content, start, end = blocks[0]
    # vLLM's qwen3_coder parser returns the prefix verbatim. Preserve it in
    # Tau2's trajectory; the template normalizes it when rendering history.
    assistant_content = text[:start] or None
    if text[end:].strip():
        raise ValueError("Unexpected text after </tool_call>")

    functions = list(_FUNCTION_RE.finditer(content))
    if not functions:
        raise ValueError("Tool call missing <function=...>...</function> block")
    function = functions[0]
    name = function.group(1).strip()
    if not name:
        raise ValueError("Tool call has an empty function name")

    function_schema = _function_schema(tools_openai, name)
    properties = (
        function_schema.get("parameters", {}).get("properties", {})
        if function_schema
        else {}
    )
    arguments: dict[str, Any] = {}
    body = function.group(2)
    parameters = list(_PARAMETER_RE.finditer(body))
    residue = _PARAMETER_RE.sub("", body).strip()
    if residue:
        raise ValueError(f"Malformed content inside <function={name}>: {residue[:80]!r}")
    for parameter in parameters:
        parameter_name = parameter.group(1).strip()
        if not parameter_name:
            raise ValueError("Tool call has an empty parameter name")
        if parameter_name in arguments:
            raise ValueError(f"Duplicate tool parameter: {parameter_name!r}")
        parameter_schema = properties.get(parameter_name, {})
        try:
            arguments[parameter_name] = _parse_parameter_value(
                parameter.group(2), parameter_schema
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid value for parameter {parameter_name!r}: {exc}"
            ) from exc

    return ParsedAction(
        name=name,
        arguments=arguments,
        raw_action_call=_to_functional_call(name, arguments),
        assistant_content=assistant_content,
    )


def _parse_qwen3_json_action(text: str) -> ParsedAction:
    """Parse Qwen3's native Hermes-style JSON tool call."""
    text = _strip_thinking_prefix(text)
    blocks = _find_tool_call_blocks(text)
    if not blocks:
        raise ValueError("Missing <tool_call>...</tool_call> block")
    if len(blocks) != 1:
        raise ValueError("Expected exactly one <tool_call> block")

    content, start, end = blocks[0]
    # Hermes exposes content before the first tool marker byte-for-byte.
    assistant_content = text[:start] or None
    if text[end:].strip():
        raise ValueError("Unexpected text after </tool_call>")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON inside <tool_call>: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Qwen3 tool call must contain one JSON object")

    name = data.get("name")
    arguments = data.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise ValueError("Tool call missing non-empty 'name'")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool call has invalid JSON arguments: {exc}") from exc
    if not isinstance(arguments, dict):
        raise ValueError("Tool call 'arguments' must be an object")

    return ParsedAction(
        name=name,
        arguments=arguments,
        raw_action_call=_to_functional_call(name, arguments),
        assistant_content=assistant_content,
    )


def parse_action(
    text: str,
    tools_openai: list[dict[str, Any]] | None = None,
    *,
    protocol: ToolProtocol = QWEN35_XML_PROTOCOL,
) -> ParsedAction:
    """Parse one native tool call for the selected Qwen model family."""
    if protocol == QWEN3_JSON_PROTOCOL:
        return _parse_qwen3_json_action(text)
    if protocol == QWEN35_XML_PROTOCOL:
        return _parse_qwen35_xml_action(text, tools_openai=tools_openai)
    raise ValueError(f"Unsupported Tau2 tool protocol: {protocol!r}")


def parse_assistant_action(
    text: str,
    tools_openai: list[dict[str, Any]] | None = None,
    *,
    protocol: ToolProtocol = QWEN35_XML_PROTOCOL,
) -> ParsedAction:
    """Classify one model turn as either user-facing text or a tool call."""
    content = _strip_thinking_prefix(text)
    visible = content.strip()
    if not visible:
        raise ValueError("Empty assistant output")
    if _TOOL_CALL_START not in visible:
        # Qwen3's native template preserves message-boundary whitespace,
        # whereas Qwen3.5 strips it. Keep sampled Qwen3 text byte-for-byte so
        # the final full-trajectory render remains on-policy.
        assistant_content = content if protocol == QWEN3_JSON_PROTOCOL else visible
        return ParsedAction(
            name="respond",
            arguments={"content": assistant_content},
            raw_action_call=f"respond(content={visible[:80]!r})",
            assistant_content=assistant_content,
        )
    return parse_action(
        content,
        tools_openai=tools_openai,
        protocol=protocol,
    )


def native_assistant_history_message(
    action: ParsedAction,
    *,
    sampled_content: str,
) -> dict[str, Any]:
    """Build the assistant history shape sent by Tau2 through LiteLLM.

    ``_sampled_content`` is ignored by Qwen's Jinja template and records the
    exact on-policy text whose tokens must receive loss. Tool calls themselves
    are structured so the next turn is re-rendered exactly like native eval.
    """
    if action.name == "respond":
        return {
            "role": "assistant",
            "content": action.assistant_content,
            "_sampled_content": sampled_content,
        }
    return {
        "role": "assistant",
        "content": action.assistant_content,
        "tool_calls": [
            {
                "id": "",
                "type": "function",
                "function": {
                    "name": action.name,
                    "arguments": action.arguments,
                },
            }
        ],
        "_sampled_content": sampled_content,
    }


_KNOWN_OBS_ROLES = {"user", "tool", "assistant", "system"}


def _strip_role_prefix(line: str) -> tuple[str | None, str]:
    """Split a leading known-role prefix off an observation line.

    Only user/tool/assistant/system count as role markers. Anything else with
    a ": " in it (e.g. "- Name: Morgan Mitchell" inside a multi-line user
    message) is a continuation line of the current role — treating it as a
    role prefix used to shunt the middle of multi-line user messages into the
    discarded `other` bucket, so the agent literally never saw them.
    """
    stripped = line.strip()
    if ": " in stripped:
        role, content = stripped.split(": ", 1)
        role_key = role.strip().lower()
        if role_key in _KNOWN_OBS_ROLES:
            return role_key, content
    return None, stripped


@dataclass(frozen=True, slots=True)
class ParsedObservation:
    user: str
    tool: str
    other: str


def split_observation(observation: str) -> ParsedObservation:
    user_lines: list[str] = []
    tool_lines: list[str] = []
    other_lines: list[str] = []
    current: list[str] | None = None

    for raw_line in (observation or "").splitlines():
        role, content = _strip_role_prefix(raw_line)
        if role == "user":
            current = user_lines
        elif role == "tool":
            current = tool_lines
        elif role is not None:  # assistant/system echoes: context we don't relay
            current = other_lines
        elif current is None:  # continuation before any role marker
            current = other_lines
        if content:
            current.append(content)

    return ParsedObservation(
        user="\n".join(user_lines).strip(),
        tool="\n".join(tool_lines).strip(),
        other="\n".join(other_lines).strip(),
    )


def followup_messages_for_observation(
    *,
    observation: str,
    last_action_call: str,
    last_action_was_tool: bool,
) -> list[dict[str, str]]:
    parsed = split_observation(observation)
    messages: list[dict[str, str]] = []

    if last_action_was_tool:
        tool_payload = parsed.tool or parsed.other or "[no_observation]"
        messages.append({"role": "tool", "content": tool_payload})
        if parsed.user:
            messages.append({"role": "user", "content": parsed.user})
        return messages

    user_payload = parsed.user or parsed.other or "[no_observation]"
    messages.append({"role": "user", "content": user_payload})
    return messages


def native_followup_messages_from_tau2(
    observation: Sequence[Any],
) -> list[dict[str, Any]]:
    """Convert GymAgent's native observation suffix like Tau2's LLMAgent.

    ``AgentGymEnv`` exposes a lossy string observation intended for generic Gym
    clients. In telecom, that string flattens user-side tool calls and can drop
    their tool responses. The underlying ``GymAgent.observation`` retains the
    original Tau2 messages. Keep only messages after the most recent assistant
    action, then mirror ``tau2.utils.llm_utils.to_litellm_messages`` for the
    user/tool roles that become the next policy observation.

    Tau2 deliberately sends user tool calls to the policy as the user's text
    content plus subsequent tool-role messages; it does not expose the user
    tool call itself through the OpenAI ``tools`` field.
    """
    last_assistant = -1
    for index, message in enumerate(observation):
        if getattr(message, "role", None) == "assistant":
            last_assistant = index

    converted: list[dict[str, Any]] = []
    for message in observation[last_assistant + 1 :]:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role == "user":
            converted.append({"role": "user", "content": content})
        elif role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": getattr(message, "id", ""),
                }
            )
        elif role == "system":
            converted.append({"role": "system", "content": content})
        elif role == "assistant":  # Defensive; excluded by the suffix above.
            continue
        else:
            raise ValueError(f"Unsupported Tau2 observation role: {role!r}")
    return converted


def env_action_from_parsed_action(action: ParsedAction) -> str:
    if action.name == "respond":
        content = action.arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("respond requires a non-empty content string")
        return content
    return action.raw_action_call
