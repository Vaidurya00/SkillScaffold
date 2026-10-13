"""Helpers for mapping generated assistant content onto a rendered chat."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from tau2_rl_pipeline.actions import (
    QWEN3_JSON_PROTOCOL,
    QWEN35_XML_PROTOCOL,
    ToolProtocol,
)

ASSISTANT_HEADER = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"
THINKING_DISABLED_PREFIX = "<think>\n\n</think>\n\n"
THINKING_ENABLED_PREFIX = "<think>\n"
QWEN_TURN_END = "<|im_end|>"


def _append_only_content(content: object, *, strip: bool) -> str:
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError(
            "Tau2 append-only rollout supports text-only user/tool observations, "
            f"got {type(content).__name__}"
        )
    return content.strip() if strip else content


def render_append_only_followup(
    messages: Sequence[dict[str, object]],
    *,
    protocol: ToolProtocol,
    assistant_turn_closed: bool = False,
    enable_thinking: bool = False,
) -> str:
    """Serialize only the new Tau2 observation and next generation prefix.

    The returned text is appended verbatim to the tokens already sampled for
    the assistant turn. It mirrors the Qwen chat templates' user/tool role
    serialization without re-rendering any prior history.
    """
    if protocol not in {QWEN3_JSON_PROTOCOL, QWEN35_XML_PROTOCOL}:
        raise ValueError(f"Unsupported Tau2 tool protocol: {protocol!r}")

    strip_content = protocol == QWEN35_XML_PROTOCOL
    # The sampled EOS token closes the assistant role but does not include the
    # template's following newline.
    parts = ["\n"] if assistant_turn_closed else [TURN_END, "\n"]
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "tool":
            parts.append("<|im_start|>user")
            while index < len(messages) and messages[index].get("role") == "tool":
                content = _append_only_content(
                    messages[index].get("content"), strip=strip_content
                )
                parts.extend(["\n<tool_response>\n", content, "\n</tool_response>"])
                index += 1
            parts.append("<|im_end|>\n")
            continue
        if role == "user" or (role == "system" and protocol == QWEN3_JSON_PROTOCOL):
            content = _append_only_content(message.get("content"), strip=strip_content)
            parts.extend([f"<|im_start|>{role}\n", content, "<|im_end|>\n"])
            index += 1
            continue
        raise ValueError(f"Unsupported append-only Tau2 observation role: {role!r}")

    parts.append(ASSISTANT_HEADER)
    if protocol == QWEN35_XML_PROTOCOL:
        parts.append(
            THINKING_ENABLED_PREFIX if enable_thinking else THINKING_DISABLED_PREFIX
        )
    return "".join(parts)


def strip_retained_turn_end(
    tokenizer, text: str, sampled_token_ids: list[int]
) -> tuple[str, list[int]]:
    """Remove SGLang's retained Qwen turn delimiter from message content."""
    if not text.endswith(QWEN_TURN_END):
        return text, sampled_token_ids
    text = text[: -len(QWEN_TURN_END)]
    turn_end_ids = tokenizer.encode(QWEN_TURN_END, add_special_tokens=False)
    if turn_end_ids and sampled_token_ids[-len(turn_end_ids) :] == turn_end_ids:
        sampled_token_ids = sampled_token_ids[: -len(turn_end_ids)]
    return text, sampled_token_ids


def _visible_assistant_content(content: object, *, template_strips_thinking: bool) -> str:
    """Mirror the Qwen3.5 template's handling of inline think blocks."""
    visible = str(content)
    if template_strips_thinking:
        visible = visible.strip()
    if template_strips_thinking and "</think>" in visible:
        visible = visible.rsplit("</think>", 1)[1].lstrip("\n")
    return visible


def _stabilize_thinking_prefixes(
    rendered_text: str,
    messages: Sequence[dict[str, object]],
    *,
    enable_thinking: bool,
) -> str:
    """Keep Qwen's generation prefix on every sampled assistant turn.

    Qwen3.5's native template removes reasoning wrappers from older assistant
    turns once a later ordinary user message appears. Reinserting the exact
    generation prefix makes the rendered trajectory causal-prefix stable
    without duplicating the template's tool or role serialization.
    """
    rendered = rendered_text
    prefix = THINKING_ENABLED_PREFIX if enable_thinking else THINKING_DISABLED_PREFIX
    cursor = 0
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        header_start = rendered.find(ASSISTANT_HEADER, cursor)
        if header_start < 0:
            raise ValueError(f"Assistant header not found for message {message_index}")
        body_start = header_start + len(ASSISTANT_HEADER)
        if not rendered.startswith("<think>", body_start):
            rendered = rendered[:body_start] + prefix + rendered[body_start:]
        turn_end = rendered.find(TURN_END, body_start)
        if turn_end < 0:
            raise ValueError(f"Assistant turn end not found for message {message_index}")
        cursor = turn_end + len(TURN_END)
    return rendered


def _restore_sampled_assistant_contents(
    rendered_text: str,
    messages: Sequence[dict[str, object]],
    *,
    enable_thinking: bool = False,
) -> str:
    """Restore the exact on-policy text recorded for sampled assistant turns.

    Qwen3.5's Jinja template trims assistant content and removes inline think
    blocks when it re-renders history. That behavior is appropriate for native
    inference history, but a policy-gradient sequence must retain the exact
    text emitted by SGLang, including malformed output that receives the
    format penalty. ``_sampled_content`` is private rollout metadata ignored by
    the native template, so replace only those assistant bodies after the
    native role/tool structure has been rendered.
    """
    rendered = rendered_text
    cursor = 0
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        header_start = rendered.find(ASSISTANT_HEADER, cursor)
        if header_start < 0:
            raise ValueError(f"Assistant header not found for message {message_index}")
        body_start = header_start + len(ASSISTANT_HEADER)
        content_start = body_start
        thinking_prefix = (
            THINKING_ENABLED_PREFIX if enable_thinking else THINKING_DISABLED_PREFIX
        )
        if rendered.startswith(thinking_prefix, body_start):
            content_start += len(thinking_prefix)

        next_header = rendered.find("<|im_start|>", content_start)
        search_end = len(rendered) if next_header < 0 else next_header
        turn_end = rendered.rfind(TURN_END, content_start, search_end)
        if turn_end < 0:
            raise ValueError(f"Assistant turn end not found for message {message_index}")

        if "_sampled_content" in message:
            sampled_content = str(message["_sampled_content"])
            rendered = (
                rendered[:content_start]
                + sampled_content
                + rendered[turn_end:]
            )
            turn_end = content_start + len(sampled_content)
        cursor = turn_end + len(TURN_END)
    return rendered


def render_prefix_stable_qwen35(
    tokenizer,
    messages: Sequence[dict[str, object]],
    *,
    tools: Sequence[dict[str, object]] | None = None,
    add_generation_prompt: bool,
    enable_thinking: bool = False,
    preserve_thinking: bool = False,
) -> str:
    """Render Qwen3.5's native tools/roles with a stable thinking prefix.

    The model directory's own chat template remains the source of truth for
    the native ``<function>/<parameter>`` protocol and ``<tool_response>``
    messages. ``enable_thinking`` controls only Qwen3.5; callers keep the
    historical no-thinking behavior by omitting it.
    """
    template_kwargs = {"enable_thinking": enable_thinking}
    if preserve_thinking:
        template_kwargs["preserve_thinking"] = True
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )
    rendered = _stabilize_thinking_prefixes(
        rendered,
        messages,
        enable_thinking=enable_thinking,
    )
    return _restore_sampled_assistant_contents(
        rendered,
        messages,
        enable_thinking=enable_thinking,
    )


def render_tau2_qwen(
    tokenizer,
    messages: Sequence[dict[str, object]],
    *,
    tools: Sequence[dict[str, object]] | None = None,
    add_generation_prompt: bool,
    protocol: ToolProtocol,
    restore_sampled_content: bool = False,
    enable_thinking: bool = False,
    preserve_thinking: bool = False,
) -> str:
    """Render the native tool protocol for either supported Qwen family."""
    if protocol == QWEN35_XML_PROTOCOL:
        return render_prefix_stable_qwen35(
            tokenizer,
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking,
        )
    if protocol == QWEN3_JSON_PROTOCOL:
        if enable_thinking or preserve_thinking:
            raise ValueError("Tau2 Qwen3 JSON rollout does not support thinking")
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
        # Native eval receives structured tool calls from vLLM and therefore
        # canonicalizes history before the next turn. Keep that behavior for
        # generation prompts. PPO's final loss sequence is different: it must
        # contain the exact tokens SGLang sampled, including harmless JSON
        # whitespace/key-order differences accepted by the Hermes parser.
        return (
            _restore_sampled_assistant_contents(rendered, messages)
            if restore_sampled_content
            else rendered
        )
    raise ValueError(f"Unsupported Tau2 tool protocol: {protocol!r}")


def validate_native_assistant_turn(
    tokenizer,
    messages: Sequence[dict[str, object]],
    assistant_message: dict[str, object],
    sampled_text: str,
    *,
    tools: Sequence[dict[str, object]] | None = None,
    protocol: ToolProtocol = QWEN35_XML_PROTOCOL,
    enable_thinking: bool = False,
    preserve_thinking: bool = False,
) -> None:
    """Require sampled text to equal the structured native-history render."""
    before = render_tau2_qwen(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=True,
        protocol=protocol,
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
    )
    # Render the new turn from its native structured fields. The private
    # sampled-content field is deliberately omitted here; otherwise restoring
    # it would make every parseable-but-noncanonical call validate itself.
    native_assistant_message = {
        key: value
        for key, value in assistant_message.items()
        if key != "_sampled_content"
    }
    after = render_tau2_qwen(
        tokenizer,
        [*messages, native_assistant_message],
        tools=tools,
        add_generation_prompt=False,
        protocol=protocol,
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
    )
    expected = before + sampled_text + TURN_END + "\n"
    if after == expected:
        return

    mismatch = min(len(after), len(expected))
    for index, (actual_char, expected_char) in enumerate(
        zip(after, expected, strict=False)
    ):
        if actual_char != expected_char:
            mismatch = index
            break
    raise ValueError(
        "Output is parseable but not canonical for the selected Qwen native "
        "function-call protocol "
        f"serialization (first mismatch at character {mismatch})"
    )


def assistant_content_spans(
    rendered_text: str,
    messages: Sequence[dict[str, object]],
    *,
    template_strips_thinking: bool = True,
    sampled_only: bool = False,
) -> list[tuple[int, int]]:
    """Locate only message-provided assistant content in a rendered chat.

    Qwen3.5 inserts an empty ``<think>`` block before assistant content when
    thinking is disabled and always appends ``<|im_end|>``. Neither string was
    sampled by the policy, so neither belongs in the policy-gradient mask.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0

    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        if sampled_only and "_sampled_content" not in message:
            continue

        content = _visible_assistant_content(
            message.get("_sampled_content", message.get("content", "")),
            template_strips_thinking=template_strips_thinking,
        )
        header_start = rendered_text.find(ASSISTANT_HEADER, cursor)
        if header_start < 0:
            raise ValueError(f"Assistant header not found for message {message_index}")
        body_start = header_start + len(ASSISTANT_HEADER)

        if not content:
            turn_end = rendered_text.find(TURN_END, body_start)
            if turn_end < 0:
                raise ValueError(f"Assistant turn end not found for message {message_index}")
            cursor = turn_end + len(TURN_END)
            continue

        content_start = rendered_text.find(content, body_start)
        while content_start >= 0:
            content_end = content_start + len(content)
            # The template's own turn terminator must immediately follow the
            # message content. This also handles sampled content ending in an
            # im_end token: the rendered text then contains two in a row.
            if rendered_text.startswith(TURN_END, content_end):
                break
            content_start = rendered_text.find(content, content_start + 1)
        else:
            raise ValueError(f"Assistant content not found in rendered message {message_index}")

        spans.append((content_start, content_end))
        cursor = content_end + len(TURN_END)

    return spans


def stitch_sampled_assistant_tokens(
    tokenizer,
    rendered_text: str,
    spans: Sequence[tuple[int, int]],
    sampled_texts: Sequence[str],
    sampled_token_ids: Sequence[Sequence[int]],
) -> tuple[list[int], list[int]]:
    """Build one trajectory while preserving each sampled action token ID.

    Generated text can have more than one valid BPE segmentation. Retokenizing
    a completed multi-turn transcript therefore occasionally produces token
    IDs different from the IDs sampled by SGLang even though the decoded text
    is identical. Encode only the context segments and splice the original
    sampled IDs into assistant spans so PPO always trains on its on-policy
    actions.
    """
    if not (len(spans) == len(sampled_texts) == len(sampled_token_ids)):
        raise ValueError(
            "Sampled assistant turn count does not match rendered spans: "
            f"spans={len(spans)} texts={len(sampled_texts)} "
            f"token_groups={len(sampled_token_ids)}"
        )

    token_ids: list[int] = []
    loss_mask: list[int] = []
    cursor = 0
    for turn_index, ((start, end), sampled_text, turn_token_ids) in enumerate(
        zip(spans, sampled_texts, sampled_token_ids, strict=True)
    ):
        if start < cursor or end < start:
            raise ValueError(f"Invalid assistant span at turn {turn_index}: {(start, end)}")
        rendered_sample = rendered_text[start:end]
        if rendered_sample != sampled_text:
            raise ValueError(
                f"Rendered assistant text differs from sampled turn {turn_index}"
            )

        context_ids = tokenizer.encode(
            rendered_text[cursor:start], add_special_tokens=False
        )
        token_ids.extend(context_ids)
        loss_mask.extend([0] * len(context_ids))
        token_ids.extend(turn_token_ids)
        loss_mask.extend([1] * len(turn_token_ids))
        cursor = end

    tail_ids = tokenizer.encode(rendered_text[cursor:], add_special_tokens=False)
    token_ids.extend(tail_ids)
    loss_mask.extend([0] * len(tail_ids))
    return token_ids, loss_mask


def loss_mask_from_offsets(
    offsets: Iterable[tuple[int, int]], spans: Sequence[tuple[int, int]]
) -> list[int]:
    """Mark tokens whose complete character offsets fall inside a span."""
    mask: list[int] = []
    span_index = 0

    for start, end in offsets:
        if start == end:
            mask.append(0)
            continue
        while span_index < len(spans) and spans[span_index][1] <= start:
            span_index += 1
        inside = (
            span_index < len(spans)
            and start >= spans[span_index][0]
            and end <= spans[span_index][1]
        )
        mask.append(1 if inside else 0)

    return mask
