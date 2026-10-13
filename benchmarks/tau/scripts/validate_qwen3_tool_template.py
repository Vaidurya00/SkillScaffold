#!/usr/bin/env python3
"""Fail fast unless a Qwen3 tokenizer supports multi-turn Hermes tools."""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--slow", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        use_fast=not args.slow,
    )
    tools = [{"type": "function", "function": {
        "name": "lookup", "description": "Look something up.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }}]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_0", "type": "function",
            "function": {"name": "lookup", "arguments": {"query": "x"}},
        }]},
        {"role": "tool", "tool_call_id": "call_0", "content": "result"},
        {"role": "user", "content": "continue"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    required = ("<tools>", '"name": "lookup"', "<tool_call>", "<tool_response>\nresult\n</tool_response>")
    missing = [marker for marker in required if marker not in rendered]
    if missing:
        raise ValueError(f"Qwen3 chat template is missing: {missing}")
    if not rendered.endswith("<|im_start|>assistant\n"):
        raise ValueError("Qwen3 chat template has an invalid generation suffix")
    print(f"tokenizer={type(tokenizer).__name__}")
    print(f"rendered_chars={len(rendered)}")
    print("qwen3_tool_template=ready")


if __name__ == "__main__":
    main()
