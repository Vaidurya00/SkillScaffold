#!/usr/bin/env python3
"""Create the shared prefix-stable Tau2 template from Qwen3.5's own Jinja."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ORIGINAL = r"""{%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}"""

PREFIX_STABLE = r"""{%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- elif enable_thinking is defined and enable_thinking is false %}
            {{- '<|im_start|>' + message.role + '\n<think>\n\n</think>\n\n' + content }}
        {%- elif enable_thinking is defined and enable_thinking is true %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}"""

NATIVE_PRESERVE_MARKER = "preserve_thinking is defined and preserve_thinking is true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    occurrences = source.count(ORIGINAL)
    if NATIVE_PRESERVE_MARKER in source:
        patched = source
        template_mode = "native_preserve_thinking"
    elif occurrences == 1:
        patched = source.replace(ORIGINAL, PREFIX_STABLE)
        template_mode = "patched_qwen35"
    elif source.count(PREFIX_STABLE) == 1:
        patched = source
        template_mode = "patched_qwen35"
    else:
        raise SystemExit(f"Expected exactly one Qwen3.5 assistant-history block, found {occurrences}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(patched, encoding="utf-8")
    print(f"qwen35_tau2_chat_template={args.output}")
    print(f"qwen35_tau2_chat_template_mode={template_mode}")
    print("qwen35_tau2_chat_template_sha256=" + hashlib.sha256(patched.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    main()
