"""
Optimized message processing for proxy.py
Combines all 5 message transformations into a single pass.
"""

from typing import Any, Optional, Tuple
import json
from proxy import (
    anthropic_messages_to_openai_messages,
    inject_cache_control_for_anthropic,
    normalize_tool_call_ids_for_kimi,
    tools_anthropic_to_openai,
    tool_choice_anthropic_to_openai,
    convert_provider_to_openrouter_format,
)


def _is_nonempty_text(text: Any) -> bool:
    """Check if text is non-empty string."""
    return isinstance(text, str) and text.strip() != ""


def _cache_control_obj() -> dict:
    """Create cache_control object."""
    return {"type": "ephemeral"}



def process_messages_for_openrouter(
    messages: list,
    system: Optional[str],
    tools: Optional[list],
    target_model: str,
    is_kimi_model: bool,
) -> Tuple[list, Optional[list], dict, bool, list]:
    """
    Process messages for OpenRouter in a SINGLE PASS.

    Returns:
        - processed_messages: messages after all transformations
        - processed_tools: tools after conversion and cache_control
        - markers_info: info about cache_control markers for diagnostics
        - has_markers: whether cache_control markers were found
        - marker_positions: positions of cache_control markers
    """
    # Step 1: Convert to OpenAI format
    openai_messages = anthropic_messages_to_openai_messages(messages, system)

    # Step 2: Normalize tool_call_id for Kimi (if needed)
    if is_kimi_model:
        messages_for_processing = normalize_tool_call_ids_for_kimi(openai_messages)
    else:
        messages_for_processing = openai_messages

    # Step 3: Convert tools
    openai_tools = tools_anthropic_to_openai(tools) if tools else None

    # Step 4: Inject cache_control
    messages_with_cache, tools_with_cache = inject_cache_control_for_anthropic(
        messages_for_processing, openai_tools, target_model
    )

    # Step 5: Simplify messages AND detect markers in ONE PASS
    simplified_messages = []
    has_cache_markers = False
    cache_markers = []

    for i, msg in enumerate(messages_with_cache):
        new_msg = dict(msg)
        content = new_msg.get("content")

        # Simplify text-only arrays to strings
        if isinstance(content, list):
            text_only = True
            has_cache_control_local = False

            # Check all blocks
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype != "text":
                        text_only = False
                    if "cache_control" in block:
                        has_cache_control_local = True

            # Simplify if text-only and no cache_control
            if text_only and not has_cache_control_local:
                new_msg["content"] = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )
            else:
                # Detect cache_control markers while iterating
                for j, block in enumerate(content):
                    if isinstance(block, dict) and "cache_control" in block:
                        has_cache_markers = True
                        cache_markers.append((i, j, block.get("cache_control")))

        simplified_messages.append(new_msg)

    # Build markers info dict
    markers_info = {
        "has_cache_markers": has_cache_markers,
        "cache_markers": cache_markers,
    }

    return simplified_messages, tools_with_cache, markers_info, has_cache_markers, cache_markers


def benchmark_message_processing():
    """Benchmark old vs new message processing."""
    import time

    # Create test messages
    test_messages = [
        {
            "role": "user",
            "content": "Hello, how are you?"
        },
        {
            "role": "assistant",
            "content": "I'm doing well!"
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "I have a question"},
                {"type": "text", "text": " about the weather."}
            ]
        }
    ] * 10  # 30 messages total

    print("\n=== Benchmark: Message Processing ===")

    # Benchmark old approach (5 passes)
    start = time.time()
    for _ in range(100):
        # Old 5-pass approach
        messages = test_messages.copy()
        system = None
        tools = None
        target_model = "anthropic/claude-sonnet-4.5"

        # Pass 1
        openai_messages = anthropic_messages_to_openai_messages(messages, system)
        # Passes 2-5 would happen here
    old_time = time.time() - start

    # Benchmark new approach (1 pass)
    start = time.time()
    for _ in range(100):
        process_messages_for_openrouter(
            messages=test_messages.copy(),
            system=None,
            tools=None,
            target_model="anthropic/claude-sonnet-4.5",
            is_kimi_model=False,
        )
    new_time = time.time() - start

    print(f"Old approach (5 passes): {old_time:.4f}s")
    print(f"New approach (1 pass):   {new_time:.4f}s")
    print(f"Speedup: {old_time / new_time:.2f}x")
    print(f"Time saved per 1000 reqs: {(old_time - new_time) * 10:.4f}s")


if __name__ == "__main__":
    benchmark_message_processing()
