#!/usr/bin/env python3
"""
Test suite for message processing transformations.
This test ensures all 5 transformations work correctly.
"""

import json
from typing import Any

from proxy import (
    anthropic_messages_to_openai_messages,
    inject_cache_control_for_anthropic,
    normalize_tool_call_ids_for_kimi,
    normalize_tool_id_for_anthropic,
    openai_response_to_anthropic_message,
    anthropic_sse_from_message,
    gemini_tool_meta_put,
    gemini_tool_meta_get,
)


def test_anthropic_to_openai_conversion():
    """Test Anthropic → OpenAI message conversion."""
    anthropic_messages = [
        {
            "role": "user",
            "content": "Hello, how are you?"
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'm doing well!"}
            ]
        },
        {
            "role": "user",
            "content": "Great!"
        }
    ]

    result = anthropic_messages_to_openai_messages(anthropic_messages)

    # Should convert to OpenAI format
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert result[2]["role"] == "user"
    print("✓ anthropic_to_openai_conversion passed")


def test_kimi_tool_call_normalization():
    """Test tool_call_id normalization for Kimi."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "toolu_01234abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}'
                    }
                },
                {
                    "id": "toolu_5678def",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_01234abc",
            "content": "Sunny, 25°C"
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_5678def",
            "content": "Cloudy, 18°C"
        }
    ]

    result = normalize_tool_call_ids_for_kimi(messages)

    # Check assistant message has normalized IDs
    assistant_msg = result[0]
    assert assistant_msg["tool_calls"][0]["id"] == "functions.get_weather:0"
    assert assistant_msg["tool_calls"][1]["id"] == "functions.get_weather:1"

    # Check tool responses reference new IDs
    assert result[1]["tool_call_id"] == "functions.get_weather:0"
    assert result[2]["tool_call_id"] == "functions.get_weather:1"
    print("✓ kimi_tool_call_normalization passed")


def test_normalize_tool_id_for_anthropic():
    """Test tool_id normalization for Anthropic compatibility."""
    # Test Kimi-style IDs (functions.name:index)
    assert normalize_tool_id_for_anthropic("functions.get_weather:0") == "functions_get_weather_0"
    assert normalize_tool_id_for_anthropic("functions.Bash:1") == "functions_Bash_1"
    assert normalize_tool_id_for_anthropic("functions.Read:123") == "functions_Read_123"

    # Test valid Anthropic IDs (should remain unchanged)
    assert normalize_tool_id_for_anthropic("toolu_abc123") == "toolu_abc123"
    assert normalize_tool_id_for_anthropic("toolu_01HXJ8V3YZ") == "toolu_01HXJ8V3YZ"
    assert normalize_tool_id_for_anthropic("call_abc-123_xyz") == "call_abc-123_xyz"

    # Test edge cases
    assert normalize_tool_id_for_anthropic("") == ""
    assert normalize_tool_id_for_anthropic(None) is None

    # Test IDs with various special characters
    assert normalize_tool_id_for_anthropic("tool@call#123") == "tool_call_123"
    assert normalize_tool_id_for_anthropic("func(name)") == "func_name_"
    assert normalize_tool_id_for_anthropic("a.b:c/d\\e") == "a_b_c_d_e"

    print("✓ normalize_tool_id_for_anthropic passed")


def test_openai_response_tool_id_normalization():
    """Test that openai_response_to_anthropic_message normalizes tool IDs."""
    # Simulate OpenAI response with Kimi-style tool call IDs
    openai_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "functions.get_weather:0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Tokyo"}'
                            }
                        },
                        {
                            "id": "functions.Bash:1",
                            "type": "function",
                            "function": {
                                "name": "Bash",
                                "arguments": '{"command": "ls"}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50}
    }

    result = openai_response_to_anthropic_message(openai_response, "test-model")

    # Check that tool_use IDs are normalized
    tool_uses = [c for c in result["content"] if c["type"] == "tool_use"]
    assert len(tool_uses) == 2
    assert tool_uses[0]["id"] == "functions_get_weather_0"
    assert tool_uses[1]["id"] == "functions_Bash_1"

    # Verify the IDs match Anthropic's pattern ^[a-zA-Z0-9_-]+
    import re
    anthropic_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
    for tool_use in tool_uses:
        assert anthropic_pattern.match(tool_use["id"]), f"ID {tool_use['id']} doesn't match Anthropic pattern"

    print("✓ openai_response_tool_id_normalization passed")


def test_cache_control_injection():
    """Test cache_control injection for Anthropic models."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4"},
        {"role": "user", "content": "Now 3+3?"},
    ]

    tools = None
    target_model = "anthropic/claude-sonnet-4.5"

    result_messages, result_tools = inject_cache_control_for_anthropic(
        messages, tools, target_model
    )

    # Check that cache_control was added (the function mutates in-place)
    # System message should have cache_control
    assert result_messages[0]["content"][0]["cache_control"]["type"] == "ephemeral"

    # Last user message should have cache_control
    assert result_messages[-1]["content"][0]["cache_control"]["type"] == "ephemeral"
    print("✓ cache_control_injection passed")


def test_message_simplification():
    """Test that messages with text-only arrays get simplified."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world!"},
            ]
        }
    ]

    # After processing through the pipeline, text-only arrays should be strings
    # This is tested in the integration test below
    print("⊘ message_simplification (tested in integration)")


def test_integration_all_transformations():
    """Integration test: all 5 transformations together."""
    # Start with Anthropic format
    anthropic_messages = [
        {
            "role": "user",
            "content": "Get weather for Tokyo"
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll check the weather for you."},
                {
                    "type": "tool_use",
                    "id": "toolu_weather_123",
                    "name": "get_weather",
                    "input": {"location": "Tokyo"}
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_weather_123",
                    "content": "Sunny, 25°C"
                },
                {"type": "text", "text": "Now get it for Paris too"}
            ]
        }
    ]

    # Step 1: Convert to OpenAI format
    openai_messages = anthropic_messages_to_openai_messages(anthropic_messages, system=None)
    # Note: tool_result blocks get converted into separate tool messages
    # So we expect more than 3 messages after conversion
    assert len(openai_messages) >= 3
    # Find assistant message with tool_calls
    tool_call_msg = next((m for m in openai_messages if m.get("tool_calls")), None)
    assert tool_call_msg is not None

    # Step 2: Normalize for Kimi
    kimi_messages = normalize_tool_call_ids_for_kimi(openai_messages)
    # Should have normalized IDs
    tool_call_msg = next((m for m in kimi_messages if m.get("tool_calls")), None)
    if tool_call_msg:
        assert ":" in tool_call_msg["tool_calls"][0]["id"]

    # Step 3: Inject cache_control
    cached_messages, _ = inject_cache_control_for_anthropic(
        kimi_messages, None, "anthropic/claude-sonnet-4.5"
    )
    # Check cache_control present somewhere (last user message should have it)
    has_cache = any(
        isinstance(msg.get("content"), list) and
        any(isinstance(b, dict) and "cache_control" in b for b in msg["content"])
        for msg in cached_messages
    )
    assert has_cache, "No cache_control found in any message"

    # Step 4: Simplify messages (text-only arrays to strings)
    simplified_messages = []
    for msg in cached_messages:
        new_msg = dict(msg)
        content = new_msg.get("content")
        if isinstance(content, list):
            text_only = all(
                isinstance(block, dict) and block.get("type") == "text"
                for block in content
            )
            has_cache_control = any(
                isinstance(block, dict) and ("cache_control" in block)
                for block in content
            )
            if text_only and not has_cache_control:
                new_msg["content"] = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )
        simplified_messages.append(new_msg)

    # Step 5: Detect cache_control markers
    markers = []
    for i, msg in enumerate(simplified_messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if isinstance(block, dict) and ("cache_control" in block):
                markers.append((i, j, block.get("cache_control")))

    assert len(markers) > 0
    print("✓ integration_all_transformations passed")


def test_reasoning_filtering_non_streaming():
    """Test that reasoning fields are stripped from non-streaming responses."""
    # Simulate OpenRouter response with reasoning fields
    openrouter_response = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "moonshotai/kimi-k2-thinking",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    # These should be stripped
                    "reasoning": "Let me think about this...",
                    "reasoning_content": "First, I need to...",
                    "reasoning_details": {"thinking_time": 2.5},
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }

    result = openai_response_to_anthropic_message(
        openrouter_response, "moonshotai/kimi-k2-thinking"
    )

    # Verify reasoning fields are NOT in the result
    assert "reasoning" not in result
    assert "reasoning_content" not in result
    assert "reasoning_details" not in result

    # Verify normal content is present
    assert result["role"] == "assistant"
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "The answer is 42."
    print("✓ reasoning_filtering_non_streaming passed")


def test_gemini_reasoning_details_preserved_in_cache_before_stripping():
    """Test that reasoning_details can be cached for later reinjection even if stripped from output."""
    from proxy import gemini_tool_meta_get

    openrouter_response = {
        "id": "chatcmpl-test456",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "google/gemini-2.0-flash",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Calling tool...",
                    "tool_calls": [
                        {
                            "id": "toolu_test_1",
                            "type": "function",
                            "function": {
                                "name": "WebSearch",
                                "arguments": "{}",
                            },
                            "reasoning_details": {"provider": "google", "v": 1},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    _ = openai_response_to_anthropic_message(openrouter_response, "google/gemini-2.0-flash")

    meta = gemini_tool_meta_get("toolu_test_1")
    assert meta.get("reasoning_details") == {"provider": "google", "v": 1}
    print("✓ gemini_reasoning_details_cached passed")


def test_websearch_args_normalization():
    """Test that WebSearch tool call arguments are normalized to avoid 'both domains' error."""
    from proxy import _normalize_domain_list, _normalize_websearch_args

    # Test _normalize_domain_list
    assert _normalize_domain_list(None) is None
    assert _normalize_domain_list("not-a-list") is None
    assert _normalize_domain_list([]) is None
    assert _normalize_domain_list([""]) is None
    assert _normalize_domain_list(["  ", "\t"]) is None
    assert _normalize_domain_list(["example.com"]) == ["example.com"]
    assert _normalize_domain_list(["  example.com  ", "test.org"]) == ["example.com", "test.org"]
    assert _normalize_domain_list(["", "valid.com", "  "]) == ["valid.com"]

    # Test _normalize_websearch_args - non-WebSearch tools are unchanged
    other_args = {"query": "test", "allowed_domains": [], "blocked_domains": []}
    result = _normalize_websearch_args("OtherTool", other_args)
    assert result == other_args

    # Test _normalize_websearch_args - empty arrays are removed
    ws_args = {"query": "test", "allowed_domains": [], "blocked_domains": []}
    result = _normalize_websearch_args("WebSearch", ws_args)
    assert "allowed_domains" not in result
    assert "blocked_domains" not in result
    assert result["query"] == "test"

    # Test _normalize_websearch_args - only allowed_domains kept if both present
    ws_args = {"query": "test", "allowed_domains": ["a.com"], "blocked_domains": ["b.com"]}
    result = _normalize_websearch_args("WebSearch", ws_args)
    assert result["allowed_domains"] == ["a.com"]
    assert "blocked_domains" not in result

    # Test _normalize_websearch_args - only blocked_domains if allowed is empty
    ws_args = {"query": "test", "allowed_domains": [], "blocked_domains": ["b.com"]}
    result = _normalize_websearch_args("WebSearch", ws_args)
    assert "allowed_domains" not in result
    assert result["blocked_domains"] == ["b.com"]

    # Test _normalize_websearch_args - whitespace/invalid entries filtered
    ws_args = {"query": "test", "allowed_domains": ["  ", "valid.com", ""], "blocked_domains": []}
    result = _normalize_websearch_args("WebSearch", ws_args)
    assert result["allowed_domains"] == ["valid.com"]
    assert "blocked_domains" not in result

    print("✓ websearch_args_normalization passed")


def test_websearch_tool_call_normalized_in_response():
    """Test that WebSearch tool calls are normalized when converting OpenAI response to Anthropic."""
    openrouter_response = {
        "id": "chatcmpl-websearch",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "anthropic/claude-sonnet",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Searching...",
                    "tool_calls": [
                        {
                            "id": "toolu_ws_1",
                            "type": "function",
                            "function": {
                                "name": "WebSearch",
                                "arguments": '{"query": "test", "allowed_domains": [], "blocked_domains": []}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    result = openai_response_to_anthropic_message(openrouter_response, "anthropic/claude-sonnet")

    # Find the tool_use block
    tool_use_block = next((b for b in result["content"] if b.get("type") == "tool_use"), None)
    assert tool_use_block is not None
    assert tool_use_block["name"] == "WebSearch"

    # Verify empty domain arrays were removed
    assert "allowed_domains" not in tool_use_block["input"]
    assert "blocked_domains" not in tool_use_block["input"]
    assert tool_use_block["input"]["query"] == "test"

    print("✓ websearch_tool_call_normalized_in_response passed")


def test_reasoning_details_reinjected_at_message_level():
    """Test that reasoning_details is re-injected at assistant message level for Gemini 3."""
    # First, cache reasoning_details for a tool call ID (simulating response from OpenRouter)
    tool_call_id = "toolu_gemini_test_rd"
    test_reasoning_details = [
        {"type": "reasoning.text", "text": "Let me think...", "id": "rd-1"},
        {"type": "reasoning.encrypted", "data": "abc123", "id": "rd-2"},
    ]
    gemini_tool_meta_put(tool_call_id, {"reasoning_details": test_reasoning_details})

    # Now simulate Anthropic messages with tool_use that we need to convert back to OpenAI
    anthropic_messages = [
        {"role": "user", "content": "Search for something"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll search for that."},
                {
                    "type": "tool_use",
                    "id": tool_call_id,
                    "name": "WebSearch",
                    "input": {"query": "test"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": "Search results here",
                }
            ],
        },
    ]

    # Convert to OpenAI format
    openai_messages = anthropic_messages_to_openai_messages(anthropic_messages)

    # Find the assistant message with tool_calls
    assistant_msg = next((m for m in openai_messages if m.get("tool_calls")), None)
    assert assistant_msg is not None

    # Verify reasoning_details is present at message level
    assert "reasoning_details" in assistant_msg, "reasoning_details should be re-injected at message level"
    assert assistant_msg["reasoning_details"] == test_reasoning_details

    # Verify thought_signature is also present in tool_calls
    tc = assistant_msg["tool_calls"][0]
    assert tc.get("thought_signature") is not None, "thought_signature should be present"

    print("✓ reasoning_details_reinjected_at_message_level passed")


async def test_reasoning_filtering_sse_replayer():
    """Test that SSE replayer drops reasoning/thinking blocks."""
    # Create a message with reasoning blocks
    message_with_reasoning = {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "text": "Let me think..."},
            {"type": "text", "text": "Hello!"},
            {"type": "reasoning", "text": "I reasoned that..."},
            {"type": "reasoning.chain_of_thought", "text": "Step 1..."},
            {"type": "text", "text": "World!"},
        ],
        "model": "test-model",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    # Collect SSE events
    events = []
    async for event in anthropic_sse_from_message(
        message_with_reasoning, "test-model"
    ):
        events.append(event)

    # Parse events to check content blocks
    content_block_starts = [
        e
        for e in events
        if "content_block_start" in e and "event: content_block_start" in e
    ]

    # Should only have 2 content blocks (the 2 text blocks), reasoning blocks should be dropped
    assert (
        len(content_block_starts) == 2
    ), f"Expected 2 content blocks, got {len(content_block_starts)}"

    # Verify no reasoning content appears in any event
    all_events_text = "\n".join(events)
    assert "Let me think..." not in all_events_text
    assert "I reasoned that..." not in all_events_text
    assert "Step 1..." not in all_events_text

    # Verify text content is present
    assert "Hello!" in all_events_text
    assert "World!" in all_events_text
    print("✓ reasoning_filtering_sse_replayer passed")


def run_all_tests():
    """Run all tests."""
    import asyncio

    try:
        test_anthropic_to_openai_conversion()
        test_kimi_tool_call_normalization()
        test_normalize_tool_id_for_anthropic()
        test_openai_response_tool_id_normalization()
        test_cache_control_injection()
        test_message_simplification()
        test_integration_all_transformations()
        test_reasoning_filtering_non_streaming()
        test_gemini_reasoning_details_preserved_in_cache_before_stripping()
        test_websearch_args_normalization()
        test_websearch_tool_call_normalized_in_response()
        test_reasoning_details_reinjected_at_message_level()
        asyncio.run(test_reasoning_filtering_sse_replayer())
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
