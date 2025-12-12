#!/usr/bin/env python3
import os
import httpx
import json

api_key = os.environ.get("OPENROUTER_API_KEY", "")
if not api_key:
    print("OPENROUTER_API_KEY not set")
    exit(1)

response = httpx.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

data = response.json()
models = data.get("data", [])

print("=== Claude models ===")
for m in models:
    mid = m.get("id", "")
    if "claude" in mid.lower():
        print(mid)

print("\n=== GPT models ===")
for m in models:
    mid = m.get("id", "")
    if "gpt" in mid.lower() and "5" in mid:
        print(mid)
