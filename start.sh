#!/bin/bash

# Stop any running proxy
pkill -f "python3 proxy.py" 2>/dev/null
sleep 1

# Check if OPENROUTER_API_KEY is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY environment variable is not set"
    echo "Please set it with: export OPENROUTER_API_KEY=your-key"
    exit 1
fi

# Start proxy with GPT-5.2 for Sonnet/Opus, Claude Haiku for small
OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
TARGET_MODEL_DEFAULT="openai/gpt-5.2" \
TARGET_MODEL_SMALL="anthropic/claude-haiku-4.5" \
TARGET_MODEL_BIG="openai/gpt-5.2" \
TARGET_MODEL_OPUS="openai/gpt-5.2-pro" \
python3 proxy.py > proxy.log 2>&1 &

PID=$!
echo "Proxy started with PID $PID"
echo "Logs: tail -f proxy.log"

# Wait and check health
sleep 2
if curl -s http://127.0.0.1:8000/health | grep -q "ok"; then
    echo "✓ Proxy is running and healthy"
    echo ""
    echo "Configure Claude Code with:"
    echo "  export ANTHROPIC_BASE_URL=\"http://127.0.0.1:8000\""
    echo "  export ANTHROPIC_API_KEY=\"proxy\""
else
    echo "✗ Proxy failed to start. Check proxy.log for errors"
    exit 1
fi
