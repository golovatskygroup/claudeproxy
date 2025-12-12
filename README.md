# Anthropic to OpenRouter Proxy

A lightweight proxy server that translates Anthropic Messages API requests to OpenRouter's OpenAI-compatible API, enabling you to use Claude Code and other Anthropic API clients with OpenRouter.

## Features

- Full Anthropic Messages API compatibility
- Real-time SSE streaming support
- Tool/function calling support
- Image content support (base64 and URL)
- Automatic model mapping (Opus, Sonnet, Haiku)
- Configurable timeout and model overrides
- Health check endpoint

## Requirements

- Python 3.8+
- OpenRouter API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd claudeproxy
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install fastapi httpx uvicorn
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | Your OpenRouter API key |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | OpenRouter API base URL |
| `TARGET_MODEL_DEFAULT` | No | `anthropic/claude-sonnet-4-20250514` | Default model for unknown requests |
| `TARGET_MODEL_SMALL` | No | `anthropic/claude-sonnet-4-20250514` | Model for Haiku requests |
| `TARGET_MODEL_BIG` | No | `anthropic/claude-sonnet-4-20250514` | Model for Sonnet requests |
| `TARGET_MODEL_OPUS` | No | `anthropic/claude-opus-4-20250514` | Model for Opus requests |
| `OPENROUTER_APP_URL` | No | - | Your app URL (for OpenRouter analytics) |
| `OPENROUTER_APP_TITLE` | No | - | Your app title (for OpenRouter analytics) |
| `TIMEOUT_S` | No | `300` | Request timeout in seconds |
| `PORT` | No | `8000` | Server port |
| `HOST` | No | `0.0.0.0` | Server host |

### Quick Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenRouter API key:
```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

## Running the Proxy

### Direct execution:
```bash
OPENROUTER_API_KEY="sk-or-v1-your-key" python3 proxy.py
```

### With .env file (using python-dotenv):
```bash
pip install python-dotenv
python3 -c "from dotenv import load_dotenv; load_dotenv()" && python3 proxy.py
```

### Using environment file directly:
```bash
export $(cat .env | xargs) && python3 proxy.py
```

The proxy will start on `http://0.0.0.0:8000` by default.

## Configuring Claude Code

To use this proxy with Claude Code, set the following environment variables before running Claude:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_API_KEY="any-value-here"  # The proxy ignores this, uses OPENROUTER_API_KEY

# Then run Claude Code
claude
```

Or add to your shell profile (~/.bashrc, ~/.zshrc):
```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_API_KEY="proxy"
```

## API Endpoints

### POST /v1/messages
Main Anthropic Messages API endpoint. Accepts Anthropic-format requests and returns Anthropic-format responses.

### GET /health
Health check endpoint. Returns `{"status": "ok"}` when the proxy is running.

### GET /
Root endpoint with API information.

## Example Requests

### Non-streaming request:
```bash
curl -X POST http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

### Streaming request:
```bash
curl --no-buffer -X POST http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4",
    "max_tokens": 1024,
    "stream": true,
    "messages": [
      {"role": "user", "content": "Tell me a short story."}
    ]
  }'
```

### Request with tools:
```bash
curl -X POST http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4",
    "max_tokens": 1024,
    "tools": [
      {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    ],
    "messages": [
      {"role": "user", "content": "What is the weather in Tokyo?"}
    ]
  }'
```

### Request with image (base64):
```bash
curl -X POST http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4",
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BASE64_DATA_HERE"}},
          {"type": "text", "text": "What is in this image?"}
        ]
      }
    ]
  }'
```

## Model Mapping

The proxy automatically maps Anthropic model names to OpenRouter models:

| Anthropic Model (contains) | OpenRouter Model |
|---------------------------|------------------|
| opus | `TARGET_MODEL_OPUS` (default: anthropic/claude-opus-4-20250514) |
| sonnet | `TARGET_MODEL_BIG` (default: anthropic/claude-sonnet-4-20250514) |
| haiku | `TARGET_MODEL_SMALL` (default: anthropic/claude-sonnet-4-20250514) |
| other | `TARGET_MODEL_DEFAULT` (default: anthropic/claude-sonnet-4-20250514) |

You can override these mappings using environment variables.

## Streaming Events

The proxy emits standard Anthropic SSE events:

1. `message_start` - Initial message metadata
2. `content_block_start` - Start of a content block (text or tool_use)
3. `content_block_delta` - Content delta (text_delta or input_json_delta)
4. `content_block_stop` - End of a content block
5. `message_delta` - Final message update (stop_reason, usage)
6. `message_stop` - End of message stream

## Error Handling

The proxy handles various error scenarios:

- **401 Unauthorized**: Invalid or missing OpenRouter API key
- **502 Bad Gateway**: Failed to connect to OpenRouter
- **504 Gateway Timeout**: Request timed out
- **SSE Errors**: Errors during streaming are sent as `event: error` SSE events

## Troubleshooting

### Proxy won't start
- Check that all dependencies are installed
- Verify OPENROUTER_API_KEY is set
- Check if port 8000 is already in use

### Authentication errors
- Verify your OpenRouter API key is valid
- Check that the key has sufficient credits

### Streaming not working
- Ensure your client supports SSE
- Check for proxy/firewall issues that might buffer responses

## License

MIT License
