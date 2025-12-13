#!/usr/bin/env python3
"""
Anthropic Messages API → OpenRouter Proxy with REAL SSE Streaming.

This proxy translates Anthropic Messages API requests to OpenRouter's OpenAI-compatible API
and streams responses back in real-time using Server-Sent Events (SSE).
"""

import os
import json
import uuid
import logging
import time
import hmac
import asyncio
from datetime import datetime, timezone, timedelta
from typing import AsyncIterator, Optional, Any
from urllib.parse import urlparse

import httpx
import tiktoken
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track proxy startup time for uptime calculation
PROXY_START_TIME = time.time()
PROXY_VERSION = "1.0.0"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration from environment variables
# ─────────────────────────────────────────────────────────────────────────────

# API Key authentication (comma-separated list of allowed keys)
# Parsed once at startup for security and performance
_PROXY_API_KEYS_RAW = os.environ.get("PROXY_API_KEYS", "")
PROXY_API_KEYS_SET = frozenset(key.strip() for key in _PROXY_API_KEYS_RAW.split(",") if key.strip()) if _PROXY_API_KEYS_RAW else frozenset()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Model mapping
TARGET_MODEL_DEFAULT = os.environ.get("TARGET_MODEL_DEFAULT", "openai/gpt-5.2")
TARGET_MODEL_SMALL = os.environ.get("TARGET_MODEL_SMALL", "anthropic/claude-haiku-4.5")
TARGET_MODEL_BIG = os.environ.get("TARGET_MODEL_BIG", "openai/gpt-5.2")
TARGET_MODEL_OPUS = os.environ.get("TARGET_MODEL_OPUS", "openai/gpt-5.2-pro")

# Fallback configuration
FALLBACK_MODELS_SONNET = os.environ.get("FALLBACK_MODELS_SONNET", "")
FALLBACK_MODELS_OPUS = os.environ.get("FALLBACK_MODELS_OPUS", "")
FALLBACK_MODELS_HAIKU = os.environ.get("FALLBACK_MODELS_HAIKU", "")

# Parse FALLBACK_ON_STATUS: ignore non-int tokens and validate range 100-599
FALLBACK_ON_STATUS_RAW = os.environ.get("FALLBACK_ON_STATUS", "429,500,502,503,504")
_fallback_status_set = set()
for token in FALLBACK_ON_STATUS_RAW.split(","):
    token = token.strip()
    if not token:
        continue
    try:
        status_code = int(token)
        if 100 <= status_code <= 599:
            _fallback_status_set.add(status_code)
        else:
            logger.warning(f"FALLBACK_ON_STATUS: ignoring out-of-range status code {status_code} (valid: 100-599)")
    except ValueError:
        logger.warning(f"FALLBACK_ON_STATUS: ignoring non-integer token '{token}'")
FALLBACK_ON_STATUS = frozenset(_fallback_status_set)

# Parse FALLBACK_MAX_TRIES: clamp to min 1, max 10
_fallback_max_tries_raw = os.environ.get("FALLBACK_MAX_TRIES", "3")
try:
    _fallback_max_tries = int(_fallback_max_tries_raw)
    if _fallback_max_tries < 1:
        logger.warning(f"FALLBACK_MAX_TRIES={_fallback_max_tries} is below minimum (1), clamping to 1")
        _fallback_max_tries = 1
    elif _fallback_max_tries > 10:
        logger.warning(f"FALLBACK_MAX_TRIES={_fallback_max_tries} exceeds maximum (10), clamping to 10")
        _fallback_max_tries = 10
except ValueError:
    logger.warning(f"FALLBACK_MAX_TRIES: invalid value '{_fallback_max_tries_raw}', using default 3")
    _fallback_max_tries = 3
FALLBACK_MAX_TRIES = _fallback_max_tries

# Optional OpenRouter headers
OPENROUTER_APP_URL = os.environ.get("OPENROUTER_APP_URL", "")
OPENROUTER_APP_TITLE = os.environ.get("OPENROUTER_APP_TITLE", "")

# Timeout in seconds
TIMEOUT_S = int(os.environ.get("TIMEOUT_S", "300"))

# OpenRouter web search (disabled by default, use search: prefix or set env var)
OPENROUTER_WEB_ENABLED = os.environ.get("OPENROUTER_WEB_ENABLED", "false").lower() == "true"
OPENROUTER_WEB_ENGINE = os.environ.get("OPENROUTER_WEB_ENGINE", "")  # e.g. "exa" or "native"
OPENROUTER_WEB_MAX_RESULTS = os.environ.get("OPENROUTER_WEB_MAX_RESULTS", "")  # int
OPENROUTER_WEB_SEARCH_PROMPT = os.environ.get("OPENROUTER_WEB_SEARCH_PROMPT", "")
OPENROUTER_WEB_SEARCH_CONTEXT_SIZE = os.environ.get("OPENROUTER_WEB_SEARCH_CONTEXT_SIZE", "")  # low|medium|high

# Proxy-side WebSearch tool (Exa) — intercepts Claude Code WebSearch tool calls.
# If enabled and EXA_API_KEY is set, the proxy will execute WebSearch calls itself
# and hide the tool loop from the client.
PROXY_WEBSEARCH_ENABLED = os.environ.get("PROXY_WEBSEARCH_ENABLED", "true").lower() == "true"
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
EXA_BASE_URL = os.environ.get("EXA_BASE_URL", "https://api.exa.ai")
EXA_MAX_RESULTS = int(os.environ.get("EXA_MAX_RESULTS", "5"))
EXA_USE_AUTOPROMPT = os.environ.get("EXA_USE_AUTOPROMPT", "true").lower() == "true"
EXA_SEARCH_TYPE = os.environ.get("EXA_SEARCH_TYPE", "auto")  # auto|neural|keyword|fast
EXA_TEXT_MAX_CHARS = int(os.environ.get("EXA_TEXT_MAX_CHARS", "2000"))
PROXY_TOOLS_MAX_ITERS = int(os.environ.get("PROXY_TOOLS_MAX_ITERS", "3"))

# Debug dump configuration
DEBUG_DUMP_REQUESTS = os.environ.get("DEBUG_DUMP_REQUESTS", "false").lower() == "true"

# Cache marker debugging (logs only cache_control positions, never full text)
DEBUG_CACHE_MARKERS = os.environ.get("DEBUG_CACHE_MARKERS", "false").lower() == "true"

# Admin API configuration (Feature E)
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────────────
# Quota Configuration (Feature D)
# ─────────────────────────────────────────────────────────────────────────────

QUOTA_CONFIG_JSON = os.environ.get("QUOTA_CONFIG_JSON", "")
QUOTA_CONFIG_FILE = os.environ.get("QUOTA_CONFIG_FILE", "")
QUOTA_STATE_FILE = os.environ.get("QUOTA_STATE_FILE", "./quota_state.json")

# Load quota configuration
QUOTA_CONFIG: dict[str, int] = {}
if QUOTA_CONFIG_FILE:
    # File takes precedence
    try:
        with open(QUOTA_CONFIG_FILE, "r") as f:
            QUOTA_CONFIG = json.load(f)
        logger.info(f"Loaded quota config from file: {len(QUOTA_CONFIG)} keys configured")
    except FileNotFoundError:
        logger.warning(f"QUOTA_CONFIG_FILE not found: {QUOTA_CONFIG_FILE}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in QUOTA_CONFIG_FILE: {e}")
    except Exception as e:
        logger.error(f"Error loading QUOTA_CONFIG_FILE: {e}")
elif QUOTA_CONFIG_JSON:
    # Parse JSON string
    try:
        QUOTA_CONFIG = json.loads(QUOTA_CONFIG_JSON)
        logger.info(f"Loaded quota config from JSON string: {len(QUOTA_CONFIG)} keys configured")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid QUOTA_CONFIG_JSON: {e}")
    except Exception as e:
        logger.error(f"Error parsing QUOTA_CONFIG_JSON: {e}")

# Quota state: {api_key: {"used_tokens": int, "reset_at": str}}
quota_state: dict[str, dict[str, Any]] = {}
quota_state_lock = asyncio.Lock()

# Load quota state from file
def load_quota_state():
    """Load quota state from file."""
    global quota_state
    try:
        with open(QUOTA_STATE_FILE, "r") as f:
            quota_state = json.load(f)
        logger.info(f"Loaded quota state from {QUOTA_STATE_FILE}: {len(quota_state)} keys")
    except FileNotFoundError:
        logger.info(f"No existing quota state file at {QUOTA_STATE_FILE}, starting fresh")
        quota_state = {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in quota state file: {e}, starting fresh")
        quota_state = {}
    except Exception as e:
        logger.error(f"Error loading quota state: {e}, starting fresh")
        quota_state = {}

def save_quota_state():
    """Save quota state to file atomically."""
    try:
        # Write to temp file in same directory for atomic replacement
        temp_fd, temp_path = None, None
        try:
            import tempfile
            state_dir = os.path.dirname(QUOTA_STATE_FILE) or "."
            temp_fd, temp_path = tempfile.mkstemp(dir=state_dir, prefix=".quota_state_", suffix=".json.tmp")

            # Write JSON data to temp file
            with os.fdopen(temp_fd, "w") as f:
                temp_fd = None  # fdopen takes ownership
                json.dump(quota_state, f, indent=2)

            # Set restrictive permissions (best-effort)
            try:
                os.chmod(temp_path, 0o600)
            except Exception as e:
                logger.warning(f"Failed to chmod quota state file: {e}")

            # Atomic replace
            os.replace(temp_path, QUOTA_STATE_FILE)
            temp_path = None  # Successful, no cleanup needed

        finally:
            # Cleanup on failure
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except Exception:
                    pass
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    except Exception as e:
        logger.error(f"Error saving quota state: {e}")

def get_next_utc_midnight() -> str:
    """Get next UTC midnight as ISO string."""
    now = datetime.now(timezone.utc)
    tomorrow = now + timedelta(days=1)
    midnight = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.isoformat()

async def check_and_reserve_quota(api_key: str, estimated_tokens: int, request_id: str) -> None:
    """
    Check if quota allows this request and reserve tokens.

    Raises HTTPException with status 429 if quota exceeded.
    """
    if not QUOTA_CONFIG:
        # No quota configured, allow all requests
        return

    if api_key not in QUOTA_CONFIG:
        # This API key has no quota limit
        return

    daily_limit = QUOTA_CONFIG[api_key]

    async with quota_state_lock:
        # Get or initialize state for this key
        if api_key not in quota_state:
            quota_state[api_key] = {
                "used_tokens": 0,
                "reset_at": get_next_utc_midnight()
            }

        state = quota_state[api_key]

        # Check if we need to reset (new day)
        reset_at = datetime.fromisoformat(state["reset_at"])
        now = datetime.now(timezone.utc)

        if now >= reset_at:
            # Reset quota for new day
            state["used_tokens"] = 0
            state["reset_at"] = get_next_utc_midnight()
            logger.info(f"Reset quota for key (masked) | request_id={request_id}")

        # Check if quota would be exceeded
        used_tokens = state["used_tokens"]
        if used_tokens + estimated_tokens > daily_limit:
            # Quota exceeded
            logger.warning(
                f"Quota exceeded | used={used_tokens} estimate={estimated_tokens} "
                f"limit={daily_limit} request_id={request_id}"
            )
            raise HTTPException(
                status_code=429,
                detail=f"Daily token quota exceeded. Used: {used_tokens}, Limit: {daily_limit}, Resets at: {state['reset_at']}"
            )

        # Reserve tokens
        state["used_tokens"] += estimated_tokens

        # Save quota state (best-effort, don't fail the request)
        try:
            save_quota_state()
        except Exception as e:
            logger.error(f"Failed to save quota state after reservation: {e}")

        logger.info(
            f"Reserved {estimated_tokens} tokens | used={state['used_tokens']}/{daily_limit} "
            f"request_id={request_id}"
        )

async def reconcile_quota(api_key: str, estimated_tokens: int, actual_tokens: int, request_id: str) -> None:
    """
    Reconcile quota after actual token usage is known.

    Adjusts the quota by the difference between estimated and actual usage.
    """
    if not QUOTA_CONFIG or api_key not in QUOTA_CONFIG:
        return

    async with quota_state_lock:
        if api_key not in quota_state:
            return

        state = quota_state[api_key]
        adjustment = actual_tokens - estimated_tokens
        state["used_tokens"] += adjustment

        # Ensure used_tokens doesn't go negative
        if state["used_tokens"] < 0:
            state["used_tokens"] = 0

        # Save quota state (best-effort, don't fail the request)
        try:
            save_quota_state()
        except Exception as e:
            logger.error(f"Failed to save quota state after reconciliation: {e}")

        # Warn on anomalous adjustments (potential parsing issues or attacks)
        daily_limit = QUOTA_CONFIG.get(api_key, 0)
        if daily_limit > 0:
            adjustment_ratio = abs(adjustment) / daily_limit if daily_limit > 0 else 0
            # Warn if adjustment is more than 10% of daily limit or very large absolute value
            if adjustment_ratio > 0.1 or abs(adjustment) > 100000:
                logger.warning(
                    f"Large quota reconcile adjustment detected | "
                    f"estimated={estimated_tokens} actual={actual_tokens} "
                    f"adjustment={adjustment:+d} ({adjustment_ratio*100:.1f}% of limit) "
                    f"request_id={request_id}"
                )

        logger.info(
            f"Reconciled quota | estimated={estimated_tokens} actual={actual_tokens} "
            f"adjustment={adjustment:+d} new_used={state['used_tokens']} request_id={request_id}"
        )

def mask_api_key(api_key: str) -> str:
    """Mask API key for display (show first 8 chars)."""
    if len(api_key) <= 8:
        return "***"
    return api_key[:8] + "..." + ("*" * (len(api_key) - 8))

def check_admin_key(request: Request) -> None:
    """
    Check X-Admin-Key header for admin endpoints (Feature E).

    Raises HTTPException with status 401 if admin key is invalid or missing.
    Should be called after normal X-API-Key authentication in admin endpoints.
    """
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Admin API not configured (ADMIN_API_KEY not set)"
        )

    admin_key = request.headers.get("X-Admin-Key")
    if not admin_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Admin-Key header"
        )

    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(admin_key, ADMIN_API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid X-Admin-Key"
        )

# Load quota state at startup
if QUOTA_CONFIG:
    load_quota_state()
    logger.info(f"Quota system enabled for {len(QUOTA_CONFIG)} API keys")

# ─────────────────────────────────────────────────────────────────────────────
# Prometheus Metrics
# ─────────────────────────────────────────────────────────────────────────────

# Request counter by endpoint and status
request_counter = Counter(
    'proxy_requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)

# Request latency histogram by endpoint
request_latency = Histogram(
    'proxy_request_duration_seconds',
    'Request latency in seconds',
    ['endpoint']
)

# Error counter by endpoint
error_counter = Counter(
    'proxy_errors_total',
    'Total number of errors',
    ['endpoint', 'error_type']
)

# ─────────────────────────────────────────────────────────────────────────────
# Models cache (to avoid slow API calls on every /v1/models request)
# ─────────────────────────────────────────────────────────────────────────────

MODELS_CACHE = None
MODELS_CACHE_TIMESTAMP = 0
MODELS_CACHE_TTL = 300  # 5 minutes

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Anthropic to OpenRouter Proxy", version="1.0.0")


# ─────────────────────────────────────────────────────────────────────────────
# API Key Authentication Middleware
# ─────────────────────────────────────────────────────────────────────────────

class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Fail-closed authentication middleware for the proxy.

    Security model:
    - ONLY /health is PUBLIC (no auth required)
    - ALL other endpoints (including /metrics) require X-API-Key header matching PROXY_API_KEYS
    - If PROXY_API_KEYS is empty/unset, returns 500 error (fail-closed)
    - Uses constant-time comparison to prevent timing attacks
    - Logs failed authentication attempts with path and client IP
    """

    async def dispatch(self, request: Request, call_next):
        # Generate request_id early for error responses
        request_id = get_or_generate_request_id(request)
        request.state.request_id = request_id

        # Allow ONLY /health endpoint without authentication (public endpoint)
        if request.url.path == "/health":
            response = await call_next(request)
            response.headers.setdefault("X-Request-Id", request_id)
            return response

        # FAIL-CLOSED: If PROXY_API_KEYS is not configured, reject all requests
        # This prevents accidentally running an open proxy
        if not PROXY_API_KEYS_SET:
            logger.error(
                f"Authentication failed: PROXY_API_KEYS not configured | "
                f"path={request.url.path} client_ip={request.client.host if request.client else 'unknown'} "
                f"request_id={request_id}"
            )
            return JSONResponse(
                status_code=500,
                content={
                    "type": "error",
                    "error": {
                        "type": "configuration_error",
                        "message": "Server authentication is not configured. Please contact the administrator."
                    }
                },
                headers={"X-Request-Id": request_id}
            )

        # Get X-API-Key header
        # Note: Starlette normalizes header names to lowercase internally, but provides
        # case-insensitive access via request.headers.get(), so "X-API-Key" works regardless
        # of how the client sends it (x-api-key, X-Api-Key, etc.)
        api_key = request.headers.get("X-API-Key")

        # Check if API key is provided
        if not api_key:
            logger.warning(
                f"Authentication failed: Missing X-API-Key header | "
                f"path={request.url.path} client_ip={request.client.host if request.client else 'unknown'} "
                f"request_id={request_id}"
            )
            return JSONResponse(
                status_code=401,
                content={
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "Missing X-API-Key header"
                    }
                },
                headers={"X-Request-Id": request_id}
            )

        # Validate API key using constant-time comparison to prevent timing attacks
        # Check ALL keys without early exit to ensure constant-time operation across the key set
        # (prevents leaking the number of keys or their position via timing side-channels)
        matches = [
            hmac.compare_digest(api_key, valid_key)
            for valid_key in PROXY_API_KEYS_SET
        ]
        is_valid = any(matches)

        if not is_valid:
            logger.warning(
                f"Authentication failed: Invalid X-API-Key | "
                f"path={request.url.path} client_ip={request.client.host if request.client else 'unknown'} "
                f"request_id={request_id}"
            )
            return JSONResponse(
                status_code=401,
                content={
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "Invalid X-API-Key"
                    }
                },
                headers={"X-Request-Id": request_id}
            )

        # API key is valid, store it in request.state for quota tracking
        request.state.api_key = api_key

        # Proceed with request
        return await call_next(request)


# Add authentication middleware
app.add_middleware(APIKeyAuthMiddleware)


# ─────────────────────────────────────────────────────────────────────────────
# HTTPException handler - Attach X-Request-Id to error responses
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Global HTTPException handler that ensures X-Request-Id header is always included.
    """
    # Get request_id from request.state if set by middleware, otherwise generate new one
    request_id = getattr(request.state, "request_id", None) or get_or_generate_request_id(request)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "error",
            "error": {
                "type": "api_error",
                "message": exc.detail
            }
        },
        headers={"X-Request-Id": request_id}
    )


# Log startup configuration (security info)
if PROXY_API_KEYS_SET:
    logger.info(f"Proxy authentication enabled: {len(PROXY_API_KEYS_SET)} API key(s) loaded")
else:
    logger.warning("PROXY_API_KEYS not configured - all requests except /health will be rejected (fail-closed mode)")


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def get_or_generate_request_id(request: Request) -> str:
    """Get X-Request-Id from request headers or generate a new one."""
    request_id = request.headers.get("X-Request-Id")
    if not request_id:
        request_id = f"req_{uuid.uuid4().hex}"
    return request_id


def redact_request_for_debug(request_dict: dict) -> dict:
    """Redact sensitive information from request for debug dumping."""
    redacted = json.loads(json.dumps(request_dict))  # Deep copy

    # Redact message content - replace with length info
    if "messages" in redacted and isinstance(redacted["messages"], list):
        for msg in redacted["messages"]:
            if isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
                if isinstance(content, str):
                    msg["content"] = f"<redacted: {len(content)} chars>"
                elif isinstance(content, list):
                    msg["content"] = f"<redacted: {len(content)} blocks>"

    return redacted


def redact_headers_for_debug(headers: dict) -> dict:
    """Redact sensitive headers (Authorization, API keys) for debug dumping."""
    redacted = dict(headers)
    sensitive_keys = ["authorization", "x-api-key", "http-referer", "referer"]

    for key in list(redacted.keys()):
        if key.lower() in sensitive_keys:
            redacted[key] = "<redacted>"

    return redacted


def _strip_claude_code_suffixes(model: str) -> str:
    """Strip Claude Code-specific suffixes like `[1m]` from the model string."""
    if not isinstance(model, str):
        return model
    # Claude Code can append bracket suffixes, e.g. "... [1m]".
    # OpenRouter model slugs do not accept these.
    if "[" in model and model.endswith("]"):
        return model[: model.rfind("[")].rstrip()
    return model


def convert_provider_to_openrouter_format(provider: Optional[dict]) -> Optional[dict]:
    """Ensure provider object uses correct snake_case format for OpenRouter API.

    OpenRouter uses snake_case for all keys in the provider object.
    This function just validates and passes through the dict as-is.
    """
    if not provider or not isinstance(provider, dict):
        return provider

    # OpenRouter expects snake_case, so we just return as-is
    return provider


def inject_cache_control_for_anthropic(
    messages: list,
    tools: Optional[list],
    target_model: str,
) -> tuple[list, Optional[list]]:
    """
    Inject cache_control breakpoints for Anthropic models on OpenRouter.

    OpenRouter requires explicit cache_control markers for Anthropic prompt caching.
    This function adds cache_control to:
    1. System message content (if present)
    2. Last 2 user messages
    3. Last tool definition

    Args:
        messages: List of OpenAI-format messages
        tools: List of OpenAI-format tools (or None)
        target_model: The target model slug

    Returns:
        tuple: (modified_messages, modified_tools)
    """
    # Only apply to Anthropic models
    if not target_model or "anthropic/" not in target_model.lower():
        return messages, tools

    # Deep copy messages to avoid mutating the original
    import copy
    messages = copy.deepcopy(messages)

    # Helper to wrap string content with cache_control
    def wrap_with_cache_control(content: Any) -> list:
        """Convert content to array format with cache_control on last block."""
        if isinstance(content, str):
            return [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"}
            }]
        elif isinstance(content, list):
            # Content is already an array, add cache_control to last text block
            if content:
                for i in range(len(content) - 1, -1, -1):
                    block = content[i]
                    if isinstance(block, dict) and block.get("type") == "text":
                        block["cache_control"] = {"type": "ephemeral"}
                        break
            return content
        return content

    # 1. Add cache_control to system message
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content")
            if content:
                msg["content"] = wrap_with_cache_control(content)
            break

    # 2. Add cache_control to last 2 user messages
    user_count = 0
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if content:
                msg["content"] = wrap_with_cache_control(content)
            user_count += 1
            if user_count >= 2:
                break

    # 3. Add cache_control to last tool definition
    if tools:
        tools = copy.deepcopy(tools)
        if tools:
            last_tool = tools[-1]
            if isinstance(last_tool, dict):
                # OpenAI format: {"type": "function", "function": {...}}
                func = last_tool.get("function")
                if isinstance(func, dict):
                    func["cache_control"] = {"type": "ephemeral"}

    return messages, tools


def get_fallback_models_for_category(anthropic_model: str) -> list[str]:
    """Get fallback models based on the model category (sonnet/opus/haiku).

    Returns a list of fallback models (without the primary model).
    Empty list means no fallbacks configured.
    """
    model_lower = anthropic_model.lower()

    # Determine category
    if "opus" in model_lower:
        fallback_str = FALLBACK_MODELS_OPUS
    elif "sonnet" in model_lower:
        fallback_str = FALLBACK_MODELS_SONNET
    elif "haiku" in model_lower:
        fallback_str = FALLBACK_MODELS_HAIKU
    else:
        # No category match, no fallbacks
        return []

    if not fallback_str:
        return []

    # Parse comma-separated list
    return [m.strip() for m in fallback_str.split(",") if m.strip()]


def pick_target_model(anthropic_model: str) -> tuple[str, Optional[dict], bool]:
    """Map incoming Anthropic model to OpenRouter model identifier.

    Supports multiple modes:
    - Web search prefix: if model starts with "search:", enable web search for this request
    - Pass-through: if `anthropic_model` looks like an OpenRouter slug (contains "/"),
      send it as-is (minus Claude Code suffixes).
    - Provider specification: if model contains ":provider" or ":provider/quant",
      extract the provider and return it separately.
    - Keyword mapping: opus/sonnet/haiku → TARGET_MODEL_* env vars.

    Returns:
        tuple: (model_name, provider_dict, enable_web_search) where provider_dict is None or contains routing info
    """
    anthropic_model = _strip_claude_code_suffixes(anthropic_model)
    provider_info = None
    enable_web_search = False

    # Check for search: prefix
    if anthropic_model.startswith("search:"):
        enable_web_search = True
        anthropic_model = anthropic_model[7:]  # Remove "search:" prefix

    # Check for provider specification in model name (e.g., "model:provider/quant" or "model:variant")
    if ":" in anthropic_model:
        base_model, suffix = anthropic_model.split(":", 1)

        # Check if suffix contains a slash (provider/quantization)
        if "/" in suffix:
            # Format: model:provider/quantization
            # The entire suffix "provider/quantization" is the provider slug in OpenRouter
            # Example: "gmicloud/fp8" is the complete provider slug, not "gmicloud" + quantization "fp8"
            provider_info = {
                "order": [suffix],  # Use the full provider slug: "gmicloud/fp8"
                "allow_fallbacks": False  # Disable fallbacks when specific provider requested
            }
            anthropic_model = base_model
        elif suffix in ["thinking", "nitro", "floor", "extended", "online"]:
            # These are valid OpenRouter suffixes, keep them in the model name
            # :thinking - reasoning mode
            # :nitro - high throughput
            # :floor - low cost
            # :extended - extended context
            # :online - web search (but deprecated, use search: prefix instead)
            if suffix == "online":
                enable_web_search = True
                # Remove :online suffix as it's not a real model variant
                pass
            pass
        else:
            # Other suffixes might be provider names
            provider_info = {
                "order": [suffix],
                "allow_fallbacks": False  # Disable fallbacks when specific provider requested
            }
            anthropic_model = base_model

    # If model contains "/" (OpenRouter format), return as-is
    if "/" in anthropic_model:
        return anthropic_model, provider_info, enable_web_search

    model_lower = anthropic_model.lower()

    # Opus models
    if "opus" in model_lower:
        return TARGET_MODEL_OPUS, provider_info, enable_web_search

    # Sonnet models (big)
    if "sonnet" in model_lower:
        return TARGET_MODEL_BIG, provider_info, enable_web_search

    # Haiku models (small)
    if "haiku" in model_lower:
        return TARGET_MODEL_SMALL, provider_info, enable_web_search

    # Default fallback
    return TARGET_MODEL_DEFAULT, provider_info, enable_web_search


def join_text_blocks(content: list) -> str:
    """Join text blocks from an Anthropic content array into a single string."""
    if not content:
        return ""

    texts = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))

    return "".join(texts)


def _tiktoken_for_model(model: str) -> tiktoken.Encoding:
    """Best-effort tokenizer selection for Claude Code token counting.

    We may receive OpenRouter slugs like `openai/gpt-4o`, optionally with `:online`
    or Claude Code suffixes like `[1m]`.
    """
    model = _strip_claude_code_suffixes(model)
    if isinstance(model, str):
        model = model.split(":", 1)[0]
        if "/" in model:
            model = model.split("/", 1)[1]

    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        # Reasonable default for modern chat models.
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")


def _count_tokens_openai_messages(model: str, messages: list) -> int:
    enc = _tiktoken_for_model(model)
    # Simple approximation: sum tokens of role + content (+ tool JSON if present).
    # Claude Code mainly needs a gating number; provider billing may differ.
    total = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        total += len(enc.encode(str(m.get("role", ""))))
        content = m.get("content", "")
        if isinstance(content, list):
            total += len(enc.encode(json.dumps(content, ensure_ascii=False)))
        else:
            total += len(enc.encode(str(content)))
        if "tool_calls" in m:
            total += len(enc.encode(json.dumps(m.get("tool_calls"), ensure_ascii=False)))
    return total


def _count_tokens_tools(model: str, tools: Optional[list]) -> int:
    if not tools:
        return 0
    enc = _tiktoken_for_model(model)
    return len(enc.encode(json.dumps(tools, ensure_ascii=False)))


def count_anthropic_request_tokens(body: dict) -> int:
    anthropic_model = body.get("model", "")
    system = body.get("system")
    messages = body.get("messages", [])
    tools = body.get("tools")

    openai_messages = anthropic_messages_to_openai_messages(messages, system)

    # Use the *target* model slug for tokenizer selection if possible.
    target_model, _, _ = pick_target_model(anthropic_model)

    return (
        _count_tokens_openai_messages(target_model, openai_messages)
        + _count_tokens_tools(target_model, tools_anthropic_to_openai(tools))
    )


def anthropic_content_to_openai_user_content(content: Any) -> Any:
    """
    Convert Anthropic content format to OpenAI content format.

    Anthropic uses: [{"type": "text", "text": "..."}, {"type": "image", "source": {...}}]
    OpenAI uses: [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {...}}]
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return content

    result = []
    for block in content:
        if isinstance(block, str):
            result.append({"type": "text", "text": block})
        elif isinstance(block, dict):
            block_type = block.get("type")

            if block_type == "text":
                out_block = {"type": "text", "text": block.get("text", "")}
                # Preserve Anthropic extras like cache_control for prompt caching.
                if "cache_control" in block:
                    out_block["cache_control"] = block["cache_control"]
                result.append(out_block)

            elif block_type == "image":
                # Convert Anthropic image format to OpenAI format
                source = block.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    result.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{data}"
                        }
                    })
                elif source.get("type") == "url":
                    result.append({
                        "type": "image_url",
                        "image_url": {
                            "url": source.get("url", "")
                        }
                    })

            elif block_type == "tool_result":
                # Tool results are handled separately in message conversion
                result.append(block)

            elif block_type == "tool_use":
                # Tool use blocks are handled separately
                result.append(block)

            else:
                # Pass through unknown types
                result.append(block)

    return result


def tools_anthropic_to_openai(tools: Optional[list]) -> Optional[list]:
    """
    Convert Anthropic tools format to OpenAI tools format.

    Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    if not tools:
        return None

    openai_tools = []
    seen_names = set()

    for tool in tools:
        tool_name = tool.get("name", "")

        # Skip duplicate tool names
        if tool_name in seen_names:
            logger.warning(f"Skipping duplicate tool name: {tool_name}")
            continue

        seen_names.add(tool_name)

        input_schema = tool.get("input_schema", {})
        if isinstance(input_schema, dict) and "type" not in input_schema:
            input_schema = {"type": "object", **input_schema}

        openai_tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool.get("description", ""),
                "parameters": input_schema
            }
        }
        openai_tools.append(openai_tool)

    return openai_tools


def tool_choice_anthropic_to_openai(tool_choice: Optional[Any]) -> Optional[Any]:
    """
    Convert Anthropic tool_choice to OpenAI tool_choice.

    Anthropic: {"type": "auto"}, {"type": "any"}, {"type": "tool", "name": "..."}
    OpenAI: "auto", "required", {"type": "function", "function": {"name": "..."}}
    """
    if tool_choice is None:
        return None

    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")

        if choice_type == "auto":
            return "auto"
        elif choice_type == "any":
            # Anthropic "any" means tools are available; do not force a tool call.
            return "auto"
        elif choice_type == "tool":
            return {
                "type": "function",
                "function": {"name": tool_choice.get("name", "")}
            }

    return tool_choice


def anthropic_messages_to_openai_messages(
    messages: list,
    system: Optional[str] = None
) -> list:
    """
    Convert Anthropic messages format to OpenAI messages format.
    """
    openai_messages = []

    # Add system message if present
    if system:
        if isinstance(system, str):
            openai_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Anthropic allows system as list of content blocks.
            # Preserve block structure so OpenRouter prompt caching breakpoints
            # (cache_control) can survive.
            openai_messages.append({
                "role": "system",
                "content": anthropic_content_to_openai_user_content(system)
            })

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "user":
            # Handle user messages with potential tool results
            if isinstance(content, list):
                # Check for tool_result blocks
                tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                other_content = [b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_result")]

                # First add any non-tool-result content as user message
                if other_content:
                    openai_messages.append({
                        "role": "user",
                        "content": anthropic_content_to_openai_user_content(other_content)
                    })

                # Then add tool results as tool messages
                for tr in tool_results:
                    tool_content = tr.get("content", "")
                    if isinstance(tool_content, list):
                        # Prefer preserving structured content; OpenAI tool messages accept
                        # either string content or (for newer APIs) content arrays.
                        tool_content = anthropic_content_to_openai_user_content(tool_content)

                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tool_content if isinstance(tool_content, (str, list)) else str(tool_content)
                    })
            else:
                openai_messages.append({
                    "role": "user",
                    "content": anthropic_content_to_openai_user_content(content)
                })

        elif role == "assistant":
            # Handle assistant messages with potential tool use
            if isinstance(content, list):
                tool_calls = []
                text_parts = []

                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {}))
                                }
                            })
                    elif isinstance(block, str):
                        text_parts.append(block)

                assistant_msg = {"role": "assistant"}
                if text_parts:
                    assistant_msg["content"] = "".join(text_parts)
                else:
                    # OpenAI-compatible: content should be a string, not null.
                    assistant_msg["content"] = ""

                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                openai_messages.append(assistant_msg)
            else:
                openai_messages.append({
                    "role": "assistant",
                    "content": content
                })

    return openai_messages


def _domain_matches(host: str, domain: str) -> bool:
    host = (host or "").lower()
    domain = (domain or "").lower().lstrip(".")
    return host == domain or host.endswith("." + domain)


async def exa_search_and_contents(
    client: httpx.AsyncClient,
    query: str,
    allowed_domains: Optional[list[str]] = None,
    blocked_domains: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Execute a web search using Exa and return normalized results.
    """
    if not EXA_API_KEY:
        raise RuntimeError("EXA_API_KEY is not set")

    max_results = max(1, EXA_MAX_RESULTS)
    payload: dict[str, Any] = {
        "query": query,
        "num_results": max_results,
        "use_autoprompt": EXA_USE_AUTOPROMPT,
        "type": EXA_SEARCH_TYPE,
        "text": True,
    }
    if allowed_domains:
        payload["include_domains"] = allowed_domains
    if blocked_domains:
        payload["exclude_domains"] = blocked_domains

    url = f"{EXA_BASE_URL.rstrip('/')}/search_and_contents"
    headers = {"Authorization": f"Bearer {EXA_API_KEY}", "Content-Type": "application/json"}
    resp = await client.post(url, headers=headers, json=payload, timeout=TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()

    items = data.get("results") or data.get("data") or []
    if isinstance(items, dict) and "results" in items:
        items = items["results"]
    if not isinstance(items, list):
        items = []

    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_url = item.get("url") or item.get("link")
        if not item_url:
            continue

        host = urlparse(item_url).hostname or ""
        if allowed_domains and not any(_domain_matches(host, d) for d in allowed_domains):
            continue
        if blocked_domains and any(_domain_matches(host, d) for d in blocked_domains):
            continue

        title = item.get("title") or item.get("name") or item_url
        text = item.get("text") or ""
        highlights = item.get("highlights")
        if (not text) and isinstance(highlights, list):
            text = "\n".join(str(h) for h in highlights if h)
        if text and len(text) > EXA_TEXT_MAX_CHARS:
            text = text[:EXA_TEXT_MAX_CHARS] + "…"

        page_age = (
            item.get("published_date")
            or item.get("publishedDate")
            or item.get("page_age")
            or item.get("pageAge")
        )

        normalized.append(
            {
                "url": item_url,
                "title": title,
                "text": text,
                "page_age": page_age,
            }
        )
        if len(normalized) >= max_results:
            break

    return normalized


def build_websearch_tool_result(tool_use_id: str, results: list[dict[str, Any]]) -> str:
    """
    Build a Claude Code-compatible WebSearch tool result.

    Claude Code expects a block shaped like:
    {
      "type": "web_search_tool_result",
      "tool_use_id": "...",
      "content": [{ "type": "web_search_result", "url": "...", "title": "...", "encrypted_content": "...", "page_age": "..." }]
    }
    """
    content_blocks: list[dict[str, Any]] = []
    for r in results:
        block: dict[str, Any] = {
            "type": "web_search_result",
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            # We cannot produce Anthropic-encrypted blobs; include plain text instead.
            "encrypted_content": r.get("text", ""),
        }
        if r.get("page_age"):
            block["page_age"] = r["page_age"]
        content_blocks.append(block)

    tool_result = {
        "type": "web_search_tool_result",
        "tool_use_id": tool_use_id,
        "content": content_blocks,
    }
    return json.dumps(tool_result, ensure_ascii=False)


def openai_response_to_anthropic_message(response: dict, model: str) -> dict:
    """
    Convert OpenAI chat completion response to Anthropic message format.
    """
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = response.get("usage", {})

    # Log cache effectiveness if cache metrics are present
    prompt_tokens_details = usage.get("prompt_tokens_details", {})
    if prompt_tokens_details and "cached_tokens" in prompt_tokens_details:
        cached = prompt_tokens_details["cached_tokens"]
        total = usage.get("prompt_tokens", 0)
        if total > 0:
            cache_hit_rate = (cached / total) * 100
            logger.info(
                f"Cache hit: {cached}/{total} tokens ({cache_hit_rate:.1f}%) | "
                f"Model: {model} | Request ID: {response.get('id', 'N/A')}"
            )

    content = []

    # Handle text content
    if message.get("content"):
        content.append({
            "type": "text",
            "text": message["content"]
        })

    # Handle tool calls
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}

        content.append({
            "type": "tool_use",
            "id": tc.get("id", str(uuid.uuid4())),
            "name": func.get("name", ""),
            "input": args
        })

    # Determine stop reason
    finish_reason = choice.get("finish_reason", "end_turn")
    stop_reason = finish_reason_to_anthropic_stop_reason(finish_reason)

    # Build usage object with cache metrics if available
    usage_obj = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0)
    }

    # Add cache metrics if present (OpenRouter/OpenAI format)
    prompt_tokens_details = usage.get("prompt_tokens_details", {})
    if prompt_tokens_details:
        # OpenAI uses cached_tokens for cache reads
        if "cached_tokens" in prompt_tokens_details:
            usage_obj["cache_read_input_tokens"] = prompt_tokens_details["cached_tokens"]
        # Calculate cache creation tokens (tokens that were cached but not read from cache)
        # This is an approximation based on total prompt tokens minus cached reads
        cache_creation = usage.get("prompt_tokens", 0) - prompt_tokens_details.get("cached_tokens", 0)
        if cache_creation > 0:
            usage_obj["cache_creation_input_tokens"] = cache_creation

    return {
        "id": response.get("id", f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_obj
    }


def finish_reason_to_anthropic_stop_reason(finish_reason: Optional[str]) -> str:
    """
    Map OpenAI finish_reason to Anthropic stop_reason.

    OpenAI: stop, length, tool_calls, content_filter, null
    Anthropic: end_turn, max_tokens, stop_sequence, tool_use
    """
    if finish_reason is None:
        return "end_turn"

    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
        "function_call": "tool_use",  # Legacy OpenAI
    }

    return mapping.get(finish_reason, "end_turn")


def sse_event(event: str, data: Any) -> str:
    """
    Format a Server-Sent Event string.
    """
    if isinstance(data, dict):
        data_str = json.dumps(data)
    else:
        data_str = str(data)

    return f"event: {event}\ndata: {data_str}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# SSE Parser for OpenRouter stream
# ─────────────────────────────────────────────────────────────────────────────

async def iter_sse_data_lines(response: httpx.Response) -> AsyncIterator[str]:
    """
    Parse SSE stream from httpx response, yielding only data lines.

    Properly handles:
    - SSE comments (lines starting with ':')
    - Empty lines (event separators)
    - Multi-line data fields
    - Partial line buffering
    """
    buffer = ""

    async for chunk in response.aiter_text():
        buffer += chunk

        # Process complete lines
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")  # Handle \r\n line endings

            # Skip empty lines (event separators)
            if not line:
                continue

            # Skip SSE comments (lines starting with ':')
            if line.startswith(":"):
                continue

            # Handle data lines
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                yield data
            elif line.startswith("data:"):
                data = line[5:]  # Remove "data:" prefix (no space)
                yield data

    # Handle any remaining data in buffer
    if buffer:
        line = buffer.rstrip("\r")
        if line and not line.startswith(":"):
            if line.startswith("data: "):
                yield line[6:]
            elif line.startswith("data:"):
                yield line[5:]


# ─────────────────────────────────────────────────────────────────────────────
# Streaming handler
# ─────────────────────────────────────────────────────────────────────────────

async def stream_openrouter_to_anthropic(
    client: httpx.AsyncClient,
    openrouter_request: dict,
    anthropic_model: str,
    request_id: str,
    fallback_models: Optional[list[str]] = None
) -> AsyncIterator[tuple[str, Optional[dict]]]:
    """
    Stream OpenRouter response and convert to Anthropic SSE format in real-time.

    Handles:
    - Text deltas (token by token)
    - Tool call deltas (with partial JSON)
    - Mid-stream errors
    - Proper event sequencing
    - Fallback to alternative models before first SSE event

    Args:
        client: HTTP client
        openrouter_request: Request dict for OpenRouter
        anthropic_model: Original Anthropic model name (for response)
        request_id: Request ID for logging
        fallback_models: List of fallback models to try if primary fails

    Yields:
        tuple: (sse_event_string, usage_dict_or_None)
            - First element is the SSE event to send to client
            - Second element is usage dict with final token counts (only on last yield), None otherwise
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    if OPENROUTER_APP_URL:
        headers["HTTP-Referer"] = OPENROUTER_APP_URL
    if OPENROUTER_APP_TITLE:
        headers["X-Title"] = OPENROUTER_APP_TITLE

    # Prepare model list: primary model + fallbacks
    models_to_try = [openrouter_request["model"]]
    if fallback_models:
        models_to_try.extend(fallback_models[:FALLBACK_MAX_TRIES - 1])

    # State tracking for streaming
    message_id = f"msg_{uuid.uuid4().hex}"
    content_blocks: list = []  # Track content blocks
    current_tool_calls: dict = {}  # Track tool calls by index
    input_tokens = 0
    output_tokens = 0
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0
    stop_reason = None
    message_started = False

    # Try each model in sequence
    last_error = None
    for attempt, model in enumerate(models_to_try, 1):
        current_request = dict(openrouter_request)
        current_request["model"] = model

        # Log attempt (both primary and fallback)
        logger.info(
            f"Attempt {attempt}/{len(models_to_try)} with model={model} | "
            f"request_id={request_id}"
        )

        try:
            async with client.stream(
                "POST",
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=current_request,
                timeout=TIMEOUT_S
            ) as response:

                # Check status code before reading any data
                if response.status_code != 200:
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors="replace")
                    logger.warning(
                        f"Model {model} failed with status {response.status_code} | "
                        f"attempt={attempt}/{len(models_to_try)} request_id={request_id}"
                    )

                    # Check if we should fallback
                    if response.status_code in FALLBACK_ON_STATUS and attempt < len(models_to_try):
                        last_error = f"OpenRouter returned {response.status_code}: {error_text}"
                        continue  # Try next model

                    # No fallback, yield error
                    yield sse_event("error", {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": f"OpenRouter returned {response.status_code}: {error_text}",
                            "request_id": request_id
                        }
                    }), None
                    return

                # Successfully opened stream with 200 status

                async for data_line in iter_sse_data_lines(response):
                    # Check for stream end
                    if data_line.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse SSE data: {data_line[:100]}... Error: {e}")
                        continue

                    # Handle error in chunk
                    if "error" in chunk:
                        yield sse_event("error", {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": chunk["error"].get("message", str(chunk["error"])),
                                "request_id": request_id,
                            }
                        }), None
                        return

                    # Extract usage if present
                    if "usage" in chunk:
                        usage = chunk["usage"]
                        input_tokens = usage.get("prompt_tokens", input_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)

                        # Extract cache metrics if available
                        prompt_tokens_details = usage.get("prompt_tokens_details", {})
                        if prompt_tokens_details:
                            if "cached_tokens" in prompt_tokens_details:
                                cache_read_input_tokens = prompt_tokens_details["cached_tokens"]
                            # Calculate cache creation tokens
                            cache_creation = usage.get("prompt_tokens", 0) - prompt_tokens_details.get("cached_tokens", 0)
                            if cache_creation > 0:
                                cache_creation_input_tokens = cache_creation

                    # Get choice delta
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Send message_start event on first chunk
                    if not message_started:
                        message_started = True
                        usage_start = {
                            "input_tokens": input_tokens,
                            "output_tokens": 0
                        }
                        if cache_creation_input_tokens > 0:
                            usage_start["cache_creation_input_tokens"] = cache_creation_input_tokens
                        if cache_read_input_tokens > 0:
                            usage_start["cache_read_input_tokens"] = cache_read_input_tokens

                        yield sse_event("message_start", {
                            "type": "message_start",
                            "message": {
                                "id": message_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "model": anthropic_model,
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": usage_start
                            }
                        }), None

                    # Handle text content delta
                    if "content" in delta and delta["content"]:
                        text_content = delta["content"]

                        # Find or create text content block
                        text_block_index = None
                        for i, block in enumerate(content_blocks):
                            if block["type"] == "text":
                                text_block_index = i
                                break

                        if text_block_index is None:
                            # Create new text block
                            text_block_index = len(content_blocks)
                            content_blocks.append({"type": "text", "text": ""})

                            yield sse_event("content_block_start", {
                                "type": "content_block_start",
                                "index": text_block_index,
                                "content_block": {
                                    "type": "text",
                                    "text": ""
                                }
                            }), None

                        # Send text delta
                        content_blocks[text_block_index]["text"] += text_content
                        yield sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": text_block_index,
                            "delta": {
                                "type": "text_delta",
                                "text": text_content
                            }
                        }), None

                    # Handle tool calls delta
                    if "tool_calls" in delta:
                        for tc_delta in delta["tool_calls"]:
                            tc_index = tc_delta.get("index", 0)

                            if tc_index not in current_tool_calls:
                                # New tool call - create it
                                tool_id = tc_delta.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                                func = tc_delta.get("function", {})
                                tool_name = func.get("name", "")

                                # Calculate content block index (after text blocks)
                                block_index = len(content_blocks)

                                current_tool_calls[tc_index] = {
                                    "block_index": block_index,
                                    "id": tool_id,
                                    "name": tool_name,
                                    "arguments": ""
                                }

                                content_blocks.append({
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": tool_name,
                                    "input": {}
                                })

                                yield sse_event("content_block_start", {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tool_id,
                                        "name": tool_name,
                                        "input": {}
                                    }
                                }), None

                            # Handle argument delta
                            func = tc_delta.get("function", {})
                            if "arguments" in func:
                                arg_delta = func["arguments"]
                                if arg_delta:
                                    tc_info = current_tool_calls[tc_index]
                                    tc_info["arguments"] += arg_delta

                                    yield sse_event("content_block_delta", {
                                        "type": "content_block_delta",
                                        "index": tc_info["block_index"],
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": arg_delta
                                        }
                                    }), None

                    # Handle finish reason
                    if finish_reason:
                        stop_reason = finish_reason_to_anthropic_stop_reason(finish_reason)

                # Close all content blocks
                for i in range(len(content_blocks)):
                    yield sse_event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": i
                    }), None

                # Log cache effectiveness if available
                if cache_read_input_tokens > 0 and input_tokens > 0:
                    cache_hit_rate = (cache_read_input_tokens / input_tokens) * 100
                    logger.info(
                        f"Cache hit (stream): {cache_read_input_tokens}/{input_tokens} tokens ({cache_hit_rate:.1f}%) | "
                        f"Model: {anthropic_model} | Message ID: {message_id}"
                    )

                # Send message_delta with final stop reason and usage
                usage_delta = {
                    "output_tokens": output_tokens
                }
                # Note: cache metrics are only in message_start, not in message_delta

                yield sse_event("message_delta", {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason or "end_turn",
                        "stop_sequence": None
                    },
                    "usage": usage_delta
                }), None

                # Send message_stop
                yield sse_event("message_stop", {
                    "type": "message_stop"
                }), None

                # Yield final usage data for quota reconciliation
                final_usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                yield "", final_usage

                # Successfully completed streaming
                return

        except httpx.TimeoutException as e:
            logger.warning(
                f"Model {model} timed out | attempt={attempt}/{len(models_to_try)} "
                f"request_id={request_id}"
            )
            # Fallback on timeout if not yet started streaming
            if not message_started and attempt < len(models_to_try):
                last_error = f"Request timed out after {TIMEOUT_S} seconds"
                continue
            # Otherwise yield error
            yield sse_event("error", {
                "type": "error",
                "error": {
                    "type": "timeout_error",
                    "message": f"Request timed out after {TIMEOUT_S} seconds",
                    "request_id": request_id
                }
            }), None
            return
        except httpx.RequestError as e:
            logger.warning(
                f"Model {model} request failed: {str(e)} | attempt={attempt}/{len(models_to_try)} "
                f"request_id={request_id}"
            )
            # Fallback on request error if not yet started streaming
            if not message_started and attempt < len(models_to_try):
                last_error = f"Request failed: {str(e)}"
                continue
            # Otherwise yield error
            yield sse_event("error", {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Request failed: {str(e)}",
                    "request_id": request_id
                }
            }), None
            return

    # If we exhausted all models without success
    if last_error:
        yield sse_event("error", {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": f"All models failed. Last error: {last_error}",
                "request_id": request_id
            }
        }), None


async def openrouter_chat_completion(
    client: httpx.AsyncClient,
    openrouter_request: dict,
    headers: dict[str, str],
    request_id: str,
    fallback_models: Optional[list[str]] = None,
) -> dict:
    """
    Execute a non-streaming chat completion with optional fallback models.

    Args:
        client: HTTP client
        openrouter_request: Request dict for OpenRouter
        headers: HTTP headers
        request_id: Request ID for logging
        fallback_models: List of fallback models to try if primary fails

    Returns:
        OpenRouter response dict

    Raises:
        HTTPException: If all models fail
    """
    # Prepare model list: primary model + fallbacks
    models_to_try = [openrouter_request["model"]]
    if fallback_models:
        models_to_try.extend(fallback_models[:FALLBACK_MAX_TRIES - 1])

    last_error = None
    last_status = None

    for attempt, model in enumerate(models_to_try, 1):
        current_request = dict(openrouter_request)
        current_request["model"] = model

        # Log attempt (both primary and fallback)
        logger.info(
            f"Attempt {attempt}/{len(models_to_try)} with model={model} | "
            f"request_id={request_id}"
        )

        try:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=current_request,
                timeout=TIMEOUT_S,
            )

            if resp.status_code == 200:
                return resp.json()

            # Non-200 status
            error_text = resp.text
            logger.warning(
                f"Model {model} failed with status {resp.status_code} | "
                f"attempt={attempt}/{len(models_to_try)} request_id={request_id}"
            )

            # Check if we should fallback
            if resp.status_code in FALLBACK_ON_STATUS and attempt < len(models_to_try):
                last_error = f"OpenRouter returned {resp.status_code}: {error_text}"
                last_status = resp.status_code
                continue  # Try next model

            # No fallback available, raise error
            logger.error("OpenRouter error: %s - %s", resp.status_code, error_text)
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"OpenRouter returned {resp.status_code}: {error_text}",
            )

        except httpx.TimeoutException:
            logger.warning(
                f"Model {model} timed out | attempt={attempt}/{len(models_to_try)} "
                f"request_id={request_id}"
            )
            if attempt < len(models_to_try):
                last_error = f"Request timed out after {TIMEOUT_S} seconds"
                last_status = 504
                continue
            # No more models to try
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {TIMEOUT_S} seconds",
            )

        except httpx.RequestError as e:
            logger.warning(
                f"Model {model} request failed: {str(e)} | attempt={attempt}/{len(models_to_try)} "
                f"request_id={request_id}"
            )
            if attempt < len(models_to_try):
                last_error = f"Request failed: {str(e)}"
                last_status = 502
                continue
            # No more models to try
            raise HTTPException(
                status_code=502,
                detail=f"Failed to connect to OpenRouter: {str(e)}",
            )

    # If we get here, all models failed
    raise HTTPException(
        status_code=last_status or 500,
        detail=f"All models failed. Last error: {last_error}",
    )


async def run_proxy_tools_loop(
    client: httpx.AsyncClient,
    openrouter_request: dict,
    headers: dict[str, str],
    intercepted_tool_names: set[str],
    request_id: str,
    fallback_models: Optional[list[str]] = None,
) -> dict:
    """
    Execute a local tool loop for a subset of tools.

    If a response contains only intercepted tool calls, execute them locally,
    append tool results to messages, and continue the conversation with OpenRouter.
    If any non-intercepted tool call appears, return that response so the client
    can handle it normally.
    """
    request = dict(openrouter_request)
    messages = list(request.get("messages", []))
    request["messages"] = messages

    last_response: Optional[dict] = None
    for _ in range(max(1, PROXY_TOOLS_MAX_ITERS)):
        last_response = await openrouter_chat_completion(
            client, request, headers, request_id, fallback_models
        )
        choice = (last_response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            return last_response

        interceptable: list[dict] = []
        non_interceptable: list[dict] = []
        for tc in tool_calls:
            func = tc.get("function") or {}
            name = func.get("name") or ""
            if name in intercepted_tool_names:
                interceptable.append(tc)
            else:
                non_interceptable.append(tc)

        if non_interceptable:
            return last_response

        # Append assistant tool-call message
        messages.append(message)

        # Execute intercepted tools
        for tc in interceptable:
            tc_id = tc.get("id", "")
            func = tc.get("function") or {}
            args_raw = func.get("arguments") or "{}"
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {}

            query = args.get("query") or ""
            allowed = args.get("allowed_domains") or None
            blocked = args.get("blocked_domains") or None

            try:
                results = await exa_search_and_contents(
                    client,
                    query=query,
                    allowed_domains=allowed,
                    blocked_domains=blocked,
                )
                tool_payload = build_websearch_tool_result(tc_id, results)
            except Exception as e:
                logger.exception("Proxy WebSearch failed")
                tool_payload = json.dumps(
                    {
                        "type": "web_search_tool_result_error",
                        "tool_use_id": tc_id,
                        "error_code": "unavailable",
                        "message": str(e),
                    },
                    ensure_ascii=False,
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_payload,
                }
            )

        request = dict(request)
        request["messages"] = messages

    return last_response or {"choices": [{"message": {"role": "assistant", "content": ""}}]}


async def anthropic_sse_from_message(message: dict, anthropic_model: str) -> AsyncIterator[str]:
    """
    Convert a full Anthropic message dict to SSE events.
    Used when we need to precompute the response (e.g., proxy-side tools).
    """
    message_id = message.get("id", f"msg_{uuid.uuid4().hex}")
    usage = message.get("usage") or {}
    usage_start = {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": 0,
    }
    if "cache_creation_input_tokens" in usage:
        usage_start["cache_creation_input_tokens"] = usage["cache_creation_input_tokens"]
    if "cache_read_input_tokens" in usage:
        usage_start["cache_read_input_tokens"] = usage["cache_read_input_tokens"]

    yield sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": anthropic_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage_start,
            },
        },
    )

    content_blocks = message.get("content") or []
    for i, block in enumerate(content_blocks):
        btype = block.get("type")
        if btype == "text":
            yield sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
            )
            text = block.get("text", "")
            if text:
                yield sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
        elif btype == "tool_use":
            yield sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    },
                },
            )
        else:
            # Fallback: serialize unknown blocks as text.
            yield sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
            )
            yield sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "text_delta", "text": json.dumps(block, ensure_ascii=False)},
                },
            )

        yield sse_event("content_block_stop", {"type": "content_block_stop", "index": i})

    yield sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": message.get("stop_reason") or "end_turn",
                "stop_sequence": None,
            },
            "usage": {"output_tokens": usage.get("output_tokens", 0)},
        },
    )
    yield sse_event("message_stop", {"type": "message_stop"})


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/v1/messages")
async def v1_messages(request: Request):
    """
    Anthropic Messages API endpoint.

    Accepts Anthropic-format requests and proxies them to OpenRouter,
    converting the response back to Anthropic format.
    """
    # Get or generate request_id and store in request.state for error handler
    request_id = get_or_generate_request_id(request)
    request.state.request_id = request_id

    # Get API key from request.state (set by middleware)
    api_key = getattr(request.state, "api_key", None)

    # Start timing for metrics
    start_time = time.time()

    if not OPENROUTER_API_KEY:
        error_counter.labels(endpoint='/v1/messages', error_type='configuration_error').inc()
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY environment variable is not set"
        )

    try:
        body = await request.json()
    except Exception as e:
        error_counter.labels(endpoint='/v1/messages', error_type='invalid_json').inc()
        request_counter.labels(endpoint='/v1/messages', status='400').inc()
        request_latency.labels(endpoint='/v1/messages').observe(time.time() - start_time)
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    # Extract Anthropic request parameters
    anthropic_model = body.get("model", "claude-sonnet-4-20250514")
    messages = body.get("messages", [])
    system = body.get("system")
    max_tokens = body.get("max_tokens", 4096)

    # Validate max_tokens (minimum 16 for OpenAI models)
    if max_tokens < 16:
        logger.warning(f"max_tokens={max_tokens} is below minimum (16), setting to 16")
        max_tokens = 16
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    top_k = body.get("top_k")
    stop_sequences = body.get("stop_sequences")
    stream = body.get("stream", False)
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")

    # Check quota (Feature D)
    # Estimate tokens: input tokens + max_tokens for output
    estimated_input_tokens = count_anthropic_request_tokens(body)
    estimated_total_tokens = estimated_input_tokens + max_tokens

    if api_key:
        try:
            await check_and_reserve_quota(api_key, estimated_total_tokens, request_id)
        except HTTPException as e:
            # Quota exceeded, return 429
            error_counter.labels(endpoint='/v1/messages', error_type='quota_exceeded').inc()
            request_counter.labels(endpoint='/v1/messages', status='429').inc()
            request_latency.labels(endpoint='/v1/messages').observe(time.time() - start_time)
            raise

    # OpenRouter-specific parameters
    provider = body.get("provider")  # Provider routing object
    reasoning = body.get("reasoning")  # Reasoning mode
    include_reasoning = body.get("include_reasoning")  # Include reasoning in response

    # Pick target model and extract provider info and web search flag from model name if present
    target_model, model_provider_info, enable_web_search = pick_target_model(anthropic_model)

    # Get fallback models for this category (if any)
    fallback_models = get_fallback_models_for_category(anthropic_model)
    if fallback_models:
        logger.info(
            f"Fallback models configured: {fallback_models} | request_id={request_id}"
        )

    # Merge provider info from model name with explicit provider parameter
    # Explicit provider parameter takes precedence
    if model_provider_info and not provider:
        provider = model_provider_info
    elif model_provider_info and provider:
        # Merge both: explicit provider settings override model-based ones
        merged_provider = {**model_provider_info, **provider}
        provider = merged_provider

    # Convert provider object to OpenRouter format (camelCase)
    if provider:
        provider = convert_provider_to_openrouter_format(provider)

    # Convert to OpenAI format
    openai_messages = anthropic_messages_to_openai_messages(messages, system)

    # Convert tools to OpenAI format
    openai_tools = tools_anthropic_to_openai(tools)

    # Inject cache_control breakpoints for Anthropic models on OpenRouter
    # This enables prompt caching for system message, last 2 user messages, and tools
    openai_messages, openai_tools = inject_cache_control_for_anthropic(
        openai_messages, openai_tools, target_model
    )

    # Simplify messages content - convert arrays to strings for providers that don't support content arrays.
    # IMPORTANT: do NOT simplify when cache_control breakpoints are present (Anthropic/OpenRouter prompt caching).
    simplified_messages = []
    for msg in openai_messages:
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
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
        simplified_messages.append(new_msg)

    # Detect prompt caching markers for diagnostics/warnings.
    has_cache_markers = False
    cache_markers: list[tuple[int, int, Any]] = []
    for i, msg in enumerate(simplified_messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if isinstance(block, dict) and ("cache_control" in block):
                has_cache_markers = True
                cache_markers.append((i, j, block.get("cache_control")))

    # Build OpenRouter request
    openrouter_request = {
        "model": target_model,
        "messages": simplified_messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if temperature is not None:
        openrouter_request["temperature"] = temperature
    if top_p is not None:
        openrouter_request["top_p"] = top_p
    if stop_sequences:
        openrouter_request["stop"] = stop_sequences

    # Add tools if present (already converted and cache_control injected above)
    if openai_tools:
        openrouter_request["tools"] = openai_tools

    openai_tool_choice = tool_choice_anthropic_to_openai(tool_choice)
    if openai_tool_choice is not None:
        openrouter_request["tool_choice"] = openai_tool_choice

    # Add OpenRouter-specific parameters
    if provider is not None:
        openrouter_request["provider"] = provider

    if reasoning is not None:
        openrouter_request["reasoning"] = reasoning
    if include_reasoning is not None:
        openrouter_request["include_reasoning"] = include_reasoning

    # Log the request (without message content)
    logger.info(
        f"request_id={request_id} model={target_model} "
        f"provider={json.dumps(provider) if provider else 'None'} "
        f"web_search={enable_web_search or OPENROUTER_WEB_ENABLED} "
        f"messages_count={len(openai_messages)}"
    )

    # Optional: log cache_control marker positions for debugging prompt caching.
    if DEBUG_CACHE_MARKERS and has_cache_markers:
        markers_str = ", ".join(
            f"msg[{i}].content[{j}]={cc!r}" for (i, j, cc) in cache_markers
        )
        logger.info(
            f"request_id={request_id} cache_control_markers={markers_str}"
        )

    # If cache markers exist but we're not targeting Anthropic, warn (caching likely won't work).
    if has_cache_markers and ("anthropic/" not in (target_model or "")):
        logger.warning(
            f"request_id={request_id} cache_control_present_but_non_anthropic_model="
            f"{target_model} provider={json.dumps(provider) if provider else 'None'}"
        )

    # Request stream_options for usage in streaming mode
    if stream:
        openrouter_request["stream_options"] = {"include_usage": True}

    # Enable OpenRouter web search plugin if requested via search: prefix or env var
    if OPENROUTER_WEB_ENABLED or enable_web_search:
        if enable_web_search:
            logger.info("Web search enabled via search: prefix in model name")
        plugin: dict[str, Any] = {"id": "web"}
        if OPENROUTER_WEB_ENGINE:
            plugin["engine"] = OPENROUTER_WEB_ENGINE
        if OPENROUTER_WEB_MAX_RESULTS:
            try:
                plugin["max_results"] = int(OPENROUTER_WEB_MAX_RESULTS)
            except ValueError:
                logger.warning("Invalid OPENROUTER_WEB_MAX_RESULTS=%r (expected int)", OPENROUTER_WEB_MAX_RESULTS)
        if OPENROUTER_WEB_SEARCH_PROMPT:
            plugin["search_prompt"] = OPENROUTER_WEB_SEARCH_PROMPT

        openrouter_request["plugins"] = [plugin]

        if OPENROUTER_WEB_SEARCH_CONTEXT_SIZE:
            openrouter_request["web_search_options"] = {
                "search_context_size": OPENROUTER_WEB_SEARCH_CONTEXT_SIZE
            }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    if OPENROUTER_APP_URL:
        headers["HTTP-Referer"] = OPENROUTER_APP_URL
    if OPENROUTER_APP_TITLE:
        headers["X-Title"] = OPENROUTER_APP_TITLE

    # Debug dump (gated behind DEBUG_DUMP_REQUESTS env var)
    # Moved here to ensure headers are defined before dump
    if DEBUG_DUMP_REQUESTS:
        redacted_request = redact_request_for_debug(openrouter_request)
        redacted_upstream_headers = redact_headers_for_debug(headers)
        debug_dump = {
            "request": redacted_request,
            "upstream_headers": redacted_upstream_headers
        }
        redacted_json = json.dumps(debug_dump, indent=2, default=str)
        path = "/tmp/last_openrouter_request.json"
        try:
            with open(path, "w") as f:
                f.write(redacted_json)
            # Best-effort chmod - log warning if fails
            try:
                os.chmod(path, 0o600)
            except Exception as e:
                logger.warning(f"Failed to chmod debug dump file: {e}")
        except Exception as e:
            logger.warning(f"Failed to write debug dump to {path}: {e}")

    # Determine whether to intercept Claude Code's WebSearch tool.
    intercepted_tool_names: set[str] = set()
    if PROXY_WEBSEARCH_ENABLED and EXA_API_KEY and isinstance(tools, list):
        try:
            if any(isinstance(t, dict) and t.get("name") == "WebSearch" for t in tools):
                intercepted_tool_names.add("WebSearch")
        except Exception:
            intercepted_tool_names = set()

    if stream:
        if intercepted_tool_names:
            # Precompute the response via proxy-side tool loop, then replay as SSE.
            async def stream_generator():
                actual_input_tokens = 0
                actual_output_tokens = 0
                try:
                    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
                        proxy_request = dict(openrouter_request)
                        proxy_request["stream"] = False
                        proxy_request.pop("stream_options", None)
                        response_dict = await run_proxy_tools_loop(
                            client, proxy_request, headers, intercepted_tool_names,
                            request_id, fallback_models
                        )
                        anthropic_msg = openai_response_to_anthropic_message(
                            response_dict, anthropic_model
                        )
                        # Extract actual usage for quota reconciliation
                        usage = anthropic_msg.get("usage", {})
                        actual_input_tokens = usage.get("input_tokens", 0)
                        actual_output_tokens = usage.get("output_tokens", 0)

                        async for event in anthropic_sse_from_message(
                            anthropic_msg, anthropic_model
                        ):
                            yield event
                    request_counter.labels(endpoint='/v1/messages', status='200').inc()
                    request_latency.labels(endpoint='/v1/messages').observe(time.time() - start_time)
                except Exception as e:
                    error_counter.labels(endpoint='/v1/messages', error_type='streaming_error').inc()
                    request_counter.labels(endpoint='/v1/messages', status='500').inc()
                    request_latency.labels(endpoint='/v1/messages').observe(time.time() - start_time)
                    logger.exception(f"Streaming error in v1_messages (proxy tools): request_id={request_id}")
                    # Yield SSE error event instead of re-raising
                    yield sse_event("error", {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": f"Streaming error: {str(e)}",
                            "request_id": request_id
                        }
                    })
                    return
                finally:
                    # Reconcile quota after stream completes
                    if api_key and (actual_input_tokens > 0 or actual_output_tokens > 0):
                        actual_total = actual_input_tokens + actual_output_tokens
                        await reconcile_quota(api_key, estimated_total_tokens, actual_total, request_id)

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Request-Id": request_id,
                },
            )

        # REAL STREAMING MODE (pass-through)
        async def stream_generator():
            actual_input_tokens = 0
            actual_output_tokens = 0
            reconcile_attempted = False
            try:
                async with httpx.AsyncClient() as client:
                    async for event, usage in stream_openrouter_to_anthropic(
                        client, openrouter_request, anthropic_model, request_id, fallback_models
                    ):
                        # Extract usage from final yield (usage dict returned on last item)
                        if usage is not None:
                            actual_input_tokens = usage.get("input_tokens", actual_input_tokens)
                            actual_output_tokens = usage.get("output_tokens", actual_output_tokens)

                        # Only yield non-empty events (final usage yield has empty string)
                        if event:
                            yield event

                request_counter.labels(endpoint='/v1/messages', status='200').inc()
                request_latency.labels(endpoint='/v1/messages').observe(time.time() - start_time)
            except Exception as e:
                error_counter.labels(endpoint='/v1/messages', error_type='streaming_error').inc()
                request_counter.labels(endpoint='/v1/messages', status='500').inc()
                request_latency.labels(endpoint='/v1/messages').observe(time.time() - start_time)
                logger.exception(f"Streaming error in v1_messages: request_id={request_id}")
                # Yield SSE error event instead of re-raising
                yield sse_event("error", {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": f"Streaming error: {str(e)}",
                        "request_id": request_id
                    }
                })
                return
            finally:
                # Reconcile quota after stream completes (always runs, even on parsing failures)
                if api_key and not reconcile_attempted:
                    reconcile_attempted = True
                    # Use tracked tokens if available, otherwise fall back to estimate
                    if actual_input_tokens > 0 or actual_output_tokens > 0:
                        actual_total = actual_input_tokens + actual_output_tokens
                        await reconcile_quota(api_key, estimated_total_tokens, actual_total, request_id)
                    else:
                        # Parsing failed - log warning and use estimate as actual
                        logger.warning(
                            f"Failed to extract usage from stream, using estimate as actual | "
                            f"estimated={estimated_total_tokens} request_id={request_id}"
                        )
                        await reconcile_quota(api_key, estimated_total_tokens, estimated_total_tokens, request_id)

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-Id": request_id,
            },
        )

    # NON-STREAMING MODE
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        try:
            if intercepted_tool_names:
                proxy_request = dict(openrouter_request)
                proxy_request["stream"] = False
                proxy_request.pop("stream_options", None)
                openai_response = await run_proxy_tools_loop(
                    client, proxy_request, headers, intercepted_tool_names,
                    request_id, fallback_models
                )
            else:
                # Use openrouter_chat_completion which now handles fallback
                openai_response = await openrouter_chat_completion(
                    client, openrouter_request, headers, request_id, fallback_models
                )

            anthropic_response = openai_response_to_anthropic_message(
                openai_response, anthropic_model
            )

            # Reconcile quota with actual usage (non-streaming)
            if api_key:
                usage = anthropic_response.get("usage", {})
                actual_input_tokens = usage.get("input_tokens", 0)
                actual_output_tokens = usage.get("output_tokens", 0)
                actual_total = actual_input_tokens + actual_output_tokens
                if actual_total > 0:
                    await reconcile_quota(api_key, estimated_total_tokens, actual_total, request_id)

            # Record successful request
            request_counter.labels(endpoint='/v1/messages', status='200').inc()
            request_latency.labels(endpoint='/v1/messages').observe(time.time() - start_time)

            return JSONResponse(
                content=anthropic_response,
                headers={"X-Request-Id": request_id}
            )

        except HTTPException:
            # HTTPException is already raised by openrouter_chat_completion with proper error handling
            # Just re-raise it (metrics already recorded in the function)
            raise
        except Exception as e:
            # Catch any other unexpected errors
            error_counter.labels(endpoint='/v1/messages', error_type='unexpected_error').inc()
            request_counter.labels(endpoint='/v1/messages', status='500').inc()
            request_latency.labels(endpoint='/v1/messages').observe(time.time() - start_time)
            logger.exception(f"Unexpected error in v1_messages: request_id={request_id}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error: {str(e)}",
            )


@app.post("/v1/messages/count_tokens")
async def v1_messages_count_tokens(request: Request):
    """Anthropic Messages API token counting (Claude Code LLM gateway compatibility)."""
    request_id = get_or_generate_request_id(request)

    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    try:
        input_tokens = count_anthropic_request_tokens(body)
    except Exception as e:
        logger.exception("Failed to count tokens")
        raise HTTPException(status_code=500, detail=f"Failed to count tokens: {str(e)}")

    return JSONResponse(
        content={"input_tokens": input_tokens},
        headers={"X-Request-Id": request_id}
    )


@app.get("/health")
async def health():
    """
    Enhanced health check endpoint (Feature E).

    Returns:
    - status: "ok" if healthy
    - uptime_seconds: seconds since proxy started
    - version: proxy version string
    - enabled_features: dict of feature flags
    - config_summary: basic config info (no secrets)
    """
    uptime_seconds = int(time.time() - PROXY_START_TIME)

    # Feature flags
    enabled_features = {
        "auth": bool(PROXY_API_KEYS_SET),
        "metrics": True,  # Always available via /metrics
        "fallback": bool(FALLBACK_MODELS_SONNET or FALLBACK_MODELS_OPUS or FALLBACK_MODELS_HAIKU),
        "quotas": bool(QUOTA_CONFIG),
        "debug_dump": DEBUG_DUMP_REQUESTS
    }

    # Basic config summary (no secrets)
    config_summary = {
        "api_keys_count": len(PROXY_API_KEYS_SET),
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "model_mapping": {
            "default": TARGET_MODEL_DEFAULT,
            "small": TARGET_MODEL_SMALL,
            "big": TARGET_MODEL_BIG,
            "opus": TARGET_MODEL_OPUS
        },
        "fallback_enabled": enabled_features["fallback"],
        "quota_keys_count": len(QUOTA_CONFIG) if QUOTA_CONFIG else 0,
        "timeout_seconds": TIMEOUT_S
    }

    return {
        "status": "ok",
        "uptime_seconds": uptime_seconds,
        "version": PROXY_VERSION,
        "enabled_features": enabled_features,
        "config_summary": config_summary
    }


@app.get("/v1/admin/stats")
async def admin_stats(request: Request):
    """
    Admin stats endpoint (Feature E).

    Requires both X-API-Key (normal auth) and X-Admin-Key (admin auth).
    Returns:
    - uptime_seconds
    - metrics snapshot (proxy_requests_total for /v1/messages if available)
    - quota enabled keys count
    """
    request_id = getattr(request.state, "request_id", None) or get_or_generate_request_id(request)

    # Check admin key (after normal X-API-Key auth from middleware)
    check_admin_key(request)

    uptime_seconds = int(time.time() - PROXY_START_TIME)

    # Get metrics snapshot - proxy_requests_total for /v1/messages
    metrics_snapshot = {}
    try:
        # Parse prometheus metrics to extract proxy_requests_total for /v1/messages
        from prometheus_client import REGISTRY
        for collector in REGISTRY._collector_to_names:
            for metric in collector.collect():
                if metric.name == 'proxy_requests_total':
                    for sample in metric.samples:
                        # Filter for /v1/messages endpoint
                        labels = sample.labels or {}
                        if labels.get('endpoint') == '/v1/messages':
                            status = labels.get('status', 'unknown')
                            key = f"requests_{status}"
                            metrics_snapshot[key] = sample.value
    except Exception as e:
        logger.warning(f"Failed to extract metrics snapshot: {e}")

    # Quota keys count
    quota_enabled_keys_count = len(QUOTA_CONFIG) if QUOTA_CONFIG else 0

    return JSONResponse(
        content={
            "uptime_seconds": uptime_seconds,
            "metrics_snapshot": metrics_snapshot,
            "quota_enabled_keys_count": quota_enabled_keys_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        headers={"X-Request-Id": request_id}
    )


@app.get("/v1/admin/config")
async def admin_config(request: Request):
    """
    Admin config endpoint (Feature E).

    Requires both X-API-Key (normal auth) and X-Admin-Key (admin auth).
    Returns sanitized config:
    - model mapping
    - fallback settings (models list, status codes, max tries)
    - quota settings (filenames/flags, not actual quotas)
    - debug flags
    """
    request_id = getattr(request.state, "request_id", None) or get_or_generate_request_id(request)

    # Check admin key (after normal X-API-Key auth from middleware)
    check_admin_key(request)

    # Sanitized config (no secrets)
    sanitized_config = {
        "version": PROXY_VERSION,
        "model_mapping": {
            "default": TARGET_MODEL_DEFAULT,
            "small": TARGET_MODEL_SMALL,
            "big": TARGET_MODEL_BIG,
            "opus": TARGET_MODEL_OPUS
        },
        "fallback_settings": {
            "sonnet_models": FALLBACK_MODELS_SONNET.split(",") if FALLBACK_MODELS_SONNET else [],
            "opus_models": FALLBACK_MODELS_OPUS.split(",") if FALLBACK_MODELS_OPUS else [],
            "haiku_models": FALLBACK_MODELS_HAIKU.split(",") if FALLBACK_MODELS_HAIKU else [],
            "on_status_codes": sorted(list(FALLBACK_ON_STATUS)),
            "max_tries": FALLBACK_MAX_TRIES
        },
        "quota_settings": {
            "enabled": bool(QUOTA_CONFIG),
            "config_file": QUOTA_CONFIG_FILE if QUOTA_CONFIG_FILE else None,
            "config_json_set": bool(QUOTA_CONFIG_JSON),
            "state_file": QUOTA_STATE_FILE,
            "keys_count": len(QUOTA_CONFIG) if QUOTA_CONFIG else 0
        },
        "debug_flags": {
            "debug_dump_requests": DEBUG_DUMP_REQUESTS,
            "proxy_websearch_enabled": PROXY_WEBSEARCH_ENABLED,
            "openrouter_web_enabled": OPENROUTER_WEB_ENABLED
        },
        "timeout_seconds": TIMEOUT_S,
        "api_keys_count": len(PROXY_API_KEYS_SET),
        "admin_api_configured": bool(ADMIN_API_KEY)
    }

    return JSONResponse(
        content=sanitized_config,
        headers={"X-Request-Id": request_id}
    )


@app.get("/metrics")
async def metrics(request: Request):
    """
    Prometheus metrics endpoint.

    This endpoint requires X-API-Key authentication for security.
    """
    request_id = getattr(request.state, "request_id", None) or get_or_generate_request_id(request)
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
        headers={"X-Request-Id": request_id},
    )


@app.get("/v1/models")
async def v1_models(request: Request = None):
    """
    List available models endpoint for Claude Code compatibility.

    Returns a list of models that can be used with the proxy.
    This includes both direct OpenRouter model slugs and Claude model names
    that will be mapped to the configured TARGET_MODEL_* env vars.

    Uses caching to avoid slow API calls on every request (5 minute TTL).

    Supports query parameter ?check=model_id to validate a specific model.
    """
    global MODELS_CACHE, MODELS_CACHE_TIMESTAMP

    request_id = get_or_generate_request_id(request)

    # Check if cache is valid
    current_time = time.time()
    if MODELS_CACHE is not None and (current_time - MODELS_CACHE_TIMESTAMP) < MODELS_CACHE_TTL:
        logger.debug(f"Returning cached models (age: {current_time - MODELS_CACHE_TIMESTAMP:.1f}s)")
        return JSONResponse(content=MODELS_CACHE, headers={"X-Request-Id": request_id})

    # Cache miss or expired - fetch from OpenRouter
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            # Fetch available models from OpenRouter
            response = await client.get(
                f"{OPENROUTER_BASE_URL}/models",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
            )

            if response.status_code == 200:
                openrouter_data = response.json()
                models = openrouter_data.get("data", [])

                # Convert OpenRouter model format to Anthropic format
                anthropic_models = []
                for model in models:
                    model_id = model.get("id", "")
                    anthropic_models.append({
                        "type": "model",
                        "id": model_id,
                        "display_name": model.get("name", model_id),
                        "created_at": model.get("created", "2024-01-01T00:00:00Z")
                    })

                result = {"data": anthropic_models, "has_more": False, "first_id": None, "last_id": None}

                # Update cache
                MODELS_CACHE = result
                MODELS_CACHE_TIMESTAMP = current_time
                logger.info(f"Cached {len(anthropic_models)} models from OpenRouter")

                return JSONResponse(content=result, headers={"X-Request-Id": request_id})
            else:
                # Fallback to basic model list if OpenRouter API fails
                logger.warning(f"Failed to fetch models from OpenRouter: {response.status_code}")
                fallback_result = {
                    "data": [
                        {"type": "model", "id": TARGET_MODEL_DEFAULT, "display_name": "Default", "created_at": "2024-01-01T00:00:00Z"},
                        {"type": "model", "id": TARGET_MODEL_SMALL, "display_name": "Small (Haiku)", "created_at": "2024-01-01T00:00:00Z"},
                        {"type": "model", "id": TARGET_MODEL_BIG, "display_name": "Big (Sonnet)", "created_at": "2024-01-01T00:00:00Z"},
                        {"type": "model", "id": TARGET_MODEL_OPUS, "display_name": "Opus", "created_at": "2024-01-01T00:00:00Z"}
                    ],
                    "has_more": False,
                    "first_id": None,
                    "last_id": None
                }
                # Cache fallback result too
                MODELS_CACHE = fallback_result
                MODELS_CACHE_TIMESTAMP = current_time
                return JSONResponse(content=fallback_result, headers={"X-Request-Id": request_id})
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            # Return minimal model list on error
            error_result = {
                "data": [
                    {"type": "model", "id": TARGET_MODEL_DEFAULT, "display_name": "Default", "created_at": "2024-01-01T00:00:00Z"}
                ],
                "has_more": False,
                "first_id": None,
                "last_id": None
            }
            # Cache error result too (with shorter TTL)
            MODELS_CACHE = error_result
            MODELS_CACHE_TIMESTAMP = current_time
            return JSONResponse(content=error_result, headers={"X-Request-Id": request_id})


@app.get("/v1/models/{model_id:path}")
async def v1_model_by_id(model_id: str, request: Request = None):
    """
    Get a specific model by ID for Claude Code compatibility.

    Claude Code calls this endpoint to validate if a model exists.
    We accept any model that:
    1. Contains "/" (OpenRouter format like "deepseek/deepseek-v3.2")
    2. Contains ":" (provider suffix like "model:provider/quant")
    3. Contains known keywords (opus, sonnet, haiku)

    This allows using arbitrary OpenRouter models with provider routing.
    """
    request_id = get_or_generate_request_id(request)

    # Strip any Claude Code suffixes
    clean_model_id = _strip_claude_code_suffixes(model_id)

    # Extract base model (without provider suffix)
    base_model = clean_model_id
    if ":" in clean_model_id:
        base_model = clean_model_id.split(":", 1)[0]

    # Accept any model that looks valid
    is_valid = (
        "/" in base_model or  # OpenRouter format
        any(kw in clean_model_id.lower() for kw in ["opus", "sonnet", "haiku", "claude", "gpt", "deepseek", "gemini", "mistral", "llama"])
    )

    if is_valid:
        return JSONResponse(
            content={
                "type": "model",
                "id": model_id,
                "display_name": model_id,
                "created_at": "2024-01-01T00:00:00Z"
            },
            headers={"X-Request-Id": request_id}
        )

    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@app.get("/v1/usage")
async def v1_usage(request: Request):
    """
    Admin view endpoint for quota usage.

    Requires X-API-Key authentication (same as other endpoints).
    Returns masked keys with their usage, quota, and reset time.
    """
    request_id = get_or_generate_request_id(request)
    api_key = getattr(request.state, "api_key", None)

    if not QUOTA_CONFIG:
        return JSONResponse(
            content={
                "quota_enabled": False,
                "message": "Quota system not configured"
            },
            headers={"X-Request-Id": request_id}
        )

    # Build usage report
    usage_report = []

    async with quota_state_lock:
        for key, limit in QUOTA_CONFIG.items():
            state = quota_state.get(key, {
                "used_tokens": 0,
                "reset_at": get_next_utc_midnight()
            })

            # Check if reset is needed
            reset_at = datetime.fromisoformat(state["reset_at"])
            now = datetime.now(timezone.utc)
            if now >= reset_at:
                used_tokens = 0
                reset_at_str = get_next_utc_midnight()
            else:
                used_tokens = state["used_tokens"]
                reset_at_str = state["reset_at"]

            usage_report.append({
                "api_key": mask_api_key(key),
                "used_tokens": used_tokens,
                "quota_limit": limit,
                "remaining_tokens": max(0, limit - used_tokens),
                "reset_at": reset_at_str
            })

    return JSONResponse(
        content={
            "quota_enabled": True,
            "keys": usage_report,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        headers={"X-Request-Id": request_id}
    )


@app.get("/")
async def root(request: Request = None):
    """Root endpoint with API info."""
    request_id = get_or_generate_request_id(request)

    return JSONResponse(
        content={
            "name": "Anthropic to OpenRouter Proxy",
            "version": PROXY_VERSION,
            "endpoints": {
                "/v1/messages": "Anthropic Messages API (POST)",
                "/v1/messages/count_tokens": "Token count (POST)",
                "/v1/models": "List available models (GET)",
                "/v1/models/{model_id}": "Get model by ID (GET)",
                "/v1/usage": "Quota usage (GET, requires auth)",
                "/v1/admin/stats": "Admin stats (GET, requires X-API-Key + X-Admin-Key)",
                "/v1/admin/config": "Admin config (GET, requires X-API-Key + X-Admin-Key)",
                "/health": "Enhanced health check (GET, public)",
                "/metrics": "Prometheus metrics (GET, requires auth)"
            }
        },
        headers={"X-Request-Id": request_id}
    )


@app.post("/api/event_logging/batch")
async def event_logging_batch(request: Request):
    """
    Event logging batch endpoint for Claude Code.

    Accepts batches of events from Claude Code and logs them.
    This endpoint is called by Claude Code to log telemetry events.
    """
    request_id = get_or_generate_request_id(request)

    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Invalid JSON in /api/event_logging/batch: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    # Extract parameters
    events = body.get("events", [])
    batch_id = body.get("batch_id", str(uuid.uuid4()))
    timestamp = body.get("timestamp", "")

    # Log events
    logger.info(f"Event batch received [batch_id={batch_id}, events_count={len(events)}]")

    for idx, event in enumerate(events):
        event_type = event.get("event_type", "unknown")
        event_id = event.get("event_id", f"event_{idx}")
        logger.debug(f"  Event [{event_id}] type={event_type}: {json.dumps(event)}")

    # Return confirmation response
    return JSONResponse(
        content={
            "status": "success",
            "batch_id": batch_id,
            "received": len(events),
            "processed": len(events),
            "errors": [],
            "timestamp": timestamp if timestamp else str(uuid.uuid4())
        },
        headers={"X-Request-Id": request_id}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")

    uvicorn.run(app, host=host, port=port)
