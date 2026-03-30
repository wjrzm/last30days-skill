"""MiniMax Chat Completions API client for Reddit discovery.

Uses MiniMax's OpenAI-compatible Chat Completions API for Reddit search.
MiniMax does not have a Responses API with web_search tool, so this module
uses a prompt-based approach where the model is instructed to search Reddit
and return structured JSON results.

Falls back to public Reddit JSON endpoint if API call fails.
"""

import json
import re
import sys
from typing import Any, Dict, List, Optional

from . import http, openai_reddit


def _log_error(msg: str):
    """Log error to stderr."""
    sys.stderr.write(f"[MINIMAX REDDIT] {msg}\n")
    sys.stderr.flush()


def _log_info(msg: str):
    """Log info to stderr."""
    sys.stderr.write(f"[MINIMAX REDDIT] {msg}\n")
    sys.stderr.flush()


# MiniMax Chat Completions endpoint (OpenAI-compatible)
MINIMAX_CHAT_URL = "https://api.minimax.chat/v1/chat/completions"

# Fallback models in priority order
MINIMAX_MODEL_FALLBACK = ["MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M1.5"]


MINIMAX_REDDIT_PROMPT = """You are a research assistant helping find Reddit discussions about: {topic}

Search for Reddit threads from {from_date} to {to_date}. Find {min_items}-{max_items} relevant threads.

Search using these queries:
1. "{topic} site:reddit.com"
2. "reddit {topic}"
3. '"{topic}" site:reddit.com'

For each thread found, provide the title, URL (must include /r/ and /comments/), subreddit name,
date (YYYY-MM-DD if determinable, otherwise null), why it's relevant, and a relevance score 0-1.

Return ONLY valid JSON in this exact format, no other text:
{{
  "items": [
    {{
      "title": "Thread title",
      "url": "https://www.reddit.com/r/subreddit/comments/id/title/",
      "subreddit": "subreddit_name",
      "date": "YYYY-MM-DD or null",
      "why_relevant": "Why this thread is relevant",
      "relevance": 0.85
    }}
  ]
}}

Rules:
- URL must be a reddit.com link containing /r/ and /comments/
- Exclude developers.reddit.com and business.reddit.com
- Include as many relevant threads as possible
- If date is unknown, use null
- relevance is 0.0 to 1.0"""


def _build_payload(model: str, topic: str, from_date: str, to_date: str, depth: str) -> Dict[str, Any]:
    """Build MiniMax Chat Completions payload."""
    from .openai_reddit import DEPTH_CONFIG

    _, max_items = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["default"])
    # Request more items since some may be filtered
    target = min(max_items, 50)

    prompt = MINIMAX_REDDIT_PROMPT.format(
        topic=topic,
        from_date=from_date,
        to_date=to_date,
        min_items=target // 2,
        max_items=target,
    )

    return {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4096,
        "temperature": 0.3,  # Low temperature for consistent structured output
    }


def search_reddit(
    api_key: str,
    base_url: str,
    model: str,
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    mock_response: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Search Reddit using MiniMax Chat Completions API.

    Args:
        api_key: MiniMax API key
        base_url: MiniMax API base URL
        model: Model to use
        topic: Search topic
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        depth: Research depth - "quick", "default", or "deep"
        mock_response: Mock response for testing

    Returns:
        Raw API response (dict with 'items' key)
    """
    if mock_response is not None:
        return mock_response

    # Build URL from base_url
    chat_url = f"{base_url.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    timeout = 90 if depth == "quick" else 120 if depth == "default" else 180

    # Try model fallback chain
    models_to_try = [model] + [m for m in MINIMAX_MODEL_FALLBACK if m != model]

    last_error = None
    for current_model in models_to_try:
        try:
            payload = _build_payload(current_model, topic, from_date, to_date, depth)
            response = http.post(chat_url, payload, headers=headers, timeout=timeout)
            return response
        except http.HTTPError as e:
            last_error = e
            _log_info(f"MiniMax model {current_model} failed: {e}, trying fallback...")
            if e.status_code in (400, 404):
                # Model not found, try next
                continue
            if e.status_code == 401:
                _log_error("MiniMax API key is invalid")
                break
            # Other errors, don't retry with different model
            raise

    # All models failed
    if last_error:
        _log_error(f"All MiniMax models failed. Last error: {last_error}")
        raise last_error
    raise http.HTTPError("No MiniMax model available")


def parse_reddit_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse MiniMax Chat Completions response to extract Reddit items.

    Args:
        response: Raw API response

    Returns:
        List of item dicts
    """
    items = []

    # Check for API errors first
    if "error" in response and response["error"]:
        error = response["error"]
        err_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        _log_error(f"MiniMax API error: {err_msg}")
        return items

    # Extract content from Chat Completions response
    output_text = ""

    # Standard Chat Completions format
    if "choices" in response:
        for choice in response["choices"]:
            if isinstance(choice, dict):
                msg = choice.get("message", {})
                if isinstance(msg, dict):
                    output_text = msg.get("content", "")
                    if output_text:
                        break

    if not output_text:
        _log_error("No output text found in MiniMax response")
        return items

    # Also try to extract from openai_reddit-style output (some APIs return this format)
    if "output" in response:
        output = response["output"]
        if isinstance(output, str):
            output_text = output
        elif isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "output_text":
                                output_text = c.get("text", "")
                                break

    if not output_text:
        return items

    # Extract JSON from the response
    json_match = re.search(r'\{[\s\S]*?"items"[\s\S]*?\}', output_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            items = data.get("items", [])
        except json.JSONDecodeError:
            _log_error("Failed to parse JSON from MiniMax response")
            # Try a broader regex
            json_match2 = re.search(r'\{[\s\S]+\}', output_text)
            if json_match2:
                try:
                    data = json.loads(json_match2.group())
                    items = data.get("items", [])
                except json.JSONDecodeError:
                    pass

    if not items:
        _log_error(f"No items extracted. Output: {output_text[:500]}")

    # Validate and clean items
    clean_items = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        url = item.get("url", "")
        if not url or "reddit.com" not in url:
            continue

        clean_item = {
            "id": f"R{i+1}",
            "title": str(item.get("title", "")).strip(),
            "url": url,
            "subreddit": str(item.get("subreddit", "")).strip().lstrip("r/"),
            "date": item.get("date"),
            "why_relevant": str(item.get("why_relevant", "")).strip(),
            "relevance": min(1.0, max(0.0, float(item.get("relevance", 0.5)))),
        }

        # Validate date format
        if clean_item["date"]:
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', str(clean_item["date"])):
                clean_item["date"] = None

        clean_items.append(clean_item)

    return clean_items


def search_reddit_with_fallback(
    api_key: str,
    base_url: str,
    model: str,
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
) -> List[Dict[str, Any]]:
    """Search Reddit using MiniMax API, falling back to public JSON endpoint.

    Args:
        api_key: MiniMax API key
        base_url: MiniMax API base URL
        model: Model to use
        topic: Search topic
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        depth: Research depth

    Returns:
        List of item dicts
    """
    try:
        raw = search_reddit(api_key, base_url, model, topic, from_date, to_date, depth)
        items = parse_reddit_response(raw)
        if items:
            _log_info(f"MiniMax returned {len(items)} items")
            return items
    except Exception as e:
        _log_error(f"MiniMax search failed: {e}, falling back to public Reddit endpoint")

    # Fallback to public Reddit JSON endpoint
    _log_info("Using public Reddit JSON endpoint as fallback")
    return openai_reddit.search_reddit_public(topic, from_date, to_date, depth)
