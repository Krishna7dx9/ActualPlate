"""Server-side FatSecret Platform API client.

Provider failures are represented with safe, stable error messages. Credentials and
upstream response bodies are never included in public error payloads.
"""
from __future__ import annotations

import math
import os
import threading
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN_URL = "https://oauth.fatsecret.com/connect/token"
API_URL = "https://platform.fatsecret.com/rest/server.api"
_DEFAULT_TIMEOUT_SECONDS = 10.0
_MAX_SEARCH_LENGTH = 120

_access_token: str | None = None
_token_expiry = 0.0
_token_lock = threading.RLock()


class FatSecretConfigurationError(RuntimeError):
    """Raised when required provider configuration is absent or invalid."""


def _credentials() -> tuple[str, str]:
    client_id = os.getenv("FATSECRET_CLIENT_ID", "").strip()
    client_secret = os.getenv("FATSECRET_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise FatSecretConfigurationError("FatSecret credentials are not configured.")
    return client_id, client_secret


def _timeout_seconds() -> float:
    raw = os.getenv("FATSECRET_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS))
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise FatSecretConfigurationError("FATSECRET_TIMEOUT_SECONDS must be a number.") from exc
    if not math.isfinite(timeout) or not 1 <= timeout <= 60:
        raise FatSecretConfigurationError("FATSECRET_TIMEOUT_SECONDS must be between 1 and 60.")
    return timeout


def _provider_error(status_code: int) -> str:
    if status_code in (401, 403):
        return "FatSecret authentication failed."
    if status_code == 429:
        return "FatSecret request limit reached."
    if status_code >= 500:
        return "FatSecret is temporarily unavailable."
    return "FatSecret rejected the request."


def _response_json(response: requests.Response) -> dict[str, Any] | None:
    if not 200 <= response.status_code < 300:
        return None
    try:
        payload = response.json()
    except (ValueError, requests.exceptions.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def get_access_token() -> str:
    """Return a cached OAuth2 application token, refreshing before expiry."""
    global _access_token, _token_expiry

    with _token_lock:
        now = time.monotonic()
        if _access_token and now < _token_expiry:
            return _access_token

        client_id, client_secret = _credentials()
        response = requests.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials", "scope": "basic"},
            auth=(client_id, client_secret),
            timeout=_timeout_seconds(),
        )
        payload = _response_json(response)
        if payload is None:
            raise RuntimeError(
                _provider_error(response.status_code)
                if not 200 <= response.status_code < 300
                else "FatSecret returned an invalid token response."
            )

        token = payload.get("access_token")
        try:
            if isinstance(payload.get("expires_in"), bool):
                raise ValueError("boolean expiry is invalid")
            expires_in = float(payload.get("expires_in"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("FatSecret returned an invalid token expiry.") from exc
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError("FatSecret returned an invalid access token.")
        if not math.isfinite(expires_in) or expires_in <= 0:
            raise RuntimeError("FatSecret returned an invalid token expiry.")

        refresh_margin = min(60.0, max(1.0, expires_in * 0.1))
        _access_token = token
        _token_expiry = time.monotonic() + max(0.0, expires_in - refresh_margin)
        return token


def _invalidate_access_token(expected_token: str | None = None) -> None:
    global _access_token, _token_expiry
    with _token_lock:
        if expected_token is not None and _access_token != expected_token:
            return
        _access_token = None
        _token_expiry = 0.0


def _api_get(params: dict[str, str]) -> dict[str, Any]:
    """Call the provider and retry once after an expired/invalid bearer token."""
    for attempt in range(2):
        token = get_access_token()
        response = requests.get(
            API_URL,
            headers={"Authorization": f"Bearer {token}"},
            params={**params, "format": "json"},
            timeout=_timeout_seconds(),
        )
        if response.status_code == 401 and attempt == 0:
            _invalidate_access_token(token)
            continue
        payload = _response_json(response)
        if payload is None:
            if not 200 <= response.status_code < 300:
                return {"error": _provider_error(response.status_code)}
            return {"error": "FatSecret returned an invalid response."}
        return payload
    return {"error": "FatSecret authentication failed."}


def get_food_details(food_id: str) -> dict[str, Any]:
    """Fetch a food record by FatSecret food ID."""
    if not isinstance(food_id, str) or not food_id.strip() or len(food_id) > 64:
        return {"error": "Food identifier is invalid."}
    try:
        return _api_get({"method": "food.get", "food_id": food_id.strip()})
    except FatSecretConfigurationError:
        return {"error": "FatSecret integration is not configured."}
    except (requests.RequestException, RuntimeError):
        return {"error": "FatSecret is temporarily unavailable."}


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number


def normalize_food_response(food_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider data using a gram-based reference serving only."""
    food = food_data.get("food") if isinstance(food_data, dict) else None
    if not isinstance(food, dict):
        return {"error": "FatSecret returned an invalid food record."}

    food_name, food_id = food.get("food_name"), food.get("food_id")
    if not isinstance(food_name, str) or not food_name.strip():
        return {"error": "FatSecret food record is missing its name."}
    if isinstance(food_id, bool) or food_id is None or not str(food_id).strip():
        return {"error": "FatSecret food record is missing its identifier."}

    servings_data = food.get("servings")
    servings = servings_data.get("serving") if isinstance(servings_data, dict) else None
    if isinstance(servings, dict):
        servings = [servings]
    if not isinstance(servings, list) or not servings:
        return {"error": "FatSecret food record has no serving information."}

    valid_servings = []
    for serving in servings:
        if not isinstance(serving, dict):
            continue
        unit = serving.get("metric_serving_unit")
        grams = _finite_nonnegative(serving.get("metric_serving_amount"))
        if isinstance(unit, str) and unit.strip().lower() in {"g", "gm", "gram", "grams"} and grams and grams > 0:
            valid_servings.append((serving, grams))

    if not valid_servings:
        return {"error": "FatSecret food record has no valid gram-based serving."}

    reference, reference_grams = next(
        ((item, grams) for item, grams in valid_servings if math.isclose(grams, 100.0)),
        valid_servings[0],
    )
    nutrients: dict[str, float] = {}
    required = ("calories", "protein", "fat", "carbohydrate")
    for key in required:
        value = _finite_nonnegative(reference.get(key))
        if value is None:
            return {"error": "FatSecret serving contains invalid nutrition values."}
        nutrients[key] = value
    for key in ("fiber", "sugar", "sodium"):
        value = _finite_nonnegative(reference.get(key, 0))
        if value is None:
            return {"error": "FatSecret serving contains invalid nutrition values."}
        nutrients[key] = value

    return {
        "food_name": food_name.strip(),
        "food_id": str(food_id),
        "food_type": food.get("food_type"),
        "food_url": food.get("food_url"),
        "reference_serving": {
            "description": reference.get("serving_description"),
            "grams": reference_grams,
            "unit": reference.get("metric_serving_unit"),
            **nutrients,
        },
        "servings": servings,
    }


def search_food(food_name: str) -> dict[str, Any]:
    """Search for a food and normalize the selected record."""
    if not isinstance(food_name, str) or not food_name.strip() or len(food_name.strip()) > _MAX_SEARCH_LENGTH:
        return {"error": "Food search query must contain 1 to 120 characters."}

    try:
        search_data = _api_get({
            "method": "foods.search",
            "search_expression": food_name.strip(),
            "max_results": "5",
        })
        if "error" in search_data:
            return search_data
        search_results = search_data.get("foods", {}).get("food")
        if isinstance(search_results, dict):
            search_results = [search_results]
        if not isinstance(search_results, list) or not search_results:
            return {"error": "Food not found."}

        first_food = search_results[0]
        if not isinstance(first_food, dict):
            return {"error": "FatSecret returned an invalid search result."}
        food_id = first_food.get("food_id")
        if food_id is None:
            return {"error": "FatSecret search result is missing its identifier."}

        details = get_food_details(str(food_id))
        if "error" in details:
            return details
        return normalize_food_response(details)
    except FatSecretConfigurationError:
        return {"error": "FatSecret integration is not configured."}
    except (requests.RequestException, RuntimeError):
        return {"error": "FatSecret is temporarily unavailable."}
