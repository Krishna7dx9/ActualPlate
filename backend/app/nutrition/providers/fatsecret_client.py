import os
import time
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# FatSecret API credentials
CLIENT_ID = os.getenv("FATSECRET_CLIENT_ID") or ""
CLIENT_SECRET = os.getenv("FATSECRET_CLIENT_SECRET") or ""

if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError(
        "FATSECRET_CLIENT_ID and FATSECRET_CLIENT_SECRET must be set."
    )

# OAuth endpoint
TOKEN_URL = "https://oauth.fatsecret.com/connect/token"

# FatSecret REST endpoint
API_URL = "https://platform.fatsecret.com/rest/server.api"

# In-memory cache to avoid requesting a new OAuth token
# before the current one expires.
_access_token = None
_token_expiry = 0


def get_access_token():
    """
    Get a valid OAuth access token.

    FatSecret tokens expire after a period of time.
    We cache the token in memory to avoid requesting a new
    token for every API call.
    """

    global _access_token, _token_expiry

    current_time = time.time()

    # Return cached token if still valid
    if _access_token and current_time < _token_expiry:
        return _access_token

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "scope": "basic"
        },
        auth=(CLIENT_ID, CLIENT_SECRET),
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    _access_token = data["access_token"]

    # Refresh one minute before expiry
    _token_expiry = current_time + data["expires_in"] - 60

    return _access_token


def get_food_details(food_id):
    """
    Fetch complete nutrition details for a FatSecret food.

    Parameters:
        food_id (str)

    Returns:
        dict: Complete FatSecret food record including servings and nutrition information.
    """

    token = get_access_token()

    response = requests.get(
        API_URL,
        headers={
            "Authorization": f"Bearer {token}"
        },
        params={
            "method": "food.get",
            "food_id": food_id,
            "format": "json"
        },
        timeout=10
    )

    if response.status_code == 401:
        return {
            "error": "FatSecret authentication failed",
            "details": "OAuth token expired or invalid credentials"
        }
    elif response.status_code == 429:
        return {
            "error": "FatSecret rate limit exceeded",
            "details": "Too many requests — slow down or upgrade plan"
        }
    elif response.status_code >= 500:
        return {
            "error": "FatSecret server error",
            "details": response.text
        }
    elif response.status_code != 200:
        return {
            "error": "Unable to fetch food details",
            "details": response.text
        }

    return response.json()

def normalize_food_response(food_data):
    """
    Convert the raw FatSecret response into Calx's internal format.

    Keeping our own response format prevents the rest of the
    application from depending directly on FatSecret's JSON structure.
    """

    food = food_data.get("food")

    if not food:
        return {
            "error": "Invalid FatSecret response"
        }

    servings = food.get("servings", {}).get("serving")

    if not servings:
        return {
            "error": "No serving information available"
        }

    # FatSecret sometimes returns a dictionary when there is only one serving.
    if isinstance(servings, dict):
        servings = [servings]

    # -----------------------------------------------------
    # Find the 100 g serving.
    #
    # If unavailable, use the first serving as the reference.
    # -----------------------------------------------------

    reference = None

    for serving in servings:

        if (
            serving.get("metric_serving_unit") in ("g", "gm")
            and float(serving.get("metric_serving_amount", 0)) == 100
        ):
            reference = serving
            break

    if reference is None:
        reference = servings[0]

    return {
        "food_name": food["food_name"],
        "food_id": food["food_id"],
        "food_type": food["food_type"],
        "food_url": food["food_url"],

        # Reference nutrition used by later AI calculations.
        "reference_serving": {
            "description": reference["serving_description"],
            "grams": float(reference["metric_serving_amount"]),
            "unit": reference["metric_serving_unit"],

            "calories": float(reference["calories"]),
            "protein": float(reference["protein"]),
            "fat": float(reference["fat"]),
            "carbohydrate": float(reference.get("carbohydrate", 0)),
            "fiber": float(reference.get("fiber", 0)),
            "sugar": float(reference.get("sugar", 0)),
            "sodium": float(reference.get("sodium", 0)),
        },

        # Keep every serving returned by FatSecret.
        "servings": servings
    }


def search_food(food_name):
    """
    Search FatSecret for the best matching food.

    Flow:

    AI
        ↓
    foods.search
        ↓
    first result
        ↓
    food_id
        ↓
    food.get
        ↓
    detailed nutrition
    """

    token = get_access_token()

    response = requests.get(
        API_URL,
        headers={
            "Authorization": f"Bearer {token}"
        },
        params={
            "method": "foods.search",
            "search_expression": food_name,
            "format": "json",
            "max_results": 1
        },
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    search_results = data.get("foods", {}).get("food")

    if not search_results:
        return {
            "error": "Food not found"
        }

    # FatSecret returns either a list or a single object
    if isinstance(search_results, list):
        first_food = search_results[0]
    else:
        first_food = search_results

    # Fetch the detailed nutrition record using the selected food ID.
    food_id = first_food["food_id"]

    # Retrieve the complete FatSecret food record.
    food_details = get_food_details(food_id)

    # Convert it into Calx's internal response model.
    return normalize_food_response(food_details)