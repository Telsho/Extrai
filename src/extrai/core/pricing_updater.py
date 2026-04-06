# extrai/core/pricing_updater.py
import json
import os
from datetime import datetime, timedelta

import requests
from jsonschema import ValidationError, validate

PRICING_URL = "https://www.llm-prices.com/current-v1.json"
CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "model_prices.json")
CACHE_EXPIRATION = timedelta(days=1)

PRICING_SCHEMA = {
    "type": "object",
    "properties": {
        "updated_at": {"type": "string", "format": "date"},
        "prices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "vendor": {"type": "string"},
                    "name": {"type": "string"},
                    "input": {"type": "number"},
                    "output": {"type": "number"},
                    "input_cached": {"type": ["number", "null"]},
                },
                "required": ["id", "vendor", "name", "input", "output"],
            },
        },
    },
    "required": ["updated_at", "prices"],
}


def fetch_pricing_data():
    """Fetches pricing data from the remote URL and validates it."""
    response = requests.get(PRICING_URL)
    response.raise_for_status()
    data = response.json()

    try:
        validate(instance=data, schema=PRICING_SCHEMA)
    except ValidationError as e:
        # Handle validation error, e.g., log it or raise an exception
        print(f"Pricing data validation error: {e}")
        return None

    return data


def save_pricing_data(data):
    """Saves pricing data to the local cache."""
    if data is None:
        return

    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_pricing_data():
    """Loads pricing data from the local cache."""
    if not os.path.exists(CACHE_FILE):
        return None
    with open(CACHE_FILE) as f:
        return json.load(f)


def is_cache_stale():
    """Checks if the cached pricing data is stale."""
    if not os.path.exists(CACHE_FILE):
        return True

    last_modified_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
    return datetime.now() - last_modified_time > CACHE_EXPIRATION


def update_prices_if_stale():
    """Updates the pricing data if the cache is stale."""
    if is_cache_stale():
        data = fetch_pricing_data()
        save_pricing_data(data)
