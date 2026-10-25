"""Open-Meteo client: city name -> coordinates -> current weather. No API key needed."""

from __future__ import annotations

from typing import Any

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 10.0

# WMO weather interpretation codes used by Open-Meteo.
WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


async def fetch_current_weather(city: str, client: httpx.AsyncClient | None = None) -> dict[str, Any]:
    """Geocode ``city`` and return its current weather conditions.

    Raises:
        ValueError: if the city cannot be found.
        httpx.HTTPError: on network or API failures.
    """
    city = city.strip()
    if not city:
        raise ValueError("city must not be empty")

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
    try:
        geo = await client.get(GEOCODING_URL, params={"name": city, "count": 1, "language": "en"})
        geo.raise_for_status()
        matches = geo.json().get("results") or []
        if not matches:
            raise ValueError(f"could not find a city named {city!r}")
        place = matches[0]

        forecast = await client.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "weather_code,wind_speed_10m,wind_direction_10m",
                "timezone": "auto",
            },
        )
        forecast.raise_for_status()
        current = forecast.json()["current"]
    finally:
        if owns_client:
            await client.aclose()

    return {
        "city": place["name"],
        "region": place.get("admin1"),
        "country": place.get("country"),
        "observed_at": current["time"],
        "temperature_c": current["temperature_2m"],
        "feels_like_c": current["apparent_temperature"],
        "humidity_percent": current["relative_humidity_2m"],
        "conditions": WMO_CODES.get(current["weather_code"], f"unknown (code {current['weather_code']})"),
        "wind_speed_kmh": current["wind_speed_10m"],
        "wind_direction_deg": current["wind_direction_10m"],
    }
