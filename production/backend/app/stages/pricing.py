"""Per-token price resolution for live runs.

Why this exists
---------------
``LivePricingRequired`` made ``price_in_per_mtok`` / ``price_out_per_mtok``
mandatory for every live run. The intent was sound: defaulting them to 0.0
made every live run report "$0.00" while real API spend accrued against the
budget cap, so an accounting failure looked like a free run.

The implementation put the burden in the wrong place. The user supplies an API
key, so the provider is already known; and the token counts those prices
multiply are a ``len(text) // 4`` character estimate, so demanding two exact
decimal figures to feed an approximation is disproportionate. Blocking the run
on it is worse still.

Resolution order, most trustworthy first:

  1. explicit    -- values passed in the run config; always wins.
  2. provider    -- looked up from the API base URL (and model, where a
                    provider's prices differ by model).
  3. unknown     -- proceed, but report cost as UNKNOWN rather than $0.00 and
                    fall back to a token ceiling so the run is still bounded.

Case 3 is the part that preserves the original guarantee: an unpriced run is
never allowed to render as a free one.

Prices are USD per 1,000,000 tokens. They are a convenience default, not a
billing source of truth -- a provider can change them at any time, which is
why an explicit config value always overrides.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

# host substring -> {model substring (lowercase) or "*": (price_in, price_out)}
_PROVIDER_PRICES: dict[str, dict[str, tuple[float, float]]] = {
    "api.tokenfactory.nebius.com": {
        "qwen3-235b": (0.20, 0.60),
        "qwen3-30b": (0.10, 0.30),
        "llama-3.3-70b": (0.13, 0.40),
        "deepseek-v3": (0.50, 1.50),
        "*": (0.20, 0.60),
    },
    "api.studio.nebius.": {
        "*": (0.20, 0.60),
    },
    # Institutional / self-hosted endpoints with no metering. 0.0 here is a
    # statement about the endpoint, not a missing value, so it is NOT "unknown".
    "llm.hpc.itc.rwth-aachen.de": {"*": (0.0, 0.0)},
    "localhost": {"*": (0.0, 0.0)},
    "127.0.0.1": {"*": (0.0, 0.0)},
    "0.0.0.0": {"*": (0.0, 0.0)},
}

# Used only when prices could not be resolved, so the dollar cap is meaningless.
DEFAULT_MAX_TOKENS_TOTAL = 5_000_000


@dataclass(frozen=True)
class ResolvedPricing:
    price_in_per_mtok: float
    price_out_per_mtok: float
    source: str            # "explicit" | "provider" | "unknown"
    provider: Optional[str] = None

    @property
    def is_known(self) -> bool:
        return self.source != "unknown"


def _host(base_url: str) -> str:
    if not base_url:
        return ""
    parsed = urlparse(base_url if "//" in base_url else f"//{base_url}")
    return (parsed.hostname or "").lower()


def lookup_provider_prices(
    base_url: str, model: str
) -> Optional[tuple[str, float, float]]:
    """(provider_host, price_in, price_out) for a known endpoint, else None."""
    host = _host(base_url)
    if not host:
        return None
    model_l = (model or "").lower()
    for known_host, table in _PROVIDER_PRICES.items():
        if known_host.rstrip(".") not in host:
            continue
        for model_key, prices in table.items():
            if model_key != "*" and model_key in model_l:
                return known_host, prices[0], prices[1]
        if "*" in table:
            return known_host, table["*"][0], table["*"][1]
    return None


def resolve_pricing(
    config: dict,
    *,
    base_url: str = "",
    model: str = "",
) -> ResolvedPricing:
    """Resolve per-token pricing without ever blocking the run."""
    price_in = config.get("price_in_per_mtok")
    price_out = config.get("price_out_per_mtok")
    if price_in is not None and price_out is not None:
        return ResolvedPricing(float(price_in), float(price_out), "explicit")

    found = lookup_provider_prices(base_url, model)
    if found is not None:
        host, p_in, p_out = found
        # A single explicitly-supplied side still wins over the table.
        return ResolvedPricing(
            float(price_in) if price_in is not None else p_in,
            float(price_out) if price_out is not None else p_out,
            "provider",
            host,
        )

    return ResolvedPricing(
        float(price_in) if price_in is not None else 0.0,
        float(price_out) if price_out is not None else 0.0,
        "unknown",
    )
