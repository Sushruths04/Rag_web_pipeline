# Note: cost tracking is an optional convenience, not a core feature

Date: 2026-08-01
Status: accepted — implemented in the commit that adds this file

## The claim

We should not present a dollar figure as if it were authoritative. The API
provider (Nebius, OpenAI, whoever holds the key) already meters spend, and
their number is the one that gets billed. Ours cannot be better than theirs,
and in this codebase it was substantially worse.

## Why our number was untrustworthy

Three independent problems, each enough on its own.

**1. The token counts were fabricated when the real ones were available.**

`app/stages/graft.py` computed:

```python
tok_in = len(prompt) // 4   # char-estimate
tok_out = len(out) // 4
```

Meanwhile `src/rag_gt/core/llm.py::APILLM.generate` posts to
`/chat/completions` and the response body contains a real `usage` block —
`prompt_tokens`, `completion_tokens`, `total_tokens`, counted by the provider's
own tokenizer. `generate()` returned only the message text and discarded it.

So the pipeline was estimating a quantity it was being handed for free. A
`chars/4` heuristic is wrong by a wide margin on exactly the content this
pipeline sends: ISO clause text is dense with numbers, units, symbols and
standard identifiers, all of which tokenize far worse than 4 characters per
token.

**2. The prices were a hardcoded table that goes stale silently.**

A provider can change its rates whenever it likes. Nothing in this repo would
notice. A stale table produces a confidently wrong number, which is worse than
no number.

**3. We cannot control cost, only stop.**

The budget cap does not negotiate with the provider — it aborts the run. What
actually bounds spend is a stopping condition, and a stopping condition does
not need prices at all. Tokens and call counts are sufficient, and unlike
dollars they are directly measurable.

## What this changes

Cost estimation moves from a required, load-bearing, run-blocking input to an
optional convenience that is off by default.

| | before | after |
|---|---|---|
| Primary run bound | dollar cap, needed prices | **token cap**, needs nothing |
| Prices | **mandatory**, run refused without them | optional, off by default |
| Token counts | `len(text) // 4` | **real `usage` from the API**, estimate only as fallback |
| Dollar figure | presented as fact | opt-in, labelled an estimate, provider named as the authority |

The honesty guarantee from the original `LivePricingRequired` guard is kept and
is now easier to satisfy: a run whose cost was never established does not
render as `$0.00`. It renders as "not tracked", which is what it is.

## What did NOT change

The run is still bounded. Removing the dollar cap without replacing it would
have been a regression — an unpriced run would add `0.0` per call and never
trip anything. The token cap is the replacement, and because token counts are
now real rather than estimated, it is a **tighter** bound than the dollar cap
ever was, not a looser one.

## Where the real number lives

The provider's dashboard. For the endpoint currently configured:
`https://tokenfactory.nebius.com` → project → usage/billing. The UI says so
rather than implying our estimate is the last word.

## Consequence worth stating plainly

Anything that previously quoted a run's cost in dollars — reports, the
leaderboard, this repo's own status documents — was quoting a number derived
from fabricated token counts. Those figures should be read as rough magnitude
indicators, not as spend. Real token totals are recorded from now on, so
future runs can be reconciled against the provider's bill.
