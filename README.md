# GTA VI trophy earn-rate forecaster

Entry for make.fun `gta6-trophy-rarity-forecast-01`: forecast every GTA VI
trophy's Exophase earn rate at 2026-11-26T00:00:00Z as five quantiles,
scored by pinball loss on the logit scale.

## Model

Work in counts, not rates: a trophy's rate is `E_i(t) / N(t)`, earned counts
over players tracked. Exophase refreshes every few hours, so reads between
refreshes are stale and at a refresh N jumps and rates fall; fitting E and N
separately handles that dilution instead of chasing it as a rate trend.

Candidates, picked by a rolling-origin CV that predicts the last read of the
fitting window (a short-horizon gate would pick persistence even when a trend
pays over the 4.5-day final horizon):

- **A** persistence (median = last rate)
- **B** counts log-linear: ln N and ln E_i linear in ln(1+h), per-trophy slopes
  shrunk toward tier means, anchored at the last read
- **C** damped B (slopes x0.6, median capped at tier ceilings)
- **D** saturating counts: E_i(t) = N(t) * R_i * (1 - exp(-h/tau_i))
- **E** logit-blend of D and persistence (50/50)
- **F** damped B (slopes x0.5)

The fit window is the last 24h before the checkpoint. Spread = slope-SE x
horizon + refresh-jump noise + random-walk floor. A flat history (N never
changed) means a refresh is due before the target: medians shift x0.2 with a
wide spread, since the first observed refresh multiplied N by ~6.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, deterministic.
