# Cross-checks against independent tools

Everything in this repo shares one codebase and data pipeline, so only an outside tool can
catch a mistake every model shares.

## 2026-10-22: testfol.io backtest (run by an outside assistant on the live site)

Settings: full common history 2000-06-29 → 2026-10-20 (VIPSX starts late June 2000), monthly
rebalancing, total return, nominal, no flows. Our engine: `regret.constant_mix_path` on the
same funds' monthly replay series, 2000-07 → 2026-09.

| | Engine: A | testfol.io: A | Engine: B | testfol.io: B |
|---|---|---|---|---|
| Weights | VTSMX 45 / VGTSX 27 / VISVX 18 / VBIIX 5 / VIPSX 5 | same | 54 / 27 / 9 / 5 / 5 | same |
| CAGR | 7.7% | **7.72%** | 7.5% | **7.57%** |
| 2008 | −35% | **−34.84%** | −35% | **−35.31%** |
| 2022 | −16% | **−15.97%** | −17% | **−16.87%** |
| Max drawdown | −50% (month-end Feb 2009) | **−53.60%** (daily, 2009-03-09) | −49% | **−53.22%** |

**Verdict: pass.** Returns agree to within ~0.1pp/yr and calendar years to within ~0.5pp.

The max drawdown differs by 3.6–4pp because the engine uses month-end prices (assumption S8)
and misses the intra-month low of 2009-03-09. **Read every drawdown in this repo as ~3–4pp
shallower than a daily statement would show.**

The cross-check also independently confirms A ≥ B: higher CAGR (+0.15pp/yr), slightly
better 2008 and 2022, and a near-identical max drawdown.
