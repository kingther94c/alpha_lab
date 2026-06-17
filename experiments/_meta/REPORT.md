# Is alpha_lab worth converting to "team mode"? — evaluation & recommendation

*Worktree experiment, 2026-06-16/17. Scratch/uncommitted — keep, commit, or delete as you like.
The live A/B/C **ran** (after two earlier session-limit failures) and the numbers below are measured,
not predicted: see [`grade.py`](grade.py) + [`team-eval-experiment.js`](team-eval-experiment.js).*

## TL;DR (recommendation — confirmed by the live run)

**Keep team mode opt-in; keep solo as the default; lean on the project-tuned checklist.** The live
experiment confirmed the call and sharpened *why*:
- **Solo is strong on mechanical/quant work** — it independently nailed the hardest *quant* trap
  (cadence-aware annualization), better than the team's own first-cut builder.
- **But solo missed the execution-leg honesty traps** — it reported a bot's drawdown as **−3.7%**
  when the strategy truly drew down **−36.2%** (it never de-fauceted), and left cost-of-cash off.
- **A *generic* team is no fix** — it shipped the cleanest, most-tested code of all three and was
  *equally* wrong on all three domain traps, because a generic reviewer has no reason to demand them.
- **Only the *project-tuned* team was honest on all three**, and it got there via adversarial review
  catching its builder's bugs + enforcing the repo's non-negotiables.

So: **good tuned instructions > solo > generic instructions.** The value is the project-tuning, not
the fan-out per se — which is exactly the opt-in rule already in `AGENTS.md`.

---

## 1. The question

> 这个项目值得变成 teams 模式吗?  +  有必要对不同成员给 instruction 吗，如果给的不好是不是还不如让 claude 自己来?

Decide (a) whether team mode should be the default here, and (b) whether differentiated per-role
instructions beat just letting one capable agent work solo.

## 2. Method — a controlled A/B/C ([`team-eval-experiment.js`](team-eval-experiment.js))

Same feature, **neutral spec** ("add a perf summary — annualized return / vol / Sharpe / max-drawdown
— for a paper-trading bot, reading its stored equity path"), built three ways in **isolated dirs**;
single variable = the team configuration:

- **A — solo:** one generic agent, no reviewer.
- **B — tuned team:** `engineer` builds → `quant-skeptic` + `execution-safety` + `research-lead`
  review → `engineer` revises.
- **C — generic team:** generic builder → one *generic* reviewer (no project priming) → revise.

**Three deliberately-unmentioned traps** were the discriminator — each a real repo non-negotiable the
neutral spec never names, so an arm only handles it on merit:
1. **cost-of-cash** — Sharpe should be excess-of-rf (AGENTS.md:27), not raw.
2. **de-faucet** — must use the de-fauceted strategy equity (`store.all_strategy_equity` / subtract
   the faucet offset), not raw `total_equity` (a demo faucet inflates the base ~10× and hides drawdowns).
3. **annualization** — the runner marks **per-cycle/intraday** (`schema.py:24`, `interval_min=15`), so a
   hardcoded `×√252` is both the wrong factor (crypto≈365) and cadence-blind.

> **Operational finding (real).** The live run only completed on the **third attempt** — the fan-out was
> killed by session rate limits twice (2026-06-14 per the `p7-product-hardening` memory; 2026-06-15
> here). It finally ran for ~13 min / **9 agents / 452k tokens**. A mode this rate-limit-fragile and
> token-heavy (~9 agents for one small feature) is a poor *default* on this account.

## 3. Measured results

Objective grader ([`grade.py`](grade.py)): one controlled demo bot — capital 10k, **90k faucet offset**
(raw total starts at 100k), 60 daily marks with a deliberate strategy drawdown — each arm's store
wrapper called **with its defaults**. Ground truth: de-fauceted strategy = **−20.7%** total, **−36.2%**
maxDD; raw faucet-inflated = −2.1% / −3.7%.

```
arm  wrapper (defaults)                 annRet     annVol     Sharpe      maxDD
A    summarize_bot()  rf=0              -0.121      0.032     -4.090     -0.037   <- tracks RAW (faucet-blind)
B    bot_perf()  rf=0.04 ppy~inferred   -0.762      0.379     -3.694     -0.362   <- tracks STRATEGY (honest)
C    bot_perf()  rf=0 periods=252       -0.121      0.026     -3.397     -0.037   <- tracks RAW (faucet-blind)
A+   summarize_bot(rf=0.04)             -0.121      0.032     -5.360     -0.037   <- A CAN do cost-of-cash, just off by default
```

Per-trap, and the full scorecard (from reading the three final `perf.py` + the grader):

| dimension | **A** solo | **B** tuned team | **C** generic team |
|---|---|---|---|
| **De-faucet** (honest base) | ✗ raw → reports **−3.7%** DD (true −36.2%) | ✓ de-fauceted → **−36.2%** DD | ✗ raw → **−3.7%** DD |
| **Cost-of-cash** (default rf) | ⚠ plumbed but **OFF** (rf=0) | ✓ **ON** (rf=0.04, matches runner) | ✗ OFF, reviewer never raised it |
| **Annualization** (cadence) | ✓ median-gap infer (~365) | ✓ median-gap (review fixed builder's exploding n/elapsed) | ✗ hardcoded **252**, cadence-blind |
| Drawdown == kill-switch def | ✓ reuses `risk.drawdown` | ✓ reuses `risk.drawdown` | ✗ reimplemented (own −110% bug, caught by generic review) |
| Generic edge-cases | partial | good (post-review) | ✓✓ most coverage |
| Caught builder's *own* bugs | — (no review) | ✓ skeptic ran a real path → `sharpe=1.5e15`; forced rf + de-faucet alignment | ✓ but **only generic** bugs |
| tests / ruff | 10 / clean | 25 / clean | 19 / clean |
| agents (cost) | **1** | **5** | **3** |

## 4. What the live run showed about the team *mechanism*

- **The team's value was the review, not the builder.** B's first-cut builder was actually *worse* than
  solo on annualization (it inferred periods from `n/elapsed`, which the `quant-skeptic` showed explodes
  to `ann_return=3,602%`, `sharpe=1.5e15` on a benign +3%/3-day path at the real 15-min cadence). The
  **adversarial review caught it** and the revise fixed it. `quant-skeptic` + `research-lead` *both*
  flagged rf=0 as a blocker against the repo's "beat cash" standard (citing `runner.py:30-36`), and
  `execution-safety` hardened the de-faucet row-alignment. That's the team working as designed: an
  independent mind with no authorship stake catches what the author cannot.
- **The generic team caught zero domain sins.** C's reviewer found genuine *generic* bugs (a drawdown
  worse than −100% on negative equity; a 2-point curve emitting a 2.7e10 "annualized return") and the
  revise fixed them — so C ended with the **most** edge-case tests. Yet it never once mentioned
  de-faucet, cost-of-cash, or the 252-vs-365/cadence issue, because those are not generic code smells.
  C is the cautionary tale: **clean, well-tested, and silently wrong on the actual product.**

## 5. Verdicts

- **A vs B (does a team help?)** — Yes, on the traps that matter. Solo was competent (and beat the team's
  builder on cadence) but shipped a faucet-blind, rf=0 summary that would tell the cockpit a −36%
  drawdown was −3.7%. The tuned team was the only honest one. This mirrors the real P7 history: solo
  shipped 8 hardening items + 181 green tests but **deprioritized cost-of-cash** (a written
  non-negotiable) until the *user* caught it (`runner.py:30-36` shipped only after). The team's payoff
  is **catching the author's domain blind-spots**, not writing code.
- **B vs C (does tuning matter?)** — Decisively yes. C had a reviewer and *more* tests than B, yet was
  identical to solo on all three traps. **"Add a reviewer" buys generic correctness; "add a
  project-tuned reviewer" buys the cardinal-sin coverage this repo actually needs.** Thin/generic
  per-role instructions ≈ worthless here; the entire ROI is the project-grounded checklist.

## 6. Limitations (honest)

- **Reviewer cross-pollination.** The `quant-skeptic` reviewing B *read* sibling `arm_a` (only *builders*
  were forbidden from reading siblings) and cited it. So B's *annualization fix* is partly informed by
  seeing A's median-gap idea. It does **not** taint the headline: de-faucet and rf=0.04 came from the
  tuned lenses citing `runner.py`/AGENTS.md (A never de-fauceted, so B couldn't copy that), and the
  explosion was found by the skeptic *running a real path*. Tighten the prompt next time.
- **Selection bias (by design).** The feature was chosen to be trap-dense — that's where a team can earn
  its keep. A trap-free feature would show team mode as pure overhead. This is *why* the recommendation
  is conditional on trap-density, not blanket.
- **n=1 per arm, single feature.** Directional, not a statistic.

## 7. Recommendation (decision rule — unchanged, now evidence-backed)

1. **Default = solo**, using the four role `.md` files as a **standing checklist** in the main loop —
   no extra agents. ~80% of the value, ~0 cost.
2. **Fan out the real team only on two triggers** — where author-bias + cardinal-sin risk peak:
   - a result/backtest that looks **too good** → `quant-skeptic` (leakage, cost-of-cash, full-sample,
     survivorship);
   - an **execution-leg change near money/gates** → `execution-safety` (never-auto-live, kill-switch,
     de-faucet).
3. **Keep the lenses ruthlessly project-tuned; treat each as a living checklist.** The generic-team arm
   proved a non-tuned reviewer misses this repo's sins. When a lens misses something, add that line.
4. **Don't grow it heavier.** 4 roles cover the real failure modes; PM/architect fold into `research-lead`.

**Verdict: NO, don't make alpha_lab team-by-default. YES, keep the current opt-in, project-tuned team as
a sharp second-opinion tool for too-good backtests and money/gate changes — and use the role files as an
always-on checklist the rest of the time.** This is what `AGENTS.md` "Team development mode" now says; the
live A/B/C validates it.

## 8. Artifacts / reproduce

- [`team-eval-experiment.js`](team-eval-experiment.js) — the workflow (re-run: `Workflow({ scriptPath: ".../experiments/_meta/team-eval-experiment.js" })`).
- [`grade.py`](grade.py) — objective grader (`& D:\conda\envs\py313\python.exe experiments\_meta\grade.py`).
- The three implementations: `experiments/team_eval/arm_{a,b,c}/perf.py` (+ `test_perf.py`).
