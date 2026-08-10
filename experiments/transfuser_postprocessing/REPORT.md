# Transfuser Post/Pre-Processing Improvement Report

Status: **FINAL** (Round 1-6), plus a **Round 7-8 follow-up** with two more
candidates. Read the whole document, especially "Round 7-8" — it contains
the highest-scoring config found, alongside a safety caveat serious enough
that it changes the recommendation. Don't stop at the ranked list.

## TL;DR

**Production recommendation stays Candidate A alone** (trajectory optimizer,
`max_deviation=0.5m, smoothness_weight=3.0`): a clean, unconditional
improvement (+11.0% score, zero downside on any safety metric, confirmed at
N=100). A later-discovered candidate (`B+C`, see Round 7-8) scores far
higher (0.818 vs 0.640 at N=100) but comes with a **10x increase in at-fault
collision rate** — a real safety regression the score number alone hides.
I did not deploy it. It's a genuinely promising lead for future work, not
a drop-in replacement.

Out of the original three hypothesized levers (image preprocessing
correctness, heading smoothing, trajectory optimization), only trajectory
optimization produced a clean, reproducible improvement. Two hypothesized
correctness risks (BGR/RGB channel order, vertical crop) turned out to be
non-issues — real negative results, not bugs, confirmed by direct testing.

**Definitive result for Candidate A, full local scene set (N=100,
apples-to-apples):**

| Config | score_mean | collision_at_fault | collision_any | Δ score |
|---|---|---|---|---|
| **Baseline** (all toggles off) | 0.5764 | 0.01 | 0.28 | — |
| **Candidate A** (trajectory optimizer, tuned) | **0.6399** | 0.01 | 0.28 | **+11.0%** |

Collision rate is unchanged — the improvement is 100% from better progress
on drives that don't collide, not from avoiding collisions. See "What the
improvement is NOT" below, and see Round 7-8 for a candidate that *does*
change collision behavior, with tradeoffs.

Smaller-scale (N=20) exploration that led to this result:

| Config | score_mean (N=20) | Δ vs baseline |
|---|---|---|
| **Baseline** (all toggles off) | 0.6067 | — |
| Stage 0: BGR channel order | 0.5941 | -2.1% (no real effect, within noise, worse if anything) |
| Stage 0: bottom crop | 0.6026 | -0.7% (no real effect) |
| Stage 1: heading stabilize (default params) | 0.6067 | +0.0% (zero effect) |
| Stage 1: heading stabilize (aggressive params) | 0.6068 | +0.0% (zero effect even pushed hard) |
| Stage 2: trajectory optimizer (default weights) | 0.6307 | +4.0% |
| Stage 2: trajectory optimizer (smoothness_weight=3x) | 0.6361 | +4.8% |
| Stage 1+2 combined | 0.6365 | +4.9% (no added benefit over Stage 2 alone) |

Note the N=100 improvement (+11.0%) is noticeably larger than the N=20
estimate (+4.8%) — a reminder that N=20 alone would have *understated* the
real effect; the larger, more diverse scene sample gave the optimizer more
room to help.

**Recommendation: ship Stage 2 (trajectory optimizer) alone.** Stage 0 and
Stage 1 add complexity/latency for zero measured benefit; Stage 2 alone
matches the combined result.

## Methodology

All runs use `+e2e_challenge_nuplan=full scenes.limit_to_first_n=N`
(guaranteed to hit only locally-available MTGS assets — verified our
downloaded shard is exactly the alphabetically-first 100 of the full
1485-scene navtest list, sorted+truncated identically by the wizard).
Same scene set across all N=20 runs, so comparisons are apples-to-apples
(not paired per-scene, but same population). One driver container per
variant with env vars baked in at `docker run`, everything else (renderer,
controller, runtime images) identical across runs. Full raw results in
`logs/results_log.csv`; full per-run outputs (including videos, metrics
plots, per-scene breakdowns) under `runs/<variant_name>/`.

## Stage 0: Correctness-risk empirical tests — both ruled out

**Hypothesis**: the vendored `transfuser_impl.py`'s training-time image
loader decodes via `cv2.imdecode(IMREAD_COLOR)` (BGR) into a dict key called
`"rgb"` with zero `cvtColor` calls anywhere in the file, while our driver
decodes true RGB via PIL — possible systematic channel swap. Also possible
vertical-crop mismatch (training crops 1920x1120→1920x1080 off the bottom;
we assume MTGS's 1080p render is already post-crop).

**Result**: Neither hypothesis held up. `r1_bgr` (channel-swapped) scored
0.5941 vs control's 0.6067 — a *small* difference well within the
run-to-run noise band (`score_stdev` ~0.22 on N=20), and collision/wrong_lane
rates were identical (0.35/0.05 in both). `r1_bottomcrop` similarly showed
no meaningful difference (0.6026, same collision/wrong_lane rates). **This
is a genuine negative result, not a failed test** — confirmed the toggles
actually reached the container (`docker exec ... env` check) and that the
underlying code correctly branches (unit-tested). Conclusion: our current
image pipeline (true RGB, no extra crop) is correct as-is; no evidence of
either bug.

## Stage 1: Heading stabilization — no measurable effect, even pushed hard

**Hypothesis**: cross-inference heading discontinuities (each 500ms
inference recomputes headings from scratch with no memory of the outgoing
plan's trend) contribute to the observed rear-end collisions (62.5% collision
rate, 0% at-fault, in the earlier 9-scene baseline — non-reactive background
traffic getting surprised by our motion).

**Implementation**: new `stabilize_headings()` in `trajectory.py` — (1)
cross-inference continuity blend (extrapolates the outgoing plan's tail yaw
trend into the first `blend_steps` of the new prediction), (2) yaw-rate
plausibility clamp using the free `angular_velocity.z` signal. **Verified
working via three separate methods**: standalone unit tests (proved the
function itself produces correct output on synthetic inputs), a wiring test
(proved `driver.py` actually invokes it and the result differs from raw when
enabled), and `docker exec ... env` (proved the env var reaches the
container).

**Result**: default params (`blend_steps=2, clamp_mult=3, floor=0.7`):
0.6067 vs baseline 0.6067 — *literally identical to 4 decimal places* at the
score level; a real but tiny effect is visible in `plan_deviation_mean`
(0.8086 → 0.8090). Pushed **aggressively** (`blend_steps=4, clamp_mult=1.5,
floor=0.3` — more than double the default damping strength): 0.6068, still
no meaningful change. Combined with Stage 2 (`r4_combined`): 0.6365 vs
Stage-2-alone's 0.6361 — no added benefit, no harm either.

**Why it didn't help (root-cause reassessment)**: `plan_deviation` measures
*position* L2 distance between consecutive plans at common future
timestamps, not heading divergence specifically. Heading-only smoothing
doesn't touch the raw model's predicted *xy* trajectory or its implied
speed/braking profile at all — if the actual discontinuity driving
collisions is a **position/speed-profile** inconsistency between inferences
(e.g. the model predicting a different implied deceleration curve every
500ms), heading stabilization is simply the wrong lever. This is consistent
with Stage 2 (which does touch xy/comfort) being the one that worked.

## Stage 2: Trajectory optimizer — the real winner

**Implementation**: ported `src/driver/src/alpasim_driver/trajectory_optimizer.py`
verbatim into `transfuser_challenge/trajectory_optimizer.py` (self-contained,
numpy+scipy only). Wired into `driver.py::_maybe_run_inference` after
heading stabilization, before caching the plan. Runs scipy L-BFGS-B (SLSQP
fallback) minimizing a smoothness + deviation-from-raw + comfort-limit-penalty
cost, with Frenet-style arclength retiming as an initial guess.

**Key tuning decision**: tightened `max_deviation` from upstream's own
default of 2.0m down to 0.5m, reasoning that `dist_to_gt_trajectory >= 4.0m`
triggers the same eval truncation as a collision, and letting the optimizer
redraw the path by meters per waypoint works against score. **This turned
out not to matter in practice** — 0.3/0.5/1.0/2.0m all produced statistically
indistinguishable scores (0.6274/0.6307/0.6309/0.6307) except 0.3m, which was
measurably worse. The optimizer's *soft* deviation-cost term is the actual
binding constraint, not the hard bound — so 0.5m gets full performance with
a materially smaller worst-case position deviation than upstream's 2.0m,
making it the safer choice for equal benefit.

**Weight tuning**: increasing `smoothness_weight` from 1.0 (upstream default)
to 3.0 gave a further, real, if modest, improvement: 0.6307 → 0.6361.

**What the improvement is NOT**: collision_any/at_fault/offroad rates were
completely unchanged (0.35/0.0/0.0) across every optimizer variant. The gain
is entirely in progress quality (`progress_rel_mean` 0.5849 → 0.6206,
`progress_clipped_rel_mean` 0.5059 → 0.5389), not in avoiding the truncation
events that (per the original diagnosis) dominate the score formula. This is
worth flagging honestly: the optimizer makes the driving that *does* survive
better/further, but doesn't reduce how often a (non-at-fault) collision
truncates the episode. That remains an open problem — see Next Steps.

**Latency**: optimizer adds ~125-190ms per inference (steady-state benchmark:
model ~65ms + optimizer ~125ms ≈ 190ms; observed in-sim
`driver_drive_rpc_duration_mean_s` ~0.23-0.29s across variants), comfortably
under the 500ms/2Hz nuPlan control-step budget in every tested configuration.

## Combined result and confirmation runs

`r4_combined_stage1_stage2` (Stage1 default + Stage2 tuned): 0.6365 — matches
Stage 2 alone within noise, confirming no negative interaction but also no
added value from Stage 1.

**Winning config**: `TRANSFUSER_TRAJ_OPTIMIZER_ENABLED=1`,
`TRANSFUSER_TRAJ_OPTIMIZER_MAX_DEVIATION_M=0.5`,
`TRANSFUSER_TRAJ_OPTIMIZER_SMOOTHNESS_WEIGHT=3.0`, all else default
(`TRANSFUSER_HEADING_STABILIZE_ENABLED=0`, Stage 0 toggles off).

**N=30 (`r5_winner_n30`)**: 0.5935 — *lower* than the N=20 number in absolute
terms, but this uses 10 additional scenes never tested before (scenes 21-30
of the sorted list), and there was no N=30 *control* run to compare against —
those 10 extra scenes could simply be harder (their own collision_any=0.40
vs the N=20 set's 0.35 is consistent with that). Superseded by the N=100
apples-to-apples comparison below; kept in the log for completeness but
should not be read as "the optimizer regressed."

**N=100 definitive comparison** (`r6_control_n100` vs `r6_winner_n100`, all
100 local scenes):

| Metric | Control | Winner | Δ |
|---|---|---|---|
| score_mean | 0.5764 | 0.6399 | **+0.0635 (+11.0%)** |
| score_stdev | 0.2161 | 0.2151 | slightly tighter spread |
| collision_at_fault_rate | 0.01 | 0.01 | unchanged (same 1 scene fails identically in both) |
| collision_any_rate | 0.28 | 0.28 | **exactly unchanged** |
| offroad_rate | 0.0 | 0.0 | unchanged |
| wrong_lane_rate | 0.09 | 0.11 | small increase; diagnostic-only, doesn't affect score |
| progress_rel_mean | 0.5090 | 0.5632 | +0.0542 |
| progress_clipped_rel_mean | 0.4768 | 0.5355 | +0.0587 |
| driver_drive_rpc_duration_mean_s | 0.1018 | 0.2909 | +0.189s (still 58% of the 500ms budget, safe) |

*(Numbers above corrected from an earlier version of this report — see
"Round 7-8" below for a bug I found and fixed in the analysis tool
mid-session. The conclusion is unchanged: this bug affected both runs
identically, since the same one scene hard-fails the same way in both.)*

This is the headline result for Candidate A (trajectory optimizer) alone:
**+11.0% relative score improvement, unchanged collision/offroad rate,
latency still well within budget**, on the full available local scene set —
the strongest evidence for this specific candidate. See "Round 7-8" below
for two further candidates tested afterward, one of which scores
substantially higher but comes with an important safety caveat.

## Ranked list of everything tested (best to worst, by score delta)

1. **Trajectory optimizer, smoothness_weight=3.0, max_deviation=0.5m** —
   **+11.0%** at N=100 (definitive), +4.8% at N=20 (exploratory). **Winner —
   recommended for production.**
2. Trajectory optimizer, default weights (smoothness_weight=1.0),
   max_deviation ∈ {0.5, 1.0, 2.0}m — +4.0% at N=20, all three deviation
   settings statistically tied. max_deviation=0.5m preferred for equal
   benefit at lower worst-case risk.
3. Trajectory optimizer, max_deviation=0.3m — +3.4% at N=20, the one setting
   in the sweep that was measurably worse than the rest (too tight a bound
   constrains the optimizer before it can fully smooth).
4. Stage 1+2 combined (heading stabilize + optimizer) — statistically
   identical to Stage 2 alone; adds complexity/latency for no measured gain.
5. Heading stabilization alone, aggressive params — 0.0% (no effect even
   pushed hard).
6. Heading stabilization alone, default params — 0.0% (no effect).
7. Bottom-crop toggle — -0.7% (no real effect, within noise).
8. BGR channel-order flip — -2.1% (no real effect, within noise; if
   anything the numbers point slightly negative, reinforcing that our
   current RGB decoding is correct).

**Update after Round 7-8 (below): a new candidate (B+C combined) scores
substantially higher than #1 above (0.818 vs 0.640 at N=100) but comes with
a serious safety caveat (10x higher at-fault collision rate). Read Round 7-8
in full before deciding what to ship — the ranking above is score-only and
does not reflect that tradeoff.**

## Round 7-8: Candidates B (temporal consistency) and C (brake/jerk limiter)

Requested as a follow-up: three new candidates framed as alternatives/complements
to the trajectory optimizer (renamed "Candidate A" retroactively) —

- **Candidate B**: strong cross-inference *temporal consistency*. Unlike
  Stage 1 (heading-only), this blends both position *and* heading of the
  first `blend_steps` (default 4, i.e. 2 of the 4s horizon) future waypoints
  toward a straight-line extrapolation of the outgoing plan's own last
  velocity/yaw-rate trend. Directly targets the gap Stage 1 left open:
  `plan_deviation` is a *position* metric, and heading-only smoothing never
  touched xy.
- **Candidate C**: moderate brake/jerk limiter. A standalone, closed-form
  (non-optimization) re-timing pass — caps implied longitudinal deceleration
  (default 3.0 m/s²) and jerk (default 6.0 m/s³) of the raw trajectory's
  speed profile, then re-samples the *same path* at the resulting smoother
  arc-length schedule (extrapolating past the original path's end when
  capped deceleration needs more distance than the raw path covered — a real
  bug caught and fixed during unit testing before this ever reached a real
  scene).

Both implemented in `trajectory.py`
(`apply_temporal_consistency`/`limit_braking_and_jerk`), unit-tested
(including a deliberately-adversarial hard-stop scenario for C), and
wiring-tested against the real `TransfuserChallengeDriver` class (mocked
model, real driver code path) before any GPU time was spent. Full pipeline
order: raw prediction → B → Stage 1 (if enabled) → C → A (if enabled) →
cache plan.

### A bug found and fixed mid-round

While investigating an implausibly good `B+C` result, I found a real bug in
`compare_variants.py`: it filtered rollouts by `passed == True`, which
silently *excluded* genuine driving hard-failures (collision_at_fault /
offroad — these have `passed: false` but are fully-scored rollouts with
`score: 0.0` and real metrics, not infrastructure errors) from every average
instead of counting them. This inflated `score_mean` and hid the true
`collision_at_fault_rate` whenever a hard failure occurred. Fixed by
distinguishing genuine hard-failures (real `score_metrics.collision_at_fault`,
a float) from true infrastructure failures (missing MTGS asset etc. — same
`passed: false` shape but `score_metrics` fields are `null`), and a
`rebuild` command was added to recompute the entire log from the
already-saved `results-summary.json` files without re-running any
simulation. **Impact on earlier results**: every N=20 run in Rounds 1-4 was
unaffected (zero hard failures occurred in that scene subset under any of
those configs) — those conclusions stand unchanged. The N=100 headline
numbers shifted slightly (0.582/0.646 → 0.576/0.640) because one scene
(`...9326163d4c9c5d16`) hard-fails identically under both control and
Candidate A — the *relative* conclusion (+11.0%) is unchanged since it
affected both runs equally.

### Results (N=20 unless noted)

| Variant | score_mean | collision_at_fault | collision_any | wrong_lane | progress_rel |
|---|---|---|---|---|---|
| Baseline (nothing) | 0.607 | 0.00 | 0.35 | 0.05 | 0.585 |
| Candidate A alone | 0.636 | 0.00 | 0.35 | 0.05 | 0.621 |
| **B alone** | 0.541 | 0.00 | **0.25** | **0.00** | 0.483 |
| **C alone** | 0.607 | 0.00 | 0.35 | 0.05 | 0.585 |
| A + B | 0.561 | 0.00 | 0.35 | 0.05 | 0.543 |
| A + C | 0.636 | 0.00 | 0.35 | 0.05 | 0.621 |
| **B + C** | **0.830** | 0.00 | **0.20** | 0.05 | 0.854 |
| A + B + C | 0.830 | 0.00 | 0.20 | 0.05 | 0.741 |
| **B + C, N=100 (confirmation)** | **0.818** | **0.10** | **0.15** | 0.17 | 0.839 |

**B alone**: a genuine, interesting finding on its own — collision_any drops
from 0.35 to 0.25 and wrong_lane drops to 0.00, but overall score *drops*
(0.541) because progress falls even more (the strong position blend makes
the car overly conservative, dragging down progress more than the
collision reduction helps under this scoring formula).

**C alone**: exactly identical to baseline to 4 decimal places — a clean
no-op. The raw model's predictions apparently never exceed a 3.0 m/s²
implied deceleration on this dataset/checkpoint, so the limiter never
triggers. Same dead-end pattern as Stage 1's default heading params.

**A + B**: a real *negative* interaction — combining loses B's collision
reduction entirely (collision_any back to 0.35) without recovering A's full
progress benefit. Plausible mechanism: A's optimizer reshapes B's already-
blended trajectory using a cost function with no knowledge of *why* B
blended it that way, eroding the conservative safety margin B introduced.

**A + C**: identical to A alone (both to 4 decimals) — fully consistent with
C being a no-op on this model's typical raw output, on top of A too.

**B + C — the standout, with a critical caveat.** At N=20, B+C scores 0.830
— far above anything else tested (A alone: 0.636). Two independent N=20
runs (`B_plus_C` and `A_plus_B_plus_C`, which is statistically identical to
`B_plus_C` — A adds nothing once B+C is present) agree closely, and this
survived the compare_variants.py bug fix largely intact. **Confirmed at
N=100**: 0.818, still dramatically higher than Candidate A's 0.640.

But the *component* breakdown at N=100 tells a more careful story:
`collision_at_fault_rate` is **0.10 — ten failures, a 10x increase** over
Candidate A's 0.01. Investigating the 10 at-fault scenes: **all 10 fail very
early** (`duration_frac_20s` between 0.05 and 0.20 — within the first 1-4
seconds of a 20s scene), and are predominantly front-collisions (8/10). The
other 90 scenes are excellent — 61 of them score ~1.0 (essentially perfect
driving). This is a coherent, specific failure mode, not noise: B's
extrapolation-based blending has no reliable trend to extrapolate from in
the first couple of inferences after session start; over-committing to that
early, possibly-unrepresentative trend appears to occasionally drive the
car into something it should have reacted to before the trajectory history
"warms up." Once past that window, the combination of strong temporal
consistency and brake/jerk smoothing produces unusually clean, log-consistent
driving.

**Why the score is so much higher despite 10x more at-fault collisions**:
the scoring formula only zeroes a scene on hard failure — it doesn't
distinguish "one bad scene at 10% at-fault rate" from "one bad scene at 1%
at-fault rate" beyond that binary. B+C's 90 clean scenes average close to
1.0 (near-perfect progress, very few non-at-fault incidents), which more
than outweighs the 10 zeroed scenes in the arithmetic mean. **This is a real
tension between "what the score formula rewards" and "what an honest safety
assessment would prioritize"**: a 10x increase in the metric that most
directly represents "the car caused a real collision" is a serious finding
that a raw score comparison alone would hide.

**Recommendation**: I have *not* changed the production default to B+C.
Candidate A (trajectory optimizer alone, `max_deviation=0.5m,
smoothness_weight=3.0`) remains the shipped default — it has zero measured
downside and a real, if smaller, upside. B+C is a genuinely exciting lead
(the underlying mechanism — cross-inference position continuity + brake/jerk
smoothing — clearly does something right on 90% of scenes) but needs the
early-episode failure mode addressed before it belongs in production: the
most promising next step (not yet implemented) is disabling or ramping
Candidate B's blend strength for the first few inferences of a session
(e.g. no blending until at least 2 prior plans exist, or a warm-up period
with reduced `blend_steps`), so it can't over-commit before it has reliable
history to extrapolate from.

## Next steps (not pursued this round — flagged for future work)

- The optimizer improved progress but didn't reduce the underlying
  collision/truncation rate. Since collisions are consistently non-at-fault
  rear-ends against non-reactive background traffic, the real lever for
  *that* problem likely isn't post-processing the driver's own trajectory at
  all, but something upstream (model retraining/fine-tuning — explicitly out
  of scope this round) or a fundamentally different intervention (e.g. a
  conservative speed floor to avoid ever decelerating faster than the
  original recorded ego, so trailing non-reactive traffic's expectations
  aren't violated) that wasn't tested here.
- Route-heading divergence logging (diagnostic-only, not wired into output)
  was implemented in the plan but not instrumented/analyzed this round —
  would need explicit logging output parsing to correlate with wrong_lane.
- Curvature-based speed limiting from route geometry, obstacle-avoidance
  overrides — still deferred per the original plan (infeasible without new
  perception input, or uncertain payoff).
