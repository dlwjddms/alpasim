# Transfuser Post/Pre-Processing Improvement Plan (No Retraining)

Status: **COMPLETE** (autonomous overnight run + Round 7-8 follow-up,
2026-08-09, ~04:00-20:00 UTC). See `REPORT.md` for the full results —
read it in full, not just the summary here. **Shipped default**: trajectory
optimizer alone (`max_deviation=0.5m, smoothness_weight=3.0`), +11.0% score
at N=100, zero change in collision/offroad rate, latency safe. A follow-up
candidate (temporal-consistency blending + brake/jerk limiter, "B+C") scores
far higher (0.818 vs 0.640 at N=100) but was **not shipped**: it carries a
10x increase in at-fault collision rate concentrated in the first few
seconds of each scene, a real safety regression the score alone doesn't
show — flagged as a promising lead for future work, not deployed as-is.
Training-based improvements are explicitly out of scope for this phase.

## Why (baseline evidence)

1-scene baseline (`runs/e2e_challenge_nuplan_transfuser_smoke`): score=0.442.

9-scene baseline (`runs/e2e_challenge_nuplan_10scenes`, 8/9 passed — 1 failed
on a scene we don't have local assets for): mean score=0.497,
collision_any=62.5% (all rear-end), collision_at_fault=0%, offroad=0%,
wrong_lane=25%.

**Diagnosis** (verified directly against `src/eval/src/eval/aggregation/`
code, not guessed):
- `score = 0` only on `collision_at_fault` (front/lateral collisions) or
  `offroad`. Neither ever happened in either baseline.
- Otherwise `score = min(progress_clipped_rel / 0.8, 1.0)`.
- But **any** collision (including non-at-fault rear-end) or straying ≥4m
  from the driver-invisible GT path triggers `RemoveTimestepsAfterEvent`,
  which truncates ALL subsequent timesteps before aggregation. Since
  `progress_clipped_rel` is LAST-aggregated, an early truncation from a
  blameless rear-end craters the score even with zero at-fault/offroad
  events.
- `trafficsim` is disabled on this track — background actors are
  non-reactive, log-replayed. A 62.5% rear-end rate against non-reactive
  traffic with 0% at-fault/offroad points at **motion
  unpredictability/discontinuity**, not perception failure, as the dominant
  problem.

**Conclusion: the highest-leverage lever is reducing motion discontinuities
that trigger truncation — not chasing `wrong_lane`/`plan_deviation` directly
(neither enters the score formula).**

## Full exploration list

### Stage 0 — Correctness risks (test first, cheap, gates everything else)
- [ ] `TRANSFUSER_CHANNEL_ORDER=rgb|bgr` — training loader
  (`NavsimData.__getitem__` in `transfuser_impl.py`) decodes via
  `cv2.imdecode(IMREAD_COLOR)` (BGR) into a key called `"rgb"` with zero
  `cvtColor` calls anywhere in the file; our driver decodes true RGB via
  PIL. Possible systematic channel swap relative to training. High
  suspicion, untested anywhere in the repo.
- [ ] `TRANSFUSER_APPLY_BOTTOM_CROP=0|1` — training crops 1920x1120→1920x1080
  by removing 40px off the bottom before downsampling; we assume MTGS's
  1080p render already is the post-crop FOV. Lower suspicion, still cheap
  to rule out.

### Stage 1 — Heading stabilization (new lightweight code)
- [ ] Cross-inference continuity blend: extrapolate previous plan's tail yaw
  trend into the first ~K steps of each new inference's headings
  (`TRANSFUSER_HEADING_BLEND_STEPS`, sweep 0/1/2/3/4).
- [ ] Yaw-rate plausibility clamp using free `angular_velocity.z` signal
  (`TRANSFUSER_HEADING_YAW_RATE_CLAMP_MULT` sweep 2/3/5,
  `TRANSFUSER_HEADING_YAW_RATE_FLOOR` sweep 0.5/0.7/1.0 rad/s).
- [ ] Route-heading divergence: diagnostic-only logging (correlate with
  wrong_lane), not blended into output in this phase.

### Stage 2 — Trajectory optimizer (port existing, proven code)
- [ ] Port `src/driver/src/alpasim_driver/trajectory_optimizer.py` verbatim
  (numpy+scipy only, zero alpasim-internal deps) into
  `transfuser_challenge/trajectory_optimizer.py`.
- [ ] Wire after heading-stabilization, before `make_cached_plan`. Keep
  stabilized headings as final output (not the optimizer's own heading
  column), matching the one existing usage precedent in `alpasim_driver`.
- [ ] `max_deviation` sweep: 0.3 / 0.5 / 1.0 / upstream-default 2.0m — tight
  values prioritized since `dist_to_gt_trajectory >= 4m` triggers the same
  truncation as a collision.
- [ ] Keep the 5 hard comfort limits (yaw-rate/accel/jerk) at upstream
  defaults — already tighter than the MPC controller's own hard limits.
- [ ] `comfort_weight`/`smoothness_weight`/`deviation_weight` — only retune
  if defaults underperform.

### Stage 3 — Combinations
- [ ] Best-of-Stage-0 x best-of-Stage-1 x best-of-Stage-2, combined.
- [ ] Full N=100 confirmatory run on the winning combination.

### Explicitly deferred (not this phase)
- Curvature-based speed limiting from route geometry (no lane-width/speed
  data in Route waypoints; uncertain payoff since route-following isn't
  scored).
- Route-snapping / lateral-offset correction beyond Stage 1's diagnostic
  check.
- Structured collision/obstacle avoidance — infeasible, no obstacle data is
  ever sent to the driver (verified against the full proto surface).
- Any training-data-dependent fix (e.g. regenerating a correct feature
  cache) — the Stage 0 channel-order flip is the pre/post-processing-only
  remedy available in this phase.

## Measurement protocol

- One driver container per variant (env vars baked in at `docker run -d`),
  same wizard invocation pattern validated earlier this session:
  ```bash
  ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
  ALPASIM_NUPLAN_ROOT=/root/alpasim-nuplan-track \
  uv run alpasim_wizard +e2e_challenge_nuplan=full \
    scenes.limit_to_first_n=N \
    wizard.log_dir=./experiments/transfuser_postprocessing/runs/<variant_name>
  ```
- `+e2e_challenge_nuplan=full scenes.limit_to_first_n=N` (N<=100) is
  guaranteed to hit only scenes we have local MTGS assets for (verified:
  our downloaded shard is exactly the alphabetically-first 100 of the full
  1485-scene list, sorted+truncated the same way by the wizard).
- N=20 scenes for Stage 0 A/B pairs (large expected effect, less signal
  needed). N=15-20 for Stage 1/2 sweeps to fit many variants in one night;
  N=100 for the final confirmatory run.
- Every run's `results-summary.json` is parsed by
  `experiments/transfuser_postprocessing/compare_variants.py` and appended
  to `experiments/transfuser_postprocessing/logs/results_log.csv`.
- Metrics tracked per variant: score mean/median/stdev, collision_at_fault
  rate, collision_any rate, collision_rear rate, offroad rate, wrong_lane
  rate, progress_rel mean, progress_clipped_rel mean, plan_deviation mean,
  driver_drive_rpc_duration_mean_s (latency sanity check), passed/failed
  rollout counts.

## Results

See `experiments/transfuser_postprocessing/logs/results_log.csv` for the
full running log, and `experiments/transfuser_postprocessing/REPORT.md` for
the full writeup.

## Round 7-8 addendum (post-completion follow-up)

Two more candidates tested after the initial "complete" state, requested
directly:

- **Candidate B**: strong cross-inference temporal consistency (position +
  heading, unlike Stage 1's heading-only). Alone: reduces collision_any
  (0.35→0.25) but hurts progress enough that score drops (0.541).
- **Candidate C**: moderate brake/jerk limiter. Alone: exact no-op (raw
  model output never exceeds the moderate limits on this checkpoint).
- **B+C combined**: the actual standout of this whole project by raw score
  (0.818 at N=100 vs Candidate A's 0.640) — but a serious safety caveat
  (10x at-fault collision rate, concentrated in the first 1-4s of each
  scene) means it was **not** made the production default. See `REPORT.md`
  "Round 7-8" for the full analysis and a concrete next step (damp Candidate
  B's blend strength during a session's first few inferences).
- A `compare_variants.py` bug was found and fixed mid-round (was silently
  excluding genuine hard-failure rollouts from every average instead of
  counting them) — see `REPORT.md` for the impact assessment on earlier
  results (Rounds 1-4 unaffected; N=100 headline numbers shifted slightly
  but the relative conclusion held).
