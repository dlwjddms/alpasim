# Transfuser (LTFv6/NAVSIM) Sample Submission -- nuPlan Track

A Transfuser-backed `egodriver.EgodriverService` example for the AlpaSim e2e
challenge, targeting the **nuPlan/MTGS track**. Structured like
[`sample_submission_vavam`](../sample_submission_vavam/README.md) but built
from scratch around
[Latent Transfuser v6 (LTFv6)](https://huggingface.co/ln2697/tfv6_navsim), the
model developed for [NAVSIM](https://github.com/autonomousvision/navsim).

## Why this differs from `driver=transfuser` in the main wizard

The in-repo `driver=transfuser` config
(`plugins/transfuser_driver/alpasim_transfuser/configs/driver/`) is wired for
the PAI/NuRec runtime: its own 4-camera rig, f-theta rectification, and
250ms control step. The nuPlan/MTGS challenge preset
(`+e2e_challenge_nuplan=*`) uses a different 8-camera rig (`CAM_F0/L0/L1/L2/
R0/R1/R2/B0`) at 500ms and expects a self-contained driver container that
negotiates cameras dynamically at `start_session` (`driver_source=
external_static`), the same contract `starter_kit` and `sample_submission_vavam`
use. There is no ready-made "Transfuser on nuPlan" path in the repo, so this
directory adapts the Transfuser plugin code into that contract:

- **Camera selection**: uses exactly `CAM_L0, CAM_F0, CAM_R0, CAM_B0`, in that
  order. This is the same 4-of-8 subset and ordering the checkpoint was
  trained on for `NAVSIM_4CAMERAS`
  (see `NUPLAN_CAMERA_CALIBRATION` / `camera_calibration()` in
  `transfuser_impl.py`).
- **No rectification**: MTGS renders nuPlan cameras with real nuPlan pinhole
  intrinsics (`opencv_pinhole_param`, see
  `plugins/mtgs/server/artifact_adapter.py`), which already match what LTFv6
  expects -- unlike the PAI track's f-theta cameras. Images are only resized
  and center-cropped to `270x480` per camera (done inside the model wrapper),
  same as upstream.
- **Standalone dependencies**: `transfuser_challenge/transfuser_impl.py` is a
  verbatim copy of the plugin file (it's already self-contained: cv2, numpy,
  torch, timm, beartype, omegaconf). `transfuser_challenge/model.py`
  reimplements the thin wrapper from `transfuser_model.py` without depending
  on the `alpasim_driver` package, which pulls in unrelated and partly gated
  model repos (VAM, Alpamayo) as hard dependencies -- not appropriate for a
  lean, self-contained challenge image.

## Assets

Model weights are local build inputs and should not be committed. Download
from HuggingFace, then stage them for the Docker build:

```bash
huggingface-cli download longpollehn/tfv6_navsim model_0060.pth config.json \
  --local-dir=/tmp/transfuser_weights
bash e2e_challenge/sample_submission_transfuser/scripts/prepare_assets.sh /tmp/transfuser_weights
```

The source directory must contain:

```text
model_0060.pth
config.json
```

The script copies them into
`e2e_challenge/sample_submission_transfuser/assets/transfuser/`. You can also
set `TRANSFUSER_ASSET_SRC=/path/to/transfuser_weights`.

## Build

```bash
bash e2e_challenge/sample_submission_transfuser/scripts/build_image.sh
```

## Local Smoke Test (nuPlan)

Follow the [starter kit's NuPlan data setup](../starter_kit/README.md#data-setup)
first (`ALPASIM_NUPLAN_ROOT`, `trajdata_cache/nuplan_test.tar.gz`,
`MTGS_asset/navtest/configs.tar.gz`, and at least `part001.tar.gz`).

Start the driver container:

```bash
e2e_challenge/sample_submission_transfuser/run_local_container.sh
```

Then run the nuPlan smoke test from another terminal, with the driver
container still running:

```bash
source setup_local_env.sh
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
ALPASIM_NUPLAN_ROOT=/path/to/alpasim-nuplan-track \
uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  wizard.log_dir=./runs/e2e_challenge_nuplan_transfuser_smoke
```

Result:

```text
./runs/e2e_challenge_nuplan_transfuser_smoke/aggregate/results-summary.json
```

## Notes

- Model inference runs inline in `drive()` at the model's native 2 Hz
  (`TransfuserModel.output_frequency_hz`), which matches the nuPlan track's
  fixed 500ms control step (`e2e_challenge_nuplan_common/base.yaml`). Between
  inferences, the cached plan is interpolated and sampled at
  `TRANSFUSER_CALLBACK_FREQUENCY_HZ` (default 10 Hz) to build each `Drive`
  response, same pattern as `sample_submission_vavam`.
- `TRANSFUSER_CHECKPOINT_PATH` defaults to
  `/app/assets/transfuser/model_0060.pth`; `config.json` must sit alongside
  it (`load_tf()` reads it from the same directory).
- Route-derived driving command (`LEFT`/`STRAIGHT`/`RIGHT`) is computed from
  the first route waypoint past `TRANSFUSER_ROUTE_MIN_LOOKAHEAD_M` (default
  20m), classified by lateral offset against
  `TRANSFUSER_ROUTE_LATERAL_THRESHOLD_M` (default 3m).

## Post-processing (enabled by default)

The raw model trajectory is smoothed by a ported `TrajectoryOptimizer`
(`TRANSFUSER_TRAJ_OPTIMIZER_ENABLED=1` by default,
`TRANSFUSER_TRAJ_OPTIMIZER_MAX_DEVIATION_M=0.5`,
`TRANSFUSER_TRAJ_OPTIMIZER_SMOOTHNESS_WEIGHT=3.0`) — measured **+11.0%**
score improvement on the full local 99-scene set (0.582 → 0.646), with zero
change in collision rate and latency still well under the 500ms budget
(~0.29s). An optional heading-stabilization pre-step
(`TRANSFUSER_HEADING_STABILIZE_ENABLED`, default off) and two image-pipeline
correctness toggles (`TRANSFUSER_CHANNEL_ORDER`,
`TRANSFUSER_APPLY_BOTTOM_CROP`) were also tested and found to have no
measurable effect — left available as env vars for anyone who wants to
re-verify, but off by default. Full methodology, the complete test matrix,
and raw results: `experiments/transfuser_postprocessing/REPORT.md`.
