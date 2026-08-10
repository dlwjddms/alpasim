# Transfuser Assets

Model weights are local build inputs and should not be committed.

Run:

```bash
bash e2e_challenge/sample_submission_transfuser/scripts/prepare_assets.sh /path/to/transfuser_weights
```

Expected source files:

```text
model_0060.pth
config.json
```

The script copies them into
`e2e_challenge/sample_submission_transfuser/assets/transfuser/` for the Docker
build. `config.json` must be present alongside the checkpoint -- `load_tf()`
reads it from the same directory.
