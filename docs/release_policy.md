# Release Policy

RegimeBench `v0.1.2` is a public reproducibility benchmark release.

The release includes:

- 18 generated `signal-*` task definitions.
- A public deterministic repair-style PnL verifier with a causality gate.
- A base parquet data bundle with train and hidden/OOS test rows.
- Local split materialization for the report split profile.
- A deterministic synthetic verifier smoke expectation.

The release does not include:

- A blind hosted leaderboard.
- Private worker infrastructure, provider credentials, logs, or internal
  orchestration.
- Reference solutions for the tasks.

Because hidden/OOS labels are included in the public data bundle, leaderboard
claims from this release should be treated as reproducibility or local-method
comparison claims, not as blind generalization claims. Future blind evaluations
should use a separate withheld data service or private test set.
