# Repository Layout Notes

This repository was reorganized toward a more standard deep learning layout
without modifying Python source code.

Non-breaking constraints:

- `Data/151674` is kept in place because example scripts and README commands
  refer to that path directly.
- `examples/` still contains runnable example scripts because those scripts use
  relative paths to `../Data/151674`.
- Only non-code assets were moved into dedicated folders:
  - notebook -> `notebooks/`
  - example figures -> `assets/figures/`

Suggested next step if code changes become allowed:

- move dataset paths out of source-controlled defaults and into CLI/config args
- migrate `examples/config.yaml` into a top-level `configs/` directory
- add smoke tests under `tests/`
