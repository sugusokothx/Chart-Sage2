# Local Backtest Runner

`runtime/local_backtest_runner.py` is the orchestration layer behind both the GUI and CLI.

## CLI example

```powershell
python .\runtime\run_backtest.py --start 2025-04-01 --end 2026-03-31 --api-file .\api.txt --config .\local_backtest_config.example.json --frequency monthly
```

To run on a provider kept outside this repository instead of the J-Quants
cache, add `--provider-module FILE.py:CLASS` and `--provider-arg KEY=VALUE`
(experimental, CLI only; see "Running the CLI on an external provider" in the
README).

## Credential file

The CLI expects a credential file for scripted runs. Supported formats:

```text
api_key=xxxx
```

## Repository scope

- Supports the formula-driven composite engine
- Does not support the removed precomputed private strategy path
- Uses the repository `12-1 momentum` implementation

For a smaller onboarding config, start with `configs/minimal_long_only.example.json`.
No JPX or J-Quants source data is bundled with this repository.
