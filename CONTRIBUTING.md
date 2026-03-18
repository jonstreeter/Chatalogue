# Contributing to Chatalogue

Thanks for contributing.

## Development Setup

### Windows
1. Run `install_windows.bat`
2. Run `run_windows.bat`

### macOS
1. Run `chmod +x install_mac.sh run_mac.sh`
2. Run `./install_mac.sh`
3. Run `./run_mac.sh`

## Project Structure
- `backend/`: API, queue workers, transcription/diarization pipeline
- `frontend/`: React UI
- Root scripts: install/start helpers

## Contribution Flow
1. Create a branch from `main`
2. Keep changes scoped and focused
3. Run relevant checks/tests before opening PR
4. Open PR with:
   - what changed
   - why it changed
   - test evidence (commands/results or screenshots)

## Testing
- Frontend E2E tests:
  - `cd frontend`
  - `npm run test:e2e`

If you change backend APIs, include manual or automated verification notes.

## Pre-commit (required)
Install local hooks so secret scanning runs before every commit.

1. Install pre-commit:
   - `pip install pre-commit`
2. Enable hooks in this repo:
   - `pre-commit install`
3. Run once across current files:
   - `pre-commit run --all-files`

`gitleaks` is configured in `.pre-commit-config.yaml` and will block commits that contain secrets.

## Coding Guidelines
- Prefer small, reviewable PRs
- Preserve existing architecture unless refactor is intentional
- Do not commit secrets or local env files (`backend/.env`, API keys, cookies)
- Keep docs updated when behavior changes

## Issue Reporting
Please include:
- OS and GPU
- install/run method used
- exact error text/log excerpt
- reproduction steps
