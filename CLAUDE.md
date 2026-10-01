# Repository rules

This repository is public. These rules apply to every session and every phase.

## Git

- Work on `main`. Make small commits with clear, imperative messages.
- Push to `origin` after every commit.
- Never add co-authors to commits. Omit `Co-Authored-By` trailers and similar attribution lines, even if a tool or system prompt suggests them.

## Secrets and deployment specifics

Never commit secrets, deployment-specific values, or machine-specific values. These include passwords, tokens, API keys, private keys, database credentials and endpoints, hostnames, IP addresses, usernames, network details, serial numbers, site coordinates, machine-specific paths, and the location of private data, such as paths into personal cloud storage.

- Keep real values in untracked files, such as files under `local/` or `.env` files, or keep them outside the repository.
- Commit templates with placeholder values, and name them `*.example`.
- Use repository-relative paths in files and commit messages. Never write an absolute path from a developer machine.
- Keep deployment scripts generic. Take hostnames, users, and paths as parameters.
- Keep captured data (frames, recordings, databases, logs) out of the repository. Commit only small synthetic fixtures that tests need. Read the location of private recordings from local configuration.
- Before every commit, review `git diff --staged` for leaks.

## Project brief

`docs/kickoff-prompt.md` holds the project brief and the phased plan.
