# Project Guidelines

## Language & Framework

- Rust + Axum web framework
- Follow idiomatic Rust conventions (clippy-clean, no unnecessary allocations)
- Use `tracing` for all logging, not `println`/`eprintln` (except startup messages before tracing is initialized)

## Code Style

- No unnecessary comments — well-named identifiers explain WHAT, comments only for non-obvious WHY
- Prefer `Result<T>` over panics in library code
- Keep functions focused; avoid large monolithic handlers

## Testing

- Run `cargo test` before committing
- Run `cargo clippy` and fix all warnings before committing
- Unit tests in same file under `#[cfg(test)] mod tests`

## Git Commit Rules

- No "Co-Authored-By" lines in commit messages
- No AI-assisted attribution in commit messages
- Commit messages should describe the change purpose, not just what changed