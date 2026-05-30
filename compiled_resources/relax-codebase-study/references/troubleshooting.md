# Troubleshooting Notes

Use this file to record reproducible codebase-study or validation failures.

## Known Study Pitfalls

- `relax-rocm-megatron/` points back to the repository root. Recursive tools can
  loop through `compiled_resources/relax-codebase-study/relax-rocm-megatron`;
  exclude `compiled_resources/relax-codebase-study` when scanning the subject
  repository recursively.
- Treat files under the subject symlink as live checkout state. Record branch,
  commit, and relevant dirty-worktree context in notes that depend on exact
  code state.
