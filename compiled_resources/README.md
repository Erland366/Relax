<!-- LAST_INDEXED: 2026-05-29 -->

# Local Compiled Resources

This directory stores project-specific curated resources. Local entries take
precedence over global resources in `~/dotfiles/compiled_resources/`.

## Freshness Protocol

Each domain directory should include a `README.md` with a
`<!-- LAST_INDEXED: YYYY-MM-DD -->` marker. Entries in this index use
`<!-- INDEXED: YYYY-MM-DD -->` markers so agents can quickly tell whether the
local index has been updated after adding or changing resources.

Before relying on a domain README:

1. Check the `LAST_INDEXED` date in this file and the domain README.
2. Inspect the target domain if the index appears stale or incomplete.
3. Update this file and the domain README when adding, removing, or materially
   changing resource contents.

## Domains

### 1. relax-codebase-study **29 May 2026**
<!-- INDEXED: 2026-05-29 -->

Structured codebase-study workspace for this exact Relax ROCm Megatron checkout.
The study references the current repository through
`relax-codebase-study/relax-rocm-megatron -> ../..`; it is not a cloned copy.
