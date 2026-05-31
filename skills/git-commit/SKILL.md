---
name: git-commit
description: >-
  Creates git commits following Conventional Commits format with type/scope/subject
  and detailed markdown body. Use when user wants to commit changes, create commit,
  save work, or stage and commit. Enforces project-specific conventions from CLAUDE.md.
  Each change type gets its own markdown heading (# emoji + type), with detailed item lists under each.
---

# Git commit

Creates git commits following Conventional Commits format with rich markdown body.

## Recent project commits

!`git log --oneline -5 2>/dev/null`

## Quick start

```bash
# 1. Stage changes
git add <files>

# 2. Run focused validation before committing
make format
make lint
pytest tests/path/to/relevant_test.py

# 3. Re-stage any files changed by formatters
git add <files>

# 4. Create commit with detailed markdown body
git commit -F /tmp/commitmsg.txt
```

## Commit message structure

### 1. Subject line (first line)

```
type(scope): concise imperative description
```

### 2. Blank line

A mandatory blank line separating the subject from the body.

### 3. Markdown body (required for all non-trivial commits)

Use first-level headings (`#`) for each change type, second-level headings (`##`) for specific changes, bullet points for details, and `---` separator between multiple types.

**Type-to-emoji mapping:**

| Type     | Emoji | Heading Format              |
| -------- | ----- | --------------------------- |
| feat     | ⭐    | `# ⭐ Feature`              |
| fix      | 🐛    | `# 🐛 Bug Fix`              |
| refactor | ♻️    | `# ♻️ Refactor`             |
| perf     | ⚡    | `# ⚡ Performance`          |
| test     | ✅    | `# ✅ Tests`                |
| docs     | 📝    | `# 📝 Documentation`        |
| ci       | 🔧    | `# 🔧 CI/CD`                |
| chore    | 🔩    | `# 🔩 Chore`                |
| style    | 🎨    | `# 🎨 Style`                |
| security | 🔒    | `# 🔒 Security`             |

**Multi-type commits**: Title uses the primary type; body uses a separate `#` heading with emoji for each type, separated by `---`.

**Multi-type body example:**

```markdown
# 🐛 Bug Fix

## Fix token expiration check

- Fix token expiration check that always returned false

---

# ♻️ Refactor

## Simplify validation logic

- Refactor auth middleware to use early return pattern

---

# ✅ Tests

## Add auth unit tests

- Add unit tests for edge cases
- Test token refresh flow
```

**Single-type body example:**

```markdown
# ⭐ Feature

## Add user endpoints

- Implement GET /users/{id} endpoint
- Add POST /users for creating new users
- Include input validation middleware
```

## Full example

```bash
printf 'feat(skills): add code-review skill with checklists

# ⭐ Feature

## Add code-review skill with reference documentation

- SOLID principles checklist
- Python/ML best practices
- Security review guidelines
' > /tmp/commitmsg.txt
git commit -F /tmp/commitmsg.txt
rm /tmp/commitmsg.txt
```

> **Note:** HEREDOC with `git commit -m` can fail with emoji/unicode.
> Always prefer `printf ... > /tmp/commitmsg.txt && git commit -F /tmp/commitmsg.txt && rm /tmp/commitmsg.txt`.

## Important rules

- **ALWAYS** run focused validation before `git commit`; prefer `make format`, `make lint`, and relevant `pytest` targets
- **ALWAYS** include scope in parentheses (kebab-case)
- **ALWAYS** use present tense imperative verb for the subject
- **ALWAYS** include a markdown body with heading(s) for non-trivial commits
- **ALWAYS** prefer `git commit -F <tmpfile>` for commits with markdown body
- **NEVER** end subject with a period
- **NEVER** exceed 50 chars in the subject line
- **NEVER** use generic messages ("update code", "fix bug", "changes")
- **NEVER** push -- only create local commits. The user will push when ready.
