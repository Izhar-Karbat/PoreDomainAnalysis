# Contributing to PoreDomainAnalysis

Thank you for your interest in contributing to PoreDomainAnalysis! We use Cursor to assist both human and AI contributors in maintaining consistency. This document outlines our contribution workflow, commit conventions, and how to engage with Cursor rules.

## 1. Project Structure & .cursor Rules

All CPS-specific guidance lives under `.cursor/rules/`:

- `foundation.mdc` — Core coding and project conventions
- `domain-context.mdc` — Biological and MD analysis context
- `feature-requests.mdc` — Guidelines for requesting new features
- `bug-fix.mdc` — Guidelines for reporting and fixing bugs
- `project-specific.mdc` — (Add here any new rules specific to this repo)

Cursor will automatically apply rules from `foundation.mdc` and `domain-context.mdc` to all Python files (`globs: "**/*.py"`), but won’t force them unless `alwaysApply: true` is set. Project‑specific rules can be toggled per session.

For human developers: Please consult
- README.md — Project overview and usage
- Code_Structure.md — Detailed package layout and workflow

## 2. How to File a Feature Request

1. **Open an issue** on GitHub, describing:
   - **Motivation**: Why this feature is needed.
   - **Scope**: What code modules or functions it affects.
   - **Integration points**: Where to insert new calls or data flows.
2. In the issue, include the Cursor rule snippet under `feature-requests.mdc` format.
3. Assign the PR label `feature` and reference the issue number.

## 3. How to Report or Fix Bugs

1. **Open an issue** with:
   - **Steps to reproduce** (commands, sample data, expected vs. actual output).
   - **Log excerpts** or stack traces.
2. The bug‑fix PR should:
   - Reference the issue in the commit message (`Fix #123`).
   - Include a `bug-fix.mdc` Cursor rule outlining the cause and resolution.
   - Add a unit test under `tests/` that reproduces the bug and confirms the fix.

## 4. Commit Message Guidelines

- Start with a **type**: `feat:`, `fix:`, `docs:`, `test:`, `chore:`.
- Keep the subject under 50 characters.
- Use the imperative mood: `Add`, `Remove`, `Refactor`.
- For PRs related to Cursor rules, mention the rule file: e.g., `chore: update domain-context.mdc`.

Example:
```
feat: add filter_com_threshold option (#456)

- Updated filter_and_save_data to accept `threshold` param
- Added test under tests/test_com_filter_threshold.py
- Corresponding Cursor rule in .cursor/rules/feature-requests.mdc
```

## 5. Running Cursor

- **Global rules** apply automatically:
  ```bash
  cursor run <your-prompt>
  ```
- To enable project‑specific rules:
  ```bash
  cursor run --alwaysApply project-specific.mdc <your-prompt>
  ```
- To list available rules:
  ```bash
  ls .cursor/rules
  ```

## 6. Pull Request Checklist

- [ ] All new code is covered by unit tests.
- [ ] `pytest -q` passes locally.
- [ ] `cursor run` does not introduce lint or rule violations.
- [ ] Documentation (README, CONTRIBUTING.md) is updated if necessary.

---
Thanks for making PoreDomainAnalysis better!

