# Review: GitHub Lens MCP — how I used “search tools” and how to improve it

## Context / Goal

You asked me to “look at PR files” for:

- https://github.com/1inch/pathfinder/pull/3475/files

Your goal now is to improve the GitHub Lens MCP capability around “search tools” (i.e., discovering which GitHub actions are available in the current session and how to use them effectively).

This document explains:

1. **How I attempted to access PR file diffs.**
2. **How I used the tool-discovery (search) capability.**
3. **What went wrong / limitations observed.**
4. **Concrete improvement suggestions** for the GitHub Lens MCP.

---

## What I tried first (human-style web access)

### Attempt A — open the PR “Files changed” page

I tried to fetch the PR’s “files changed” page directly. The response was **404 / Not Found**.

**Why this happens:**
- For **private repos**, GitHub intentionally returns 404 when you’re not authenticated or authorized.
- For some resources, even if the PR is public, GitHub may still require certain headers/cookies or may block automated fetches.
- The `/pull/<id>/files` page is HTML and may rely on JS/hydration; a raw fetch may not reveal the patch content you actually want.

### Attempt B — use the `*.diff` endpoint

I also tried the conventional endpoint:

- `/pull/<id>.diff`

This also returned **404**.

**Interpretation:**
- Almost certainly access/auth issue (private repo or permissions), not merely parsing difficulty.

**Result:** I could not see the PR diff contents via unauthenticated web fetch.

---

## How I used the “search tools” capability

Once I couldn’t access PR files by fetching HTML/diff, I pivoted to GitHub Lens MCP’s tool-discovery mechanism (“search tools”).

### Search 1 — discover PR-related tools

I queried for PR-oriented capabilities with keywords akin to:

- “pull request files”
- “list changed files”
- “diff”
- “review”

**Outcome:** the tool-discovery returned only a small set of PR actions (e.g., listing PRs, creating/merging PRs, updating PR branch). Importantly, there was **no tool that directly retrieves PR file diffs / changed files**.

### Search 2 — discover review/diff tools

I searched under a “reviews” category (or equivalent) for something like “get diff”, “files changed”, “review”.

**Outcome:** no tools were returned.

---

## Observed gaps / pain points

### 1) Tool discovery is not “task-aligned”
From the user’s perspective, “review a PR” implies:

- Get PR metadata (title, description, base/head)
- Get changed files list
- Get patch/diff per file
- Optionally: inline comments / create review

But the discovered tools were centered around management actions (create/merge/update) rather than retrieval for analysis.

### 2) No “diff retrieval” capability exposed
Without a “get diff / list files changed” capability, the agent cannot do code review unless the repo is mirrored locally or the user provides the diff manually.

### 3) Authentication/authorization is opaque
When web fetch returns 404, there’s no clear guidance from the tool layer explaining:
- whether the resource is private,
- whether GitHub auth exists in the current environment,
- whether the MCP can access it via API anyway.

### 4) Poor fallback guidance
When tools can’t retrieve diffs, the MCP could proactively guide the user to provide:
- a patch file,
- a `gh pr diff` output,
- or permissions setup steps.

Right now this guidance is possible, but not “built-in” or structured.

---

## Recommendations to improve GitHub Lens MCP

### A) Add retrieval primitives for PR review (highest impact)

1. **Get pull request details**
   - Input: repo, PR number
   - Output: title, body, base/head refs, author, state, mergeable status, labels

2. **List changed files**
   - Input: repo, PR number
   - Output: list of file paths, status (added/modified/deleted), additions/deletions, blob URLs

3. **Get unified diff / patch**
   - Input: repo, PR number, optional file path
   - Output: unified diff text (or per-file patch)

4. **Get file contents at base/head**
   - Input: repo, ref (sha/branch), path
   - Output: raw text
   - Useful when diff is truncated or for context.

These four capabilities allow an agent to do real code review without needing UI scraping.

**Design note:** if you already have API coverage internally, expose it as stable “read” tools. These are safer than merge/update actions.

---

### B) Improve the tool discovery UX (search relevance)

1. **Synonym expansion / intent mapping**
   When the query contains “review PR”, “files changed”, “diff”, “patch”, “compare”, expand to:
   - compare endpoints
   - pull request files endpoints
   - commits endpoints
   - review endpoints

2. **Return “capability hints”**
   The search response should include:
   - what the tool returns (e.g., “returns unified diff”)
   - required parameters
   - scope requirements (read-only vs write)
   - auth requirements

3. **Rank read-only tools higher for “review” intent**
   For review workflows, you almost always want retrieval before mutation.

---

### C) Better auth diagnostics and guidance

When a GitHub web fetch or API call fails with 404, the MCP should:

1. Detect likely causes:
   - private repo
   - missing auth
   - insufficient scopes

2. Provide actionable next steps:
   - “authenticate GitHub”
   - “grant repo scope”
   - “confirm repo visibility”
   - “use CLI to export diff”

This can be done without exposing sensitive details; it’s about clarifying the path forward.

---

### D) Add a structured “review workflow” helper

Create a higher-level tool, for example:

- “Prepare PR review bundle”

It could return a single JSON payload:

- PR metadata
- changed files list
- diffs (with truncation strategy)
- commit list
- optionally: checks status

This reduces the need for multiple calls and makes the agent’s review deterministic.

**Truncation strategy:**
- If diff is too large, return:
  - file list with stats
  - top N largest diffs
  - and a continuation token for paging

---

### E) Safety and ergonomics improvements

1. **Read-only default**
   Ensure “merge/update” actions are clearly separated and not suggested unless explicitly asked.

2. **Pagination everywhere**
   Changed files and diff retrieval must page reliably.

3. **Deterministic formatting**
   Ensure diffs preserve line numbers and context to support precise feedback.

---

## What a “good” experience would look like

If you ask: “Посмотри PR …/pull/3475/files”, the ideal path is:

1. MCP tries: `get_pull_request` → success
2. MCP tries: `list_pull_request_files` → success
3. MCP tries: `get_pull_request_diff` → success
4. Agent produces review:
   - summary
   - file-by-file notes
   - suggested patches

If auth is missing:

- MCP responds with a clear explanation + a ready-to-run command:
  - `gh pr diff ... > pr.diff`
  - or a link to authorize

---

## Appendix: Summary of the observed behavior

- Direct web fetch of GitHub PR pages/diff returned **404**, likely due to access/auth.
- Tool discovery (“search tools”) did not surface any tool to fetch:
  - PR diffs
  - changed files
  - review artifacts
- Result: I could not actually review code changes without external diff input.

---

## Next steps (how we can improve it together)

1. Decide the minimal read-only tool set:
   - PR details
   - file list
   - diff/patch retrieval
2. Implement intent-aware tool discovery:
   - map “review” → retrieval tools
3. Add auth troubleshooting guidance:
   - explain 404 patterns
4. Add a “review bundle” helper:
   - one call → everything needed for review

If ты покажешь, как сейчас устроен твой GitHub Lens MCP (список доступных действий и их схемы/параметры), я могу предложить конкретный дизайн интерфейса: названия, параметры, форматы ответов и стратегию пагинации/транкации.