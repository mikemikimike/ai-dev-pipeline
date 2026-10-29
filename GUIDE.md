# GATE Pipeline: Step-by-Step Guide

This guide outlines exactly how the modern GATE pipeline processes an issue, from initialization to the final merged commit.

## 1. Initialization & Isolation
The pipeline begins when you execute:
`python run_generator.py --repo /path/to/repo --issue-id "fix-auth-bug" --description "Fix the auth bug"`

The very first action GATE takes is to create a secure **Git Worktree**. This guarantees that all AI agent experiments and file modifications happen in complete isolation and will never disrupt your active branch.

Simultaneously, the pipeline spins up background **MCP (Model Context Protocol) Servers**. These servers act as standard interfaces that give the agents the ability to read your filesystem, search files, and run terminal commands safely.

## 2. Context Gathering & Planning (Supervisor Agent)
Before writing any code, the **Supervisor Agent** wakes up to map out an implementation strategy. 
It starts with a cheap repository snapshot plus fast code search (`rg`) and AST search (`grep-ast`). If semantic retrieval is enabled, it can also use the `semantic_code_search` tool, which is backed by a local or hosted embedding index.

The Supervisor executes a dynamic ReAct (Reason + Act) loop, repeatedly calling tools like `read_file_content` to build context. Once satisfied, it decomposes the original issue into a strict JSON `SupervisorPlan` consisting of one or more atomic `TaskDefinitions`.

## 3. Test-Driven Execution Loop (Per Task)
For every single task in the Supervisor's plan, the pipeline runs a rigorous, multi-agent loop:

### A. Test Generation (Test Writer Agent)
If a task explicitly includes test targets in its allowlist, the **Test Writer Agent** is invoked first. Primed with a strict prompt to only write validation logic, it uses the Aider CLI underneath to generate failing unit tests or integration tests targeting the expected behavior. It uses the native testing framework of your project (e.g., `pytest`, `npm run test`).

### B. Implementation (Worker Agent)
The **Worker Agent** (also powered by Aider) is then unleashed. It receives the task instructions and sees the newly written failing tests in the git worktree. It implements the necessary source code changes to make the tests pass.

### C. Deterministic Verification (Verifier Engine)
The **Verifier Engine** takes over to check the change. Unlike LLMs, this engine is strictly deterministic. It runs applicable syntax checks over MCP and project checks in Docker, recording the commands that ran. Missing prerequisites, no applicable syntax checks, and empty pytest collection are recorded as skipped rather than passed and block the release.
If any of these commands fail, the exact terminal output is fed back to the Worker Agent, forcing it to self-correct and retry until the tests pass.

The verifier also blocks temporary scratch artifacts, checks task allowlists, and can run project-native checks inside Docker when local package metadata is available.

### D. Architectural Review (Gatekeeper Agent)
Once the tests pass, the **Gatekeeper Agent** steps in as the senior reviewer. The Gatekeeper does not just blindly look at a diff; it runs its own ReAct tool loop. It proactively searches the codebase to verify that the worker's changes didn't accidentally break a downstream dependency or violate system architecture rules.
If the Gatekeeper rejects the implementation, it returns a detailed critique, sending the loop back to the Worker (or back to the Supervisor if the plan itself was flawed).

## 4. Finalization
Once every atomic task has successfully passed the Test Writer, Worker, Verifier, and Gatekeeper, the pipeline performs a final, holistic check:
- The Verifier runs a global build/test check across the entire repository.
- The Gatekeeper performs a broad-context review of the total compiled diff against the original issue.

Upon final approval, the isolated worktree is committed, the MCP servers gracefully shut down, and the feature is ready to be merged into your main branch.

## Optional Semantic Search
Semantic retrieval is now optional and budget-aware:

- `GATE_RAG_PROVIDER=disabled` keeps the pipeline on the cheapest path and relies on `rg` plus `grep-ast`
- `GATE_RAG_PROVIDER=local` enables the default local `sentence-transformers` embedding model
- `GATE_RAG_PROVIDER=api` enables hosted embeddings through LiteLLM
- `GATE_GEMINI_SAFETY_MODE=block_none` enables the relaxed Gemini safety override when a workflow explicitly needs it

Recommended default: keep semantic retrieval local, and let GATE automatically disable it only when the task is trivially small and semantic lookup would be wasted overhead.
