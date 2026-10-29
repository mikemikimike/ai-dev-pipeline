import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import structlog
import yaml

# Redirect structlog console output to pipeline.log to clean up stdout
try:
    log_file = open("pipeline.log", "a", encoding="utf-8")
    structlog.configure(
        logger_factory=structlog.WriteLoggerFactory(file=log_file)
    )
except Exception:
    pass

from agents.aider_worker import AiderWorkerAgent
from agents.gatekeeper import GateResult, GatekeeperAgent
from agents.meta_analyzer import MetaAnalyzerAgent
from agents.models import GateConfig, VerificationResult
from agents.researcher import ResearcherAgent
from agents.supervisor import SupervisorAgent, SupervisorPlan, TaskDefinition
from agents.worker import WorkerResult
from agents.test_writer import TestWriterAgent
from environment.mcp_client import PipelineMCPClient
from ledger.database import get_db
from orchestrator.self_improvement import (
    CircuitBreaker,
    FailureAnalysis,
    FailureAnalyzer,
    ModelRouter,
    PlanRepairAgent,
    PromptRewriter,
    Rule,
    RuleMiner,
    RuleStore,
    prompt_hash,
)
from orchestrator.verifier import VerifierEngine

C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

logger = structlog.get_logger()


class GateUI:
    @staticmethod
    def header(text: str):
        print(f"\n{C_BOLD}{'=' * 60}\n{text}\n{'=' * 60}{C_END}")

    @staticmethod
    def step(label: str, text: str, color: str = C_BLUE):
        print(f"{color}[{label}] {text}{C_END}")

    @staticmethod
    def success(text: str):
        print(f"{C_GREEN}[OK] {text}{C_END}")

    @staticmethod
    def warning(text: str):
        print(f"{C_YELLOW}[WARN] {text}{C_END}")

    @staticmethod
    def error(text: str, details: str = ""):
        print(f"{C_RED}[FAIL] {text}{C_END}")
        if details:
            print(f"      {C_YELLOW}{details[:1000]}{C_END}")

    @staticmethod
    def gate_result(gate_name: str, approved: bool, critique: str = ""):
        status = f"{C_GREEN}APPROVED{C_END}" if approved else f"{C_RED}REJECTED{C_END}"
        print(f"   {C_BOLD}{gate_name.upper()}:{C_END} {status}")
        if not approved and critique:
            wrapped = "\n      ".join(critique.split("\n")[:4])
            print(f"      {C_YELLOW}{wrapped}{C_END}")


class ReleaseArcOrchestrator:
    """GATE release controller with Aider as the implementation worker."""

    def __init__(self, target_repo: str, guidelines: Optional[str] = None, config: Optional[GateConfig] = None):
        self.source_repo = Path(target_repo).resolve()
        self.project_name = self.source_repo.name
        self.metadata_dir = Path("metadata") / self.project_name
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.guidelines = guidelines or "Follow professional best practices."
        self.config = config or GateConfig()
        self.supervisor = SupervisorAgent(model_id=self.config.planner_model)

        self.worker = AiderWorkerAgent(model_id=self.config.executor_model)
        if self.config.show_ui_steps:
            GateUI.step("setup", f"Using Aider Agent Worker with {self.config.executor_model}")

        self.gatekeeper = GatekeeperAgent(model_id=self.config.verifier_model)
        self.researcher = ResearcherAgent(model_id=self.config.planner_model)
        self.test_writer = TestWriterAgent(model_id=self.config.executor_model)
        self.meta_analyzer = MetaAnalyzerAgent(model_id=self.config.planner_model)
        self.failure_analyzer = FailureAnalyzer()
        self.prompt_rewriter = PromptRewriter()
        self.plan_repairer = PlanRepairAgent()
        self.model_router = ModelRouter(self.config)
        self.rule_store = RuleStore(self.metadata_dir)
        self.rule_miner = RuleMiner()
        self.circuit_breaker = CircuitBreaker()
        self.rag_provider = "local"

    def _default_rag_provider(self) -> str:
        provider = (os.environ.get("GATE_RAG_PROVIDER") or "").strip().lower()
        if provider in {"disabled", "local", "api"}:
            return provider
        return "local"

    def _should_disable_semantic_rag(self, issue_description: str, manual_plan: Optional[SupervisorPlan] = None) -> bool:
        text = issue_description.lower().strip()
        simple_markers = (
            "typo",
            "rename",
            "readme",
            "docs",
            "documentation",
            "format",
            "lint",
            "version",
            "changelog",
            "comment",
            "whitespace",
            "small",
            "trivial",
        )
        code_markers = (
            "api",
            "database",
            "schema",
            "refactor",
            "feature",
            "test",
            "bug",
            "generate",
            "workflow",
            "typescript",
            "python",
            "javascript",
        )
        simple_text = len(text) < 120 or len(text.split()) < 18
        simple_shape = any(marker in text for marker in simple_markers) and not any(marker in text for marker in code_markers)
        single_file_hint = bool(manual_plan and len(manual_plan.tasks) == 1 and len(manual_plan.tasks[0].allowed_files) <= 1)
        return (simple_text and simple_shape) or single_file_hint

    def _resolve_rag_provider(self, issue_description: str, manual_plan: Optional[SupervisorPlan] = None) -> str:
        provider = self._default_rag_provider()
        if provider == "disabled":
            return provider
        if self._should_disable_semantic_rag(issue_description, manual_plan):
            return "disabled"
        return provider

    async def process_issue(
        self,
        issue_id: str,
        issue_description: str,
        manual_plan: Optional[SupervisorPlan] = None,
    ) -> bool:
        db = await get_db()
        work_repo: Optional[Path] = None
        all_diffs: List[WorkerResult] = []

        try:
            await self._migrate_ledger(db)
            arc_id = await self._get_or_create_arc(db, issue_id)
            active_rules = await self.rule_store.load_active_rules(db, self.project_name)
            active_rules_text = self.rule_store.render_for_prompt(active_rules)
            self.rag_provider = self._resolve_rag_provider(issue_description, manual_plan)

            GateUI.header(f"MISSION: {issue_id}")
            if active_rules:
                GateUI.step("rules", f"Loaded {len(active_rules)} approved learned rule(s).")
            self._ensure_git_repo(self.source_repo)
            work_repo = self._prepare_execution_repo(issue_id, arc_id)
            await db.execute(
                "UPDATE release_arcs SET status = 'in_progress', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (arc_id,),
            )
            await db.commit()

            base_sha = self._git(work_repo, ["rev-parse", "HEAD"]).strip()
            startup_route = self.model_router.route(None, attempt=1)
            self.researcher.model_id = startup_route.classifier_model
            self.meta_analyzer.model_id = startup_route.classifier_model
            repo_context = await self.gather_context(work_repo)

            meta_result = await self.meta_analyzer.invoke({}, db)
            historical_warning = meta_result.output.strip() if meta_result.success and meta_result.output else ""
            if historical_warning:
                GateUI.step("history", "Loaded historical failure warning for this arc.")

            GateUI.step("context", "Building technical discovery report.")
            research_result = await self.researcher.invoke(
                {"repo_path": str(work_repo), "repo_context": repo_context},
                issue_description,
            )
            discovery_report = research_result.output
            if historical_warning:
                discovery_report = (
                    f"{discovery_report}\n\n"
                    f"Historical failure warning:\n{historical_warning}"
                )

            plan = await self._load_or_create_plan(
                db=db,
                arc_id=arc_id,
                issue_id=issue_id,
                issue_description=issue_description,
                repo_context=repo_context,
                discovery_report=discovery_report,
                manual_plan=manual_plan,
                active_rules_text=active_rules_text,
                rag_provider=self.rag_provider,
            )
            if not plan:
                await self._fail_arc(db, arc_id)
                return False

            await self._persist_tasks(db, arc_id, plan)

            can_parallelize = self._tasks_can_run_in_parallel(plan.tasks)

            if can_parallelize and len(plan.tasks) > 1:
                GateUI.step("parallel", f"Executing {len(plan.tasks)} independent tasks in parallel")
                task_repos = []
                for task in plan.tasks:
                    task_repo = work_repo.parent / f"{work_repo.name}_{task.id}"
                    if not task_repo.exists():
                        subprocess.run(["git", "worktree", "add", "-b", f"gate_{arc_id}_{task.id}", str(task_repo), "HEAD"], cwd=str(work_repo), check=True)
                    task_repos.append(task_repo)

                coroutines = []
                for idx, task in enumerate(plan.tasks):
                    coroutines.append(
                        self._execute_task(
                            db=db, arc_id=arc_id, issue_id=issue_id, issue_description=issue_description,
                            repo_context=repo_context, discovery_report=discovery_report, plan=plan, task=task,
                            work_repo=task_repos[idx], all_diffs=all_diffs, active_rules=active_rules, active_rules_text=active_rules_text
                        )
                    )
                
                results = await asyncio.gather(*coroutines, return_exceptions=True)
                
                for idx, result in enumerate(results):
                    if isinstance(result, Exception) or not result[0]:
                        GateUI.error(f"Task {plan.tasks[idx].id} failed during parallel execution.")
                        await self._fail_arc(db, arc_id)
                        return False
                
                for task in plan.tasks:
                    merge = subprocess.run(
                        ["git", "merge", f"gate_{arc_id}_{task.id}", "--no-edit"],
                        cwd=str(work_repo),
                        check=False,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    if merge.returncode != 0:
                        GateUI.error(f"Failed to merge parallel task branch {task.id}.", merge.stdout)
                        await self._fail_arc(db, arc_id)
                        return False

            else:
                if len(plan.tasks) > 1:
                    GateUI.step("parallel", "Falling back to sequential execution because task allowlists overlap or dependencies exist.")
                index = 0
                while index < len(plan.tasks):
                    task = plan.tasks[index]
                    GateUI.header(f"TASK {index + 1}/{len(plan.tasks)}: {task.id}")

                    if await self._task_is_current(db, arc_id, task, work_repo):
                        GateUI.success("Task checkpoint is already current; skipping.")
                        index += 1
                        continue

                    success, plan = await self._execute_task(
                        db=db,
                        arc_id=arc_id,
                        issue_id=issue_id,
                        issue_description=issue_description,
                        repo_context=repo_context,
                        discovery_report=discovery_report,
                        plan=plan,
                        task=task,
                        work_repo=work_repo,
                        all_diffs=all_diffs,
                        active_rules=active_rules,
                        active_rules_text=active_rules_text,
                    )
                    if not success:
                        await self._fail_arc(db, arc_id)
                        return False
                    await self._persist_tasks(db, arc_id, plan)
                    index += 1

            release_files = self._changed_files_between(work_repo, base_sha, "HEAD")
            if release_files:
                GateUI.step("verify", "Running final release verification.")
                sandbox = PipelineMCPClient(str(work_repo))
                await sandbox.connect()
                final_verifier = VerifierEngine(sandbox)
                final_result = await final_verifier.verify_release(repo_context, release_files)
                await sandbox.disconnect()
                await self._record_verification(db, arc_id, None, final_result, 1, release_files)
                if not final_result.success or final_result.skipped:
                    GateUI.error("Final release verification failed.", final_result.reason)
                    await self._fail_arc(db, arc_id)
                    return False

            if all_diffs:
                system_gate = await self.gatekeeper.review_code(
                    issue_description,
                    plan,
                    all_diffs,
                    active_rules_text,
                    repo_path=str(work_repo),
                    rag_provider=self.rag_provider,
                )
                await self._record_gate(db, arc_id, None, "review_code", system_gate, 1)
                GateUI.gate_result("system", system_gate.approved, system_gate.critique)
                if not system_gate.approved:
                    await self._fail_arc(db, arc_id)
                    return False

            await db.execute(
                "UPDATE release_arcs SET status = 'completed', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (arc_id,),
            )
            await db.commit()
            await self._mine_rules(db, arc_id)
            GateUI.header("MISSION COMPLETE")
            GateUI.success(f"Execution repo: {work_repo}")
            return True
        except Exception as exc:
            GateUI.error(f"Fatal pipeline error: {exc}")
            traceback.print_exc()
            return False

    async def gather_context(self, repo_path: Path) -> str:
        def run_capture(args: List[str]) -> str:
            result = subprocess.run(
                args,
                cwd=str(repo_path),
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            return result.stdout.strip()

        top_level = run_capture(["rg", "--files", "-g", "*", "--max-depth", "2"]) if shutil.which("rg") else ""
        if top_level:
            top_entries = top_level.splitlines()[:80]
        else:
            top_entries = sorted(
                str(path.relative_to(repo_path))
                for path in repo_path.rglob("*")
                if len(path.relative_to(repo_path).parts) <= 2
            )[:80]

        key_files = [
            rel
            for rel in top_entries
            if Path(rel).name in {
                "package.json",
                "tsconfig.json",
                "pyproject.toml",
                "requirements.txt",
                "README.md",
                "docker-compose.yml",
                "Dockerfile",
                ".env.example",
            }
        ]

        languages = sorted(
            {
                suffix
                for suffix in (
                    Path(rel).suffix.lower()
                    for rel in top_entries
                    if Path(rel).suffix
                )
                if suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".yml", ".yaml"}
            }
        )

        if shutil.which("rg"):
            package_files = run_capture(
                ["rg", "--files", "-g", "package.json", "-g", "pyproject.toml", "-g", "requirements.txt"]
            )
            package_paths = package_files.splitlines()[:20] if package_files else []
        else:
            package_paths = [
                str(path.relative_to(repo_path))
                for path in repo_path.rglob("*")
                if path.is_file() and path.name in {"package.json", "pyproject.toml", "requirements.txt"}
            ][:20]

        git_status = self._git(repo_path, ["status", "--short"]).splitlines()[:40]

        return (
            f"Repository root: {repo_path}\n"
            f"Detected languages: {', '.join(languages) or 'unknown'}\n"
            f"Key files: {json.dumps(key_files[:20])}\n"
            f"Package manifests: {json.dumps(package_paths)}\n"
            f"Top-level repo snapshot ({len(top_entries)} items):\n"
            + "\n".join(f"- {entry}" for entry in top_entries)
            + "\n\nCurrent git status:\n"
            + ("\n".join(git_status) if git_status else "clean")
        )

    async def _execute_task(
        self,
        db,
        arc_id: int,
        issue_id: str,
        issue_description: str,
        repo_context: str,
        discovery_report: str,
        plan: SupervisorPlan,
        task: TaskDefinition,
        work_repo: Path,
        all_diffs: List[WorkerResult],
        active_rules: List[Rule],
        active_rules_text: str,
    ) -> tuple[bool, SupervisorPlan]:
        feedback = ""
        repair_brief = ""
        failure_history: List[FailureAnalysis] = []
        changed_file_history: List[List[str]] = []
        plan_repairs = 0
        prompt_rewrites = 0

        for attempt in range(1, self.config.max_task_attempts + 1):
            if attempt == 1:
                self._reset_worktree(work_repo)
            await db.execute(
                "UPDATE tasks SET status = 'in_progress', updated_at = CURRENT_TIMESTAMP WHERE arc_id = ? AND task_id = ?",
                (arc_id, task.id),
            )
            await db.commit()

            route = self.model_router.route(failure_history[-1] if failure_history else None, attempt)
            self.supervisor.model_id = route.planner_model
            self.gatekeeper.model_id = route.verifier_model

            GateUI.step("worker", f"Running Aider attempt {attempt}/{self.config.max_task_attempts}.")
            worker_context = {
                "attempt": attempt,
                "repo_path": str(work_repo),
                "guidelines": self.guidelines,
                "repo_context": repo_context,
                "discovery_report": discovery_report,
                "issue_description": issue_description,
                "feedback": feedback,
                "repair_brief": repair_brief,
                "active_rules": active_rules_text,
                "sandbox": PipelineMCPClient(str(work_repo)),
            }
            prompt = self.worker.build_prompt(worker_context, task)
            prompt_digest = prompt_hash(prompt)
            await self._record_prompt_rewrite(
                db,
                arc_id,
                task.id,
                attempt,
                prompt_digest,
                repair_brief or "initial_prompt",
                [rule.id for rule in active_rules],
                route.reason,
            )
            
            should_write_tests = self._should_run_test_writer(task)
            if should_write_tests:
                GateUI.step("tdd", "Invoking Test Writer Agent")
                test_result = await self.test_writer.invoke(worker_context, task)
                if not test_result.success:
                    analysis = await self._handle_failed_attempt(
                        db=db,
                        arc_id=arc_id,
                        task=task,
                        attempt=attempt,
                        changed_files=[],
                        worker_error=f"Test Writer Agent failed: {test_result.output}",
                        failure_history=failure_history,
                        changed_file_history=changed_file_history,
                    )
                    failure_history.append(analysis)
                    feedback, repair_brief, prompt_rewrites = self._next_repair_prompt(
                        task=task,
                        issue_description=issue_description,
                        analysis=analysis,
                        changed_files=[],
                        prior_diff="",
                        active_rules=active_rules,
                        prompt_rewrites=prompt_rewrites,
                    )
                    GateUI.warning(feedback)
                    continue
            else:
                GateUI.step("tdd", "Skipping Test Writer Agent (no explicit test targets in allowlist)")

            GateUI.step("execute", "Invoking Worker Agent")
            worker_result = await self.worker.invoke(
                worker_context,
                task,
            )
            if not worker_result.success:
                analysis = await self._handle_failed_attempt(
                    db=db,
                    arc_id=arc_id,
                    task=task,
                    attempt=attempt,
                    changed_files=[],
                    worker_error=str(worker_result.output),
                    failure_history=failure_history,
                    changed_file_history=changed_file_history,
                )
                failure_history.append(analysis)
                stop_reason = self.circuit_breaker.should_stop(
                    analyses=failure_history,
                    changed_file_history=changed_file_history,
                    latest_changed_files=[],
                )
                if stop_reason:
                    GateUI.error("Circuit breaker stopped retries.", stop_reason)
                    break
                feedback, repair_brief, prompt_rewrites = self._next_repair_prompt(
                    task=task,
                    issue_description=issue_description,
                    analysis=analysis,
                    changed_files=[],
                    prior_diff="",
                    active_rules=active_rules,
                    prompt_rewrites=prompt_rewrites,
                )
                GateUI.warning(feedback)
                continue

            changed_files = self._changed_files(work_repo)
            changed_file_history.append(changed_files)
            diff = self._diff(work_repo)
            w_out: WorkerResult = worker_result.output
            w_out.changed_files = changed_files
            w_out.diff = diff

            allowlist_error = self._validate_changed_files(task, changed_files, work_repo)
            if allowlist_error:
                analysis = await self._handle_failed_attempt(
                    db=db,
                    arc_id=arc_id,
                    task=task,
                    attempt=attempt,
                    changed_files=changed_files,
                    allowlist_error=allowlist_error,
                    failure_history=failure_history,
                    changed_file_history=changed_file_history,
                )
                failure_history.append(analysis)
                stop_reason = self.circuit_breaker.should_stop(
                    analyses=failure_history,
                    changed_file_history=changed_file_history,
                    latest_changed_files=changed_files,
                )
                if stop_reason:
                    GateUI.error("Circuit breaker stopped retries.", stop_reason)
                    break
                if analysis.recommended_action == "repair_plan" and plan_repairs < self.config.max_plan_repairs:
                    plan, task, plan_repairs = await self._repair_plan(
                        db=db,
                        arc_id=arc_id,
                        issue_id=issue_id,
                        issue_description=issue_description,
                        plan=plan,
                        task=task,
                        analysis=analysis,
                        changed_files=changed_files,
                        plan_repairs=plan_repairs,
                    )
                    feedback = f"Plan repaired after {analysis.failure_class}; retrying with updated allowlist/context."
                    repair_brief = feedback
                else:
                    feedback, repair_brief, prompt_rewrites = self._next_repair_prompt(
                        task=task,
                        issue_description=issue_description,
                        analysis=analysis,
                        changed_files=changed_files,
                        prior_diff=diff,
                        active_rules=active_rules,
                        prompt_rewrites=prompt_rewrites,
                    )
                GateUI.warning(feedback)
                continue

            if self.config.allow_dependency_install:
                bootstrap_error = await self._bootstrap_dependencies(work_repo, changed_files)
                if bootstrap_error:
                    analysis = await self._handle_failed_attempt(
                        db=db,
                        arc_id=arc_id,
                        task=task,
                        attempt=attempt,
                        changed_files=changed_files,
                        worker_error=bootstrap_error,
                        failure_history=failure_history,
                        changed_file_history=changed_file_history,
                    )
                    failure_history.append(analysis)
                    GateUI.error("Dependency bootstrap failed.", bootstrap_error)
                    break
                changed_files = self._changed_files(work_repo)
                w_out.changed_files = changed_files
                w_out.diff = self._diff(work_repo)

                allowlist_error = self._validate_changed_files(task, changed_files, work_repo)
                if allowlist_error:
                    analysis = await self._handle_failed_attempt(
                        db=db,
                        arc_id=arc_id,
                        task=task,
                        attempt=attempt,
                        changed_files=changed_files,
                        allowlist_error=allowlist_error,
                        failure_history=failure_history,
                        changed_file_history=changed_file_history,
                    )
                    failure_history.append(analysis)
                    if analysis.recommended_action == "repair_plan" and plan_repairs < self.config.max_plan_repairs:
                        plan, task, plan_repairs = await self._repair_plan(
                            db=db,
                            arc_id=arc_id,
                            issue_id=issue_id,
                            issue_description=issue_description,
                            plan=plan,
                            task=task,
                            analysis=analysis,
                            changed_files=changed_files,
                            plan_repairs=plan_repairs,
                        )
                        feedback = f"Plan repaired after dependency-generated allowlist change; retrying."
                        repair_brief = feedback
                    else:
                        feedback, repair_brief, prompt_rewrites = self._next_repair_prompt(
                            task=task,
                            issue_description=issue_description,
                            analysis=analysis,
                            changed_files=changed_files,
                            prior_diff=w_out.diff,
                            active_rules=active_rules,
                            prompt_rewrites=prompt_rewrites,
                        )
                    GateUI.warning(feedback)
                    continue

            sandbox = PipelineMCPClient(str(work_repo))
            await sandbox.connect()
            verifier = VerifierEngine(sandbox)
            verification = await verifier.verify(
                task_description=task.description,
                repo_context=repo_context,
                changed_files=changed_files or task.allowed_files,
                acceptance_criteria=task.acceptance_criteria,
            )
            await sandbox.disconnect()
            await self._record_verification(db, arc_id, task.id, verification, attempt, changed_files)
            if not verification.success or verification.skipped:
                analysis = await self._handle_failed_attempt(
                    db=db,
                    arc_id=arc_id,
                    task=task,
                    attempt=attempt,
                    changed_files=changed_files,
                    verifier_result=verification,
                    failure_history=failure_history,
                    changed_file_history=changed_file_history,
                )
                failure_history.append(analysis)
                stop_reason = self.circuit_breaker.should_stop(
                    analyses=failure_history,
                    changed_file_history=changed_file_history,
                    latest_changed_files=changed_files,
                )
                if stop_reason:
                    GateUI.error("Circuit breaker stopped retries.", stop_reason)
                    break
                if analysis.recommended_action == "repair_plan" and plan_repairs < self.config.max_plan_repairs:
                    plan, task, plan_repairs = await self._repair_plan(
                        db=db,
                        arc_id=arc_id,
                        issue_id=issue_id,
                        issue_description=issue_description,
                        plan=plan,
                        task=task,
                        analysis=analysis,
                        changed_files=changed_files,
                        plan_repairs=plan_repairs,
                    )
                    feedback = f"Plan repaired after {analysis.failure_class}; retrying."
                    repair_brief = feedback
                elif analysis.recommended_action == "block_environment":
                    GateUI.error("Deterministic environment/dependency failure.", verification.reason)
                    break
                else:
                    feedback, repair_brief, prompt_rewrites = self._next_repair_prompt(
                        task=task,
                        issue_description=issue_description,
                        analysis=analysis,
                        changed_files=changed_files,
                        prior_diff=w_out.diff,
                        verifier_result=verification,
                        active_rules=active_rules,
                        prompt_rewrites=prompt_rewrites,
                    )
                GateUI.warning(feedback)
                continue

                code_gate = await self.gatekeeper.codereview(
                    task,
                    w_out,
                    active_rules_text,
                    repo_path=str(work_repo),
                    rag_provider=self.rag_provider,
                )
            await self._record_gate(db, arc_id, task.id, "codereview", code_gate, attempt)
            GateUI.gate_result("code", code_gate.approved, code_gate.critique)
            if not code_gate.approved:
                analysis = await self._handle_failed_attempt(
                    db=db,
                    arc_id=arc_id,
                    task=task,
                    attempt=attempt,
                    changed_files=changed_files,
                    gate_critique=code_gate.critique,
                    failure_history=failure_history,
                    changed_file_history=changed_file_history,
                )
                failure_history.append(analysis)
                if analysis.recommended_action == "repair_plan" and plan_repairs < self.config.max_plan_repairs:
                    plan, task, plan_repairs = await self._repair_plan(
                        db=db,
                        arc_id=arc_id,
                        issue_id=issue_id,
                        issue_description=issue_description,
                        plan=plan,
                        task=task,
                        analysis=analysis,
                        changed_files=changed_files,
                        plan_repairs=plan_repairs,
                    )
                    feedback = f"Plan repaired after Gatekeeper critique; retrying."
                    repair_brief = feedback
                else:
                    feedback, repair_brief, prompt_rewrites = self._next_repair_prompt(
                        task=task,
                        issue_description=issue_description,
                        analysis=analysis,
                        changed_files=changed_files,
                        prior_diff=w_out.diff,
                        gate_critique=code_gate.critique,
                        active_rules=active_rules,
                        prompt_rewrites=prompt_rewrites,
                    )
                continue

            commit_sha = self._commit_exact_files(work_repo, changed_files, task.id)
            await db.execute(
                "UPDATE tasks SET status = 'completed', commit_sha = ?, updated_at = CURRENT_TIMESTAMP WHERE arc_id = ? AND task_id = ?",
                (commit_sha, arc_id, task.id),
            )
            await db.commit()

            all_diffs.append(w_out)
            GateUI.success(f"Task finalized at {commit_sha[:8]}.")
            return True, plan

        await db.execute(
            "UPDATE tasks SET status = 'failed', updated_at = CURRENT_TIMESTAMP WHERE arc_id = ? AND task_id = ?",
            (arc_id, task.id),
        )
        await db.commit()
        GateUI.error("Task exhausted all attempts.", feedback)
        return False, plan

    async def _load_or_create_plan(
        self,
        db,
        arc_id: int,
        issue_id: str,
        issue_description: str,
        repo_context: str,
        discovery_report: str,
        manual_plan: Optional[SupervisorPlan],
        active_rules_text: str,
        rag_provider: str,
    ) -> Optional[SupervisorPlan]:
        plan = manual_plan or self._load_plan_file(issue_id)
        guidelines = f"{self.guidelines}\n\nAPPROVED LEARNED RULES:\n{active_rules_text}"

        if not plan:
            GateUI.step("plan", "Drafting implementation plan.")
            plan_result = await self.supervisor.invoke(
                {
                    "guidelines": guidelines,
                    "repo_path": str(self.source_repo),
                    "repo_context": repo_context,
                    "discovery_report": discovery_report,
                    "rag_provider": rag_provider,
                },
                issue_description,
            )
            if not plan_result.success:
                GateUI.error("Supervisor failed to create a plan.", str(plan_result.output))
                return None
            plan = plan_result.output

        for attempt in range(1, 4):
            plan_gate = await self.gatekeeper.review_plan(issue_description, plan, active_rules_text)
            await self._record_gate(db, arc_id, None, "review_plan", plan_gate, attempt)
            GateUI.gate_result("plan", plan_gate.approved, plan_gate.critique)
            if plan_gate.approved:
                self._write_plan_file(issue_id, issue_description, plan)
                if await self._latest_plan_version(db, arc_id) == 0:
                    await self._record_plan_version(db, arc_id, 1, plan, "initial_approved_plan", None)
                return plan

            revision = await self.supervisor.revise_plan(
                plan,
                plan_gate.critique,
                {
                    "guidelines": guidelines,
                    "repo_path": str(self.source_repo),
                    "repo_context": repo_context,
                    "discovery_report": discovery_report,
                    "rag_provider": rag_provider,
                },
            )
            if not revision.success:
                GateUI.error("Supervisor failed to revise rejected plan.", str(revision.output))
                return None
            plan = revision.output

        GateUI.error("Plan rejected after revision attempts.")
        return None

    def _load_plan_file(self, issue_id: str) -> Optional[SupervisorPlan]:
        plan_file = self.metadata_dir / f"PLAN_{issue_id}.md"
        if not plan_file.exists():
            return None
        content = plan_file.read_text(encoding="utf-8")
        try:
            json_str = content.split("```json", 1)[1].split("```", 1)[0].strip()
            return SupervisorPlan(tasks=[TaskDefinition(**task) for task in json.loads(json_str)])
        except (IndexError, json.JSONDecodeError, TypeError) as exc:
            GateUI.warning(f"Could not parse existing plan file: {exc}")
            return None

    def _write_plan_file(self, issue_id: str, issue_description: str, plan: SupervisorPlan) -> None:
        plan_file = self.metadata_dir / f"PLAN_{issue_id}.md"
        payload = [task.model_dump() for task in plan.tasks]
        plan_file.write_text(
            f"# Implementation Plan for {issue_id}\n\n"
            f"## Original Requirement\n{issue_description}\n\n"
            f"## Execution Tasks\n```json\n{json.dumps(payload, indent=2)}\n```\n",
            encoding="utf-8",
        )

    async def _persist_tasks(self, db, arc_id: int, plan: SupervisorPlan) -> None:
        for task in plan.tasks:
            async with db.execute(
                "SELECT id FROM tasks WHERE arc_id = ? AND task_id = ? ORDER BY id DESC LIMIT 1",
                (arc_id, task.id),
            ) as cursor:
                row = await cursor.fetchone()
            target_files = json.dumps(task.allowed_files)
            dependencies = json.dumps(task.dependencies)
            if row:
                await db.execute(
                    "UPDATE tasks SET description = ?, dependencies = ?, target_files = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (task.description, dependencies, target_files, row[0]),
                )
            else:
                await db.execute(
                    "INSERT INTO tasks (arc_id, task_id, description, dependencies, target_files, status) VALUES (?, ?, ?, ?, ?, 'pending')",
                    (arc_id, task.id, task.description, dependencies, target_files),
                )
        await db.commit()

    async def _task_is_current(self, db, arc_id: int, task: TaskDefinition, work_repo: Path) -> bool:
        async with db.execute(
            "SELECT status, commit_sha FROM tasks WHERE arc_id = ? AND task_id = ? ORDER BY id DESC LIMIT 1",
            (arc_id, task.id),
        ) as cursor:
            row = await cursor.fetchone()
        if not row or row[0] != "completed" or not row[1]:
            return False

        actual_sha = self._git(work_repo, ["rev-parse", "HEAD"]).strip()
        if actual_sha != row[1]:
            return False
        return all((work_repo / path).is_file() for path in task.allowed_files)

    async def _get_or_create_arc(self, db, issue_id: str) -> int:
        async with db.execute(
            "SELECT id FROM release_arcs WHERE issue_id = ? AND status IN ('planning', 'in_progress') ORDER BY id DESC LIMIT 1",
            (issue_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            return row[0]

        cursor = await db.execute(
            "INSERT INTO release_arcs (issue_id, repository, status) VALUES (?, ?, 'planning')",
            (issue_id, str(self.source_repo)),
        )
        await db.commit()
        return cursor.lastrowid

    async def _record_gate(
        self,
        db,
        arc_id: int,
        task_id: Optional[str],
        gate_name: str,
        result: GateResult,
        attempt_number: int,
    ) -> None:
        await db.execute(
            """
            INSERT INTO gate_reviews (
                arc_id, task_id, gate_name, model_id, status, error_type,
                critique_summary, attempt_number, prompt_tokens, completion_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                arc_id,
                task_id,
                gate_name,
                self.gatekeeper.model_id,
                "approved" if result.approved else "rejected",
                result.error_type,
                result.critique,
                attempt_number,
                result.metrics.get("prompt_tokens", 0),
                result.metrics.get("completion_tokens", 0),
            ),
        )
        await db.commit()

    async def _record_verification(
        self,
        db,
        arc_id: int,
        task_id: Optional[str],
        result: VerificationResult,
        attempt_number: int,
        changed_files: List[str],
    ) -> None:
        await db.execute(
            """
            INSERT INTO verification_runs (
                arc_id, task_id, status, reason, used_command, evidence,
                changed_files, attempt_number
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                arc_id,
                task_id,
                "skipped" if result.skipped else "passed" if result.success else "failed",
                result.reason,
                result.used_command,
                result.evidence,
                json.dumps(changed_files),
                attempt_number,
            ),
        )
        await db.commit()

    async def _handle_failed_attempt(
        self,
        db,
        arc_id: int,
        task: TaskDefinition,
        attempt: int,
        changed_files: List[str],
        failure_history: List[FailureAnalysis],
        changed_file_history: List[List[str]],
        verifier_result: Optional[VerificationResult] = None,
        gate_critique: str = "",
        allowlist_error: str = "",
        worker_error: str = "",
    ) -> FailureAnalysis:
        analysis = self.failure_analyzer.analyze(
            task=task,
            attempt=attempt,
            changed_files=changed_files,
            verifier_result=verifier_result,
            gate_critique=gate_critique,
            allowlist_error=allowlist_error,
            worker_error=worker_error,
            previous_failures=failure_history,
        )
        await db.execute(
            """
            INSERT INTO failure_analyses (
                arc_id, task_id, attempt_number, failure_class, confidence,
                evidence, recommended_action, changed_files
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                arc_id,
                task.id,
                attempt,
                analysis.failure_class,
                analysis.confidence,
                analysis.evidence,
                analysis.recommended_action,
                json.dumps(changed_files),
            ),
        )
        await db.commit()
        GateUI.step("analyze", f"{analysis.failure_class} -> {analysis.recommended_action} ({analysis.confidence:.2f})")
        return analysis

    def _next_repair_prompt(
        self,
        *,
        task: TaskDefinition,
        issue_description: str,
        analysis: FailureAnalysis,
        changed_files: List[str],
        prior_diff: str,
        active_rules: List[Rule],
        prompt_rewrites: int,
        verifier_result: Optional[VerificationResult] = None,
        gate_critique: str = "",
    ) -> tuple[str, str, int]:
        if prompt_rewrites >= self.config.max_prompt_rewrites:
            return (
                f"Prompt rewrite budget exhausted after {prompt_rewrites} rewrite(s).",
                "",
                prompt_rewrites,
            )

        rewrite = self.prompt_rewriter.rewrite(
            task=task,
            issue_description=issue_description,
            analysis=analysis,
            changed_files=changed_files,
            prior_diff=prior_diff,
            verifier_result=verifier_result,
            gate_critique=gate_critique,
            active_rules=active_rules,
        )
        return rewrite.rewrite_summary, rewrite.repair_brief, prompt_rewrites + 1

    async def _repair_plan(
        self,
        *,
        db,
        arc_id: int,
        issue_id: str,
        issue_description: str,
        plan: SupervisorPlan,
        task: TaskDefinition,
        analysis: FailureAnalysis,
        changed_files: List[str],
        plan_repairs: int,
    ) -> tuple[SupervisorPlan, TaskDefinition, int]:
        completed_task_ids = await self._completed_task_ids(db, arc_id)
        repaired, reason = self.plan_repairer.repair(
            plan=plan,
            failed_task=task,
            analysis=analysis,
            changed_files=changed_files,
            completed_task_ids=completed_task_ids,
        )
        parent_version = await self._latest_plan_version(db, arc_id)
        version = parent_version + 1
        await self._record_plan_version(db, arc_id, version, repaired, reason, parent_version)
        await self._persist_tasks(db, arc_id, repaired)
        self._write_plan_file(issue_id, issue_description, repaired)
        repaired_task = next((candidate for candidate in repaired.tasks if candidate.id == task.id), task)
        GateUI.step("plan", f"Recorded repaired plan version {version}.")
        return repaired, repaired_task, plan_repairs + 1

    async def _completed_task_ids(self, db, arc_id: int) -> set[str]:
        async with db.execute(
            "SELECT task_id FROM tasks WHERE arc_id = ? AND status = 'completed'",
            (arc_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return {row["task_id"] for row in rows if row["task_id"]}

    async def _record_plan_version(
        self,
        db,
        arc_id: int,
        version: int,
        plan: SupervisorPlan,
        reason: str,
        parent_version: Optional[int],
    ) -> None:
        payload = json.dumps([task.model_dump() for task in plan.tasks], indent=2)
        await db.execute(
            """
            INSERT INTO plan_versions (arc_id, version, plan_json, reason, parent_version)
            VALUES (?, ?, ?, ?, ?)
            """,
            (arc_id, version, payload, reason, parent_version),
        )
        await db.commit()

    async def _latest_plan_version(self, db, arc_id: int) -> int:
        async with db.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM plan_versions WHERE arc_id = ?",
            (arc_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["version"] or 0)

    async def _record_prompt_rewrite(
        self,
        db,
        arc_id: int,
        task_id: str,
        attempt: int,
        prompt_digest: str,
        rewrite_summary: str,
        active_rule_ids: List[str],
        model_route: str,
    ) -> None:
        await db.execute(
            """
            INSERT INTO prompt_rewrites (
                arc_id, task_id, attempt_number, prompt_hash, rewrite_summary,
                active_rules_used, model_route
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                arc_id,
                task_id,
                attempt,
                prompt_digest,
                rewrite_summary[:1000],
                json.dumps(active_rule_ids),
                model_route,
            ),
        )
        await db.commit()

    async def _mine_rules(self, db, arc_id: int) -> None:
        proposals = await self.rule_miner.mine_arc(db, arc_id, self.project_name)
        if not proposals:
            GateUI.step("rules", "No new durable rule proposals.")
            return

        GateUI.step("rules", f"Proposed {len(proposals)} inactive learned rule(s).")
        for proposal in proposals[:5]:
            GateUI.warning(
                f"rule_proposal #{proposal['id']} [{proposal['scope']}] "
                f"{proposal['rule_text']} (confidence {proposal['confidence']:.2f})"
            )
        if self.config.rule_mode == "review_first":
            GateUI.step("rules", "Review proposals in rule_proposals; set status='approved' to activate.")

    async def _fail_arc(self, db, arc_id: int) -> None:
        await self._mark_arc_failed(db, arc_id)
        await self._mine_rules(db, arc_id)

    async def _migrate_ledger(self, db) -> None:
        migrations = [
            "ALTER TABLE tasks ADD COLUMN commit_sha TEXT",
            "ALTER TABLE tasks ADD COLUMN target_files TEXT",
            """
            CREATE TABLE IF NOT EXISTS verification_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arc_id INTEGER NOT NULL,
                task_id TEXT,
                status TEXT NOT NULL,
                reason TEXT,
                used_command TEXT,
                evidence TEXT,
                changed_files TEXT,
                attempt_number INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (arc_id) REFERENCES release_arcs(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS failure_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arc_id INTEGER NOT NULL,
                task_id TEXT,
                attempt_number INTEGER DEFAULT 1,
                failure_class TEXT NOT NULL,
                confidence REAL DEFAULT 0,
                evidence TEXT,
                recommended_action TEXT,
                changed_files TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (arc_id) REFERENCES release_arcs(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS plan_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arc_id INTEGER NOT NULL,
                version INTEGER NOT NULL,
                plan_json TEXT NOT NULL,
                reason TEXT,
                parent_version INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (arc_id) REFERENCES release_arcs(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS rule_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_text TEXT NOT NULL,
                scope TEXT NOT NULL DEFAULT 'global',
                source_failures TEXT,
                confidence REAL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'proposed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS prompt_rewrites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arc_id INTEGER NOT NULL,
                task_id TEXT,
                attempt_number INTEGER DEFAULT 1,
                prompt_hash TEXT NOT NULL,
                rewrite_summary TEXT,
                active_rules_used TEXT,
                model_route TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (arc_id) REFERENCES release_arcs(id)
            )
            """,
        ]
        for migration in migrations:
            try:
                await db.execute(migration)
                await db.commit()
            except Exception:
                pass

        try:
            await db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_arc_task ON tasks(arc_id, task_id)")
            await db.commit()
        except Exception:
            GateUI.warning("Task table contains duplicate task rows; future inserts will still use latest-row reconciliation.")

    async def _mark_arc_failed(self, db, arc_id: int) -> None:
        await db.execute(
            "UPDATE release_arcs SET status = 'failed', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (arc_id,),
        )
        await db.commit()

    def _ensure_git_repo(self, repo: Path) -> None:
        if not repo.exists():
            raise FileNotFoundError(f"Target repository does not exist: {repo}")

        is_repo = self._run(["git", "rev-parse", "--is-inside-work-tree"], repo, check=False).strip() == "true"
        if not is_repo:
            self._run(["git", "init"], repo)

        self._run(["git", "config", "user.email", "gate@sevisolutions.com"], repo)
        self._run(["git", "config", "user.name", "Gatekeeper"], repo)

        has_head = self._run(["git", "rev-parse", "--verify", "HEAD"], repo, check=False)
        if "fatal" in has_head.lower() or not has_head.strip():
            self._run(["git", "commit", "--allow-empty", "-m", "GATE Init"], repo)

    def _prepare_execution_repo(self, issue_id: str, arc_id: int) -> Path:
        if not self.config.use_git_worktree:
            return self.source_repo

        slug = self._slug(issue_id)
        branch = f"gate/{slug}-{arc_id}"
        worktree_root = Path(os.environ.get("GATE_WORKTREE_ROOT", tempfile.gettempdir())) / "gate-worktrees"
        worktree_root.mkdir(parents=True, exist_ok=True)
        worktree = worktree_root / f"{self.project_name}-{slug}-{arc_id}"

        if (worktree / ".git").exists():
            GateUI.step("worktree", f"Reusing isolated worktree {worktree}.")
            self._run(["git", "reset", "--hard", "HEAD"], worktree)
            self._run(["git", "clean", "-fd"], worktree)
            return worktree

        if worktree.exists():
            raise RuntimeError(f"Worktree path exists but is not a git worktree: {worktree}")

        GateUI.step("worktree", f"Creating isolated worktree {worktree}.")
        self._run(["git", "worktree", "add", "-B", branch, str(worktree), "HEAD"], self.source_repo)
        self._run(["git", "config", "user.email", "gate@sevisolutions.com"], worktree)
        self._run(["git", "config", "user.name", "Gatekeeper"], worktree)
        return worktree

    def _reset_worktree(self, work_repo: Path) -> None:
        self._run(["git", "reset", "--hard", "HEAD"], work_repo)
        self._run(["git", "clean", "-fd"], work_repo)

    async def _bootstrap_dependencies(self, work_repo: Path, changed_files: List[str]) -> Optional[str]:
        package_dirs = []
        for rel_path in changed_files:
            path = work_repo / rel_path
            current = path.parent
            while work_repo in [current, *current.parents]:
                package_json = current / "package.json"
                if package_json.exists():
                    package_dirs.append(current)
                    break
                if current == work_repo:
                    break
                current = current.parent

        for package_dir in sorted(set(package_dirs)):
            if (package_dir / "node_modules").exists():
                continue
            rel_dir = str(package_dir.relative_to(work_repo))
            GateUI.step("deps", f"Installing dependencies in {rel_dir}.")
            sandbox = PipelineMCPClient(str(work_repo))
            await sandbox.connect()
            output, code = await sandbox.execute_command(f"cd {shlex.quote(rel_dir)} && npm install")
            await sandbox.disconnect()
            if code != 0:
                return f"Dependency install failed in {rel_dir}: {output[:1200]}"
        return None

    def _validate_changed_files(self, task: TaskDefinition, changed_files: List[str], work_repo: Optional[Path] = None) -> Optional[str]:
        if not changed_files:
            if work_repo:
                filtered_allowed = [
                    path for path in task.allowed_files
                    if not self._is_dependency_artifact(path) and not self._is_temp_artifact(path)
                ]
                if all((work_repo / path).is_file() for path in filtered_allowed):
                    return None
            return "Aider did not produce any file changes."

        temp_files = [path for path in changed_files if self._is_temp_artifact(path)]
        if temp_files:
            return f"Aider produced forbidden temporary artifacts: {', '.join(temp_files)}"

        protected = self._protected_gate_source_changes(changed_files)
        if protected:
            return (
                "Autonomous arcs may not modify GATE source-control paths. "
                f"Protected changes: {protected}. Generate a proposal instead."
            )

        allowed = set(self._normalize_files(task.allowed_files))
        if not allowed:
            return None
        allowed = set(self._with_generated_lockfiles(allowed))

        unexpected = [path for path in changed_files if path not in allowed]
        if unexpected:
            return (
                "Aider modified files outside the task allowlist. "
                f"Allowed: {sorted(allowed)}. Unexpected: {unexpected}"
            )
        return None

    def _with_generated_lockfiles(self, allowed: set[str]) -> List[str]:
        expanded = set(allowed)
        for rel_path in allowed:
            if rel_path.endswith("package.json"):
                package_dir = os.path.dirname(rel_path)
                lockfile = os.path.join(package_dir, "package-lock.json") if package_dir else "package-lock.json"
                expanded.add(lockfile)
        return sorted(expanded)

    def _protected_gate_source_changes(self, changed_files: List[str]) -> List[str]:
        if not self.config.protect_gate_source_paths:
            return []
        gate_root = Path.cwd().resolve()
        if self.source_repo != gate_root:
            return []
        protected_prefixes = ("agents/", "orchestrator/", "ledger/", "integrations/", "environment/")
        protected_files = ("trust_ledger.db",)
        return [
            path
            for path in changed_files
            if path.startswith(protected_prefixes) or path in protected_files
        ]

    def _commit_exact_files(self, work_repo: Path, files: List[str], task_id: str) -> str:
        files = self._normalize_files(files)
        if files:
            self._run(["git", "add", "--", *files], work_repo)
        staged = self._git(work_repo, ["diff", "--cached", "--name-only"]).splitlines()
        if not staged:
            self._run(["git", "commit", "--allow-empty", "-m", f"GATE: {task_id} (no changes)"], work_repo)
            return self._git(work_repo, ["rev-parse", "HEAD"]).strip()
        self._run(["git", "commit", "-m", f"GATE: {task_id}"], work_repo)
        return self._git(work_repo, ["rev-parse", "HEAD"]).strip()

    def _changed_files(self, work_repo: Path) -> List[str]:
        tracked = self._git(work_repo, ["diff", "--name-only"]).splitlines()
        untracked = self._git(work_repo, ["ls-files", "--others", "--exclude-standard"]).splitlines()
        return [
            path
            for path in self._normalize_files(sorted({*tracked, *untracked}))
            if not self._is_dependency_artifact(path) and not self._is_temp_artifact(path)
        ]

    def _changed_files_between(self, work_repo: Path, base: str, head: str) -> List[str]:
        output = self._git(work_repo, ["diff", "--name-only", f"{base}..{head}"])
        return self._normalize_files(output.splitlines())

    def _diff(self, work_repo: Path) -> str:
        diff = self._git(work_repo, ["diff", "--no-ext-diff", "--binary"]).rstrip()
        chunks = [diff] if diff else []
        for rel_path in self._git(work_repo, ["ls-files", "--others", "--exclude-standard"]).splitlines():
            if self._is_dependency_artifact(rel_path):
                continue
            path = work_repo / rel_path
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = "[binary or non-UTF-8 file omitted]"
            chunks.append(f"--- /dev/null\n+++ b/{rel_path}\n@@ untracked file @@\n{content[:12000]}")
        return "\n\n".join(chunk for chunk in chunks if chunk)

    def _git(self, cwd: Path, args: List[str]) -> str:
        return self._run(["git", *args], cwd, check=False)

    def _run(self, command: List[str], cwd: Path, check: bool = True) -> str:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed ({' '.join(command)}): {result.stdout}")
        return result.stdout

    def _normalize_files(self, files: List[str]) -> List[str]:
        normalized = []
        for path in files:
            path = path.strip()
            if not path:
                continue
            normalized.append(path[2:] if path.startswith("./") else path)
        return sorted(set(normalized))

    def _tasks_can_run_in_parallel(self, tasks: List[TaskDefinition]) -> bool:
        if any(task.dependencies for task in tasks):
            return False

        seen_files: set[str] = set()
        for task in tasks:
            allowlist = set(self._normalize_files(task.allowed_files))
            if not allowlist:
                return False
            if seen_files.intersection(allowlist):
                return False
            seen_files.update(allowlist)
        return True

    def _should_run_test_writer(self, task: TaskDefinition) -> bool:
        if not getattr(task, "requires_tests", True):
            return False

        test_markers = ("/tests/", "/__tests__/", ".test.", ".spec.", "test_", "_test.")
        for rel_path in task.allowed_files:
            normalized = f"/{rel_path.lower()}"
            if any(marker in normalized for marker in test_markers):
                return True
        return False

    def _is_dependency_artifact(self, rel_path: str) -> bool:
        parts = rel_path.split("/")
        return "node_modules" in parts or "vendor" in parts and rel_path.endswith("/vendor")

    def _is_temp_artifact(self, rel_path: str) -> bool:
        name = os.path.basename(rel_path)
        parts = rel_path.split("/")
        return (
            name.startswith(".tmp")
            or name.endswith(".tmp")
            or name.endswith(".bak")
            or rel_path.endswith(".rej")
            or name.endswith(".pyc")
            or "__pycache__" in parts
        )

    def _slug(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-").lower()
        return slug or "release"


if __name__ == "__main__":
    async def run():
        try:
            session_file = Path("metadata/LAST_SESSION.json")
            target_repo = None
            issue_id = None

            if session_file.exists():
                last = json.loads(session_file.read_text(encoding="utf-8"))
                response = input(f"{C_YELLOW}Resume {last.get('issue_id')}? [Y/n]: {C_END}").strip().lower()
                if response != "n":
                    target_repo = last.get("repo")
                    issue_id = last.get("issue_id")

            if not target_repo:
                target_repo = input("Target Repo: ").strip() or "."

            project_name = Path(target_repo).resolve().name
            metadata_dir = Path("metadata") / project_name
            metadata_dir.mkdir(parents=True, exist_ok=True)

            guidelines = "Standard practices."
            config_path = metadata_dir / "gate.yml"
            if config_path.exists():
                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                guidelines = (
                    f"GOAL: {cfg.get('project_goal', '')}\n"
                    f"STACK: {cfg.get('technical_stack', '')}\n"
                    f"ARCH: {cfg.get('architecture', '')}\n"
                    f"RULES: {cfg.get('guidelines', '')}"
                )

            if not issue_id:
                issue_id = input("Issue ID: ").strip()

            session_file.write_text(json.dumps({"repo": target_repo, "issue_id": issue_id}), encoding="utf-8")

            plan_file = metadata_dir / f"PLAN_{issue_id}.md"
            description = ""
            if plan_file.exists():
                content = plan_file.read_text(encoding="utf-8")
                if "## Original Requirement\n" in content:
                    description = content.split("## Original Requirement\n", 1)[1].split("## Execution Tasks", 1)[0].strip()
            if not description:
                description = input("Task Description: ").strip()

            orchestrator = ReleaseArcOrchestrator(target_repo=target_repo, guidelines=guidelines)
            await orchestrator.process_issue(issue_id, description)
        except Exception as exc:
            GateUI.error(f"Fatal startup error: {exc}")
        finally:
            print(f"{C_GREEN}Shutdown.{C_END}")

    asyncio.run(run())
