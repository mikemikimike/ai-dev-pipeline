import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import List, Optional, Set

import structlog

from agents.models import VerificationResult
from environment.mcp_client import PipelineMCPClient

logger = structlog.get_logger()


class FileValidator:
    def __init__(self, sandbox: PipelineMCPClient, repo_root: Path):
        self.sandbox = sandbox
        self.repo_root = repo_root
        
    async def validate(self, rel_path: str) -> VerificationResult:
        raise NotImplementedError()

class JsonValidator(FileValidator):
    async def validate(self, rel_path: str) -> VerificationResult:
        script = "const fs=require('fs'); JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));"
        cmd = f"node -e {shlex.quote(script)} {shlex.quote(rel_path)}"
        output, code = await self.sandbox.execute_command(cmd)
        if code != 0:
            return VerificationResult(success=False, reason=f"Invalid JSON in {rel_path}: {output[:800]}", used_command=cmd, evidence=output)
        return VerificationResult(success=True, reason=f"JSON parsed: {rel_path}", evidence=cmd)

class JavaScriptValidator(FileValidator):
    async def validate(self, rel_path: str) -> VerificationResult:
        cmd = f"node --check {shlex.quote(rel_path)}"
        output, code = await self.sandbox.execute_command(cmd)
        if code != 0:
            return VerificationResult(success=False, reason=f"JavaScript syntax failed for {rel_path}: {output[:800]}", used_command=cmd, evidence=output)
        return VerificationResult(success=True, reason=f"JavaScript syntax passed: {rel_path}", evidence=cmd)

class PythonValidator(FileValidator):
    async def validate(self, rel_path: str) -> VerificationResult:
        cmd = f"python3 -m py_compile {shlex.quote(rel_path)}"
        output, code = await self.sandbox.execute_command(cmd)
        if code != 0:
            return VerificationResult(success=False, reason=f"Python syntax failed for {rel_path}: {output[:800]}", used_command=cmd, evidence=output)
        return VerificationResult(success=True, reason=f"Python syntax passed: {rel_path}", evidence=cmd)

class TypeScriptValidator(FileValidator):
    async def validate(self, rel_path: str) -> VerificationResult:
        cmd = f"npx tsc --noEmit --skipLibCheck --target esnext --module commonjs {shlex.quote(rel_path)}"
        output, code = await self.sandbox.execute_command(cmd)
        if code != 0:
            return VerificationResult(success=False, reason=f"TypeScript check failed for {rel_path}: {output[:800]}", used_command=cmd, evidence=output)
        return VerificationResult(success=True, reason=f"TypeScript check passed: {rel_path}", evidence=cmd)


class VerifierEngine:
    """Deterministic verifier for GATE task acceptance."""

    def __init__(self, sandbox: PipelineMCPClient):
        self.sandbox = sandbox
        self.repo_root = Path(self.sandbox.target_repo_path).resolve()

    def _detect_docker_image(self) -> str:
        if (self.repo_root / "package.json").exists():
            return "node:20-alpine"
        return "python:3.10-slim"

    async def _run_in_docker(self, cmd: str, rel_dir: str = "") -> tuple[str, int]:
        import docker
        client = docker.from_env()
        image = self._detect_docker_image()
        target_dir = f"/workspace/{rel_dir}" if rel_dir else "/workspace"
        try:
            container = client.containers.run(
                image,
                command=["sh", "-c", cmd],
                volumes={str(self.repo_root): {'bind': '/workspace', 'mode': 'rw'}},
                working_dir=target_dir,
                detach=True
            )
            result = container.wait(timeout=120)
            logs = container.logs().decode("utf-8", errors="replace")
            container.remove(force=True)
            return logs, result['StatusCode']
        except Exception as e:
            return f"Docker execution error: {str(e)}", 1

    async def verify(
        self,
        task_description: str,
        repo_context: str,
        changed_files: List[str],
        acceptance_criteria: str = "",
    ) -> VerificationResult:
        logger.info("Starting deterministic verification", task=task_description[:50])

        changed_files = sorted({p for p in changed_files if p.strip()})
        if not changed_files:
            return VerificationResult(success=False, reason="No changed files were produced.")

        temp_files = [p for p in changed_files if self._is_temp_artifact(p)]
        if temp_files:
            return VerificationResult(
                success=False,
                reason=f"Temporary or scratch artifacts are not allowed: {', '.join(temp_files)}",
            )

        missing = await self._missing_files(changed_files)
        if missing:
            return VerificationResult(success=False, reason=f"Changed files are missing: {', '.join(missing)}")

        schema_result = await self._schema_and_syntax_checks(changed_files)
        if not schema_result.success and not schema_result.skipped:
            return schema_result

        project_result = await self._project_checks(changed_files)
        if not project_result.success and not project_result.skipped:
            return project_result

        evidence = "\n".join(
            part for part in [schema_result.evidence, project_result.evidence] if part
        )
        skipped = [r.reason for r in (schema_result, project_result) if r.skipped]
        if skipped:
            return VerificationResult(success=False, skipped=True, reason="; ".join(skipped), evidence=evidence[:4000])
        return VerificationResult(success=True, reason="Deterministic verification passed.", evidence=evidence[:4000])

    async def verify_release(self, repo_context: str, changed_files: List[str]) -> VerificationResult:
        return await self.verify(
            task_description="Final release-level verification across all changed files.",
            repo_context=repo_context,
            changed_files=changed_files,
            acceptance_criteria="All changed source/config files are valid and project checks pass when available.",
        )

    def _is_temp_artifact(self, rel_path: str) -> bool:
        name = os.path.basename(rel_path)
        return (
            name.startswith(".tmp")
            or name.endswith(".tmp")
            or name.endswith(".bak")
            or ".orig" in name
            or rel_path.endswith(".rej")
        )

    async def _missing_files(self, changed_files: List[str]) -> List[str]:
        missing = []
        for rel_path in changed_files:
            quoted = shlex.quote(rel_path)
            output, _ = await self.sandbox.execute_command(f"[ -f {quoted} ] && echo yes || echo no")
            if output.strip() != "yes":
                missing.append(rel_path)
        return missing

    async def _schema_and_syntax_checks(self, changed_files: List[str]) -> VerificationResult:
        evidence = []
        checked = 0
        ts_projects: Set[Path] = set()

        validators = {
            ".json": JsonValidator(self.sandbox, self.repo_root),
            ".js": JavaScriptValidator(self.sandbox, self.repo_root),
            ".ts": TypeScriptValidator(self.sandbox, self.repo_root),
            ".tsx": TypeScriptValidator(self.sandbox, self.repo_root),
            ".py": PythonValidator(self.sandbox, self.repo_root),
        }

        for rel_path in changed_files:
            suffix = Path(rel_path).suffix
            if suffix in {".ts", ".tsx"}:
                tsconfig = self._nearest_file(rel_path, "tsconfig.json")
                if tsconfig:
                    ts_projects.add(tsconfig.parent)
                    result = VerificationResult(success=True, reason="Deferred to project TypeScript check.")
                else:
                    result = await validators[suffix].validate(rel_path)
            elif suffix in validators:
                result = await validators[suffix].validate(rel_path)
            else:
                result = VerificationResult(success=True, reason="No schema or syntax check required.")

            if suffix in validators:
                checked += 1
            if not result.success:
                return result
            if result.evidence:
                evidence.append(result.evidence)

        for project_dir in sorted(ts_projects):
            result = await self._check_ts_project(project_dir)
            if not result.success:
                return result
            if result.evidence:
                evidence.append(result.evidence)

        if checked == 0:
            return VerificationResult(success=False, skipped=True, reason="No source/config syntax checks were applicable.")

        return VerificationResult(success=True, reason="Schema and syntax checks passed.", evidence="\n".join(evidence))

    async def _project_checks(self, changed_files: List[str]) -> VerificationResult:
        evidence = []
        skipped = []
        package_dirs = self._package_dirs(changed_files)

        for package_dir in sorted(package_dirs):
            node_modules = package_dir / "node_modules"
            package_json = package_dir / "package.json"
            scripts = self._package_scripts(package_json)
            rel_dir = self._rel(package_dir)

            if not node_modules.exists():
                for script in ("lint", "test", "build"):
                    if script in scripts:
                        skipped.append(f"npm run {script} not run in {rel_dir}: node_modules missing")
                continue

            for script in ("lint", "test", "build"):
                if script not in scripts:
                    continue
                cmd = f"npm run {script} --if-present"
                output, code = await self._run_in_docker(cmd, rel_dir)
                if code != 0:
                    return VerificationResult(
                        success=False,
                        reason=f"`npm run {script}` failed in {rel_dir}: {output[:800]}",
                        used_command=cmd,
                        evidence=output[:4000],
                    )
                evidence.append(f"{cmd}\n{output[:1000]}")

        if (self.repo_root / "tests").exists() or (self.repo_root / "pytest.ini").exists():
            cmd = "pip install pytest && pytest"
            output, code = await self._run_in_docker(cmd)
            if code == 5:
                skipped.append("pytest collected no tests")
                evidence.append(f"{cmd}\n{output[:1000]}")
            elif code != 0:
                return VerificationResult(
                    success=False,
                    reason=f"pytest failed: {output[:800]}",
                    used_command=cmd,
                    evidence=output[:4000]
                )
            else:
                evidence.append(f"{cmd}\n{output[:1000]}")
        else:
            skipped.append("pytest not run: no tests directory or pytest.ini")

        if skipped:
            return VerificationResult(success=False, skipped=True, reason="; ".join(skipped), evidence="\n".join(evidence))
        return VerificationResult(success=True, reason="Project checks passed.", evidence="\n".join(evidence))

    async def _check_ts_project(self, project_dir: Path) -> VerificationResult:
        rel_dir = self._rel(project_dir)
        cmd = f"cd {shlex.quote(rel_dir)} && npx tsc --noEmit -p tsconfig.json"
        output, code = await self.sandbox.execute_command(cmd)
        if code != 0:
            return VerificationResult(success=False, reason=f"TypeScript project check failed in {rel_dir}: {output[:800]}", used_command=cmd, evidence=output)
        return VerificationResult(success=True, reason=f"TypeScript project check passed in {rel_dir}", evidence=cmd)

    def _nearest_file(self, rel_path: str, filename: str) -> Optional[Path]:
        current = (self.repo_root / rel_path).parent
        while self.repo_root in [current, *current.parents]:
            candidate = current / filename
            if candidate.exists():
                return candidate
            if current == self.repo_root:
                break
            current = current.parent
        return None

    def _package_dirs(self, changed_files: List[str]) -> Set[Path]:
        dirs: Set[Path] = set()
        for rel_path in changed_files:
            package_json = self._nearest_file(rel_path, "package.json")
            if package_json:
                dirs.add(package_json.parent)
        return dirs

    def _package_scripts(self, package_json: Path) -> Set[str]:
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        scripts = data.get("scripts") or {}
        if not isinstance(scripts, dict):
            return set()
        return set(scripts)

    def _rel(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.repo_root))
