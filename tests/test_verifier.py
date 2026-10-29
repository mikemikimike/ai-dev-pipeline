import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiosqlite

from agents.models import VerificationResult
from orchestrator.pipeline import ReleaseArcOrchestrator
from orchestrator.verifier import VerifierEngine


class VerifierTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        sandbox = type("Sandbox", (), {"target_repo_path": str(self.root)})()
        sandbox.execute_command = AsyncMock(return_value=("yes", 0))
        self.engine = VerifierEngine(sandbox)
        self.docker = AsyncMock(return_value=("no tests collected", 5))
        docker_patch = patch.object(self.engine, "_run_in_docker", self.docker)
        docker_patch.start()
        self.addCleanup(docker_patch.stop)

    async def test_empty_pytest_suite_does_not_pass(self):
        (self.root / "tests").mkdir()
        result = await self.engine._project_checks(["app.py"])
        self.assertFalse(result.success)
        self.assertIn("pytest", result.reason.lower())

    async def test_real_empty_pytest_collection_is_skipped(self):
        (self.root / "tests").mkdir()

        async def run_pytest(_cmd: str, _rel_dir: str = "") -> tuple[str, int]:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "tests"], cwd=self.root,
                capture_output=True, text=True, check=False,
            )
            return result.stdout + result.stderr, result.returncode

        self.docker.side_effect = run_pytest
        result = await self.engine._project_checks(["app.py"])
        self.assertFalse(result.success)
        self.assertTrue(result.skipped)
        self.assertIn("collected no tests", result.reason)
        self.assertIn("pytest", result.evidence)
        self.assertIn("no tests ran", result.evidence)

    async def test_missing_pytest_records_skip(self):
        result = await self.engine._project_checks(["app.py"])
        self.assertTrue(result.skipped)
        self.assertIn("pytest", result.reason.lower())

    async def test_missing_node_modules_records_skip(self):
        (self.root / "package.json").write_text(json.dumps({"scripts": {"test": "jest"}}))
        result = await self.engine._project_checks(["app.js"])
        self.assertTrue(result.skipped)
        self.assertIn("node_modules", result.reason)

    async def test_no_syntax_checks_records_skip(self):
        result = await self.engine._schema_and_syntax_checks(["notes.md"])
        self.assertTrue(result.skipped)

    async def test_release_never_passes_when_checks_skipped(self):
        result = await self.engine.verify_release("", ["notes.md"])
        self.assertFalse(result.success)
        self.assertTrue(result.skipped)

    async def test_passing_pytest_and_skipped_npm_preserve_evidence(self):
        (self.root / "tests").mkdir()
        (self.root / "package.json").write_text(json.dumps({"scripts": {"test": "jest"}}))
        self.docker.return_value = ("one passed", 0)
        result = await self.engine._project_checks(["app.js"])
        self.assertFalse(result.success)
        self.assertTrue(result.skipped)
        self.assertIn("node_modules", result.reason)
        self.assertIn("one passed", result.evidence)

    async def test_complete_checks_can_pass(self):
        (self.root / "tests").mkdir()
        self.docker.return_value = ("one passed", 0)
        result = await self.engine.verify("task", "", ["app.py"])
        self.assertTrue(result.success)
        self.assertFalse(result.skipped)
        self.assertIn("one passed", result.evidence)

    async def test_pytest_failure_is_not_marked_skipped(self):
        (self.root / "tests").mkdir()
        self.docker.return_value = ("assertion failed", 1)
        result = await self.engine._project_checks(["app.py"])
        self.assertFalse(result.success)
        self.assertFalse(result.skipped)
        self.assertIn("assertion failed", result.reason)

    async def test_ledger_records_skipped_status(self):
        async with aiosqlite.connect(":memory:") as db:
            await db.execute("""CREATE TABLE verification_runs
                (arc_id INTEGER, task_id TEXT, status TEXT, reason TEXT, used_command TEXT,
                 evidence TEXT, changed_files TEXT, attempt_number INTEGER)""")
            result = VerificationResult(success=False, skipped=True, reason="pytest not run")
            orchestrator = ReleaseArcOrchestrator(str(self.root))
            await orchestrator._record_verification(db, 1, "task", result, 1, ["app.py"])
            async with db.execute("SELECT status FROM verification_runs") as cursor:
                row = await cursor.fetchone()
            self.assertEqual(row[0], "skipped")
