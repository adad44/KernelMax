"""Command-line smoke tests for the offline orchestrator demonstration."""

import contextlib
import io
import unittest

from orchestrator.main import main


class OrchestratorCliTests(unittest.TestCase):
    def test_demo_prints_multiple_outcomes_and_keeps_diagnostic_distinct(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["demo"])
        summary = output.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("Submissions: 4", summary)
        self.assertIn("evaluator attempts: 2/2", summary)
        self.assertIn("duplicate=1", summary)
        self.assertIn("failed=1", summary)
        self.assertIn("budget_exhausted=1", summary)
        self.assertIn("completed_no_verdict=1", summary)
        self.assertIn("diagnostics recorded; no candidate approval", summary)
        self.assertNotIn("explicit evaluator approval", summary)

    def test_demo_accepts_an_explicit_shared_attempt_limit(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["demo", "--max-attempts", "1"])

        self.assertEqual(exit_code, 0)
        self.assertIn("evaluator attempts: 1/1", output.getvalue())
        self.assertIn("budget_exhausted=2", output.getvalue())

    def test_demo_rejects_non_positive_attempt_limit(self):
        with self.assertRaises(SystemExit) as raised:
            main(["demo", "--max-attempts", "0"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
