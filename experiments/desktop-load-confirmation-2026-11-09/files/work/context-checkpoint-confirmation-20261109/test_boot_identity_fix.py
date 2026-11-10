import subprocess
import unittest
from unittest.mock import patch

from evidence import EvidenceError
from runtime_integration import boot_identity


VALID_A = "BAABCDE9-1030-4388-AC21-59096DF4CFDD"
VALID_B = "719F5C6E-5FD9-4E70-8F16-CF2D6A25D2E3"


class BootIdentityFixTests(unittest.TestCase):
    def result(self, value):
        return subprocess.CompletedProcess(
            args=["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
            returncode=0, stdout=value, stderr="",
        )

    def test_real_boot_identity_matches_normalized_session_uuid(self):
        observed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        expected = f"boot-session:{observed.lower()}"
        self.assertEqual(boot_identity(), expected)

    def test_wall_boot_time_changes_do_not_change_session_identity(self):
        with patch("runtime_integration.subprocess.run", side_effect=[
            self.result(VALID_A), self.result(VALID_A),
        ]) as run:
            identity_before = boot_identity()
            # A changed wall-clock boot-time representation is irrelevant: the
            # helper reads only the stable boot-session UUID.
            wall_clock_boot_times = [
                "{ sec = 1791090735, usec = 663152 }",
                "{ sec = 1791090735, usec = 586177 }",
            ]
            identity_after = boot_identity()
        self.assertNotEqual(wall_clock_boot_times[0], wall_clock_boot_times[1])
        self.assertEqual(identity_before, identity_after)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0], [
            "/usr/sbin/sysctl", "-n", "kern.bootsessionuuid",
        ])

    def test_changed_session_uuid_changes_identity(self):
        with patch("runtime_integration.subprocess.run", side_effect=[
            self.result(VALID_A), self.result(VALID_B),
        ]):
            self.assertNotEqual(boot_identity(), boot_identity())

    def test_malformed_missing_zero_and_sysctl_failures_fail_closed(self):
        for raw in ("", "not-a-uuid", "00000000-0000-0000-0000-000000000000"):
            with self.subTest(raw=raw), patch(
                "runtime_integration.subprocess.run", return_value=self.result(raw),
            ), self.assertRaises(EvidenceError):
                boot_identity()
        with patch("runtime_integration.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaises(EvidenceError):
                boot_identity()
        with patch(
            "runtime_integration.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "sysctl"),
        ):
            with self.assertRaises(EvidenceError):
                boot_identity()
        with patch("runtime_integration.subprocess.run", side_effect=subprocess.TimeoutExpired("sysctl", 5)):
            with self.assertRaises(EvidenceError):
                boot_identity()


if __name__ == "__main__":
    unittest.main()
