import subprocess
import tempfile
import unittest
from pathlib import Path

from app.utils.updater import UpdateChecker


class UpdateCheckerGitTests(unittest.TestCase):
    def run_git(self, directory, *args):
        result = subprocess.run(
            ["git", *args], cwd=directory, text=True, capture_output=True, check=True
        )
        return result.stdout.strip()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.origin = root / "origin.git"
        self.seed = root / "seed"
        self.local = root / "local"
        self.run_git(root, "init", "--bare", "--initial-branch=master", str(self.origin))
        self.run_git(root, "clone", str(self.origin), str(self.seed))
        for repository in (self.seed,):
            self.run_git(repository, "config", "user.email", "tests@example.invalid")
            self.run_git(repository, "config", "user.name", "Updater Tests")
        (self.seed / "tracked.txt").write_text("base\n", encoding="utf-8")
        self.run_git(self.seed, "add", "tracked.txt")
        self.run_git(self.seed, "commit", "-m", "initial")
        self.run_git(self.seed, "push", "origin", "master")
        self.run_git(root, "clone", str(self.origin), str(self.local))
        self.run_git(self.local, "config", "user.email", "tests@example.invalid")
        self.run_git(self.local, "config", "user.name", "Updater Tests")

    def tearDown(self):
        self.temporary.cleanup()

    def push_remote_change(self, filename="remote.txt"):
        path = self.seed / filename
        path.write_text("remote update\n", encoding="utf-8")
        self.run_git(self.seed, "add", filename)
        self.run_git(self.seed, "commit", "-m", f"update {filename}")
        self.run_git(self.seed, "push", "origin", "master")

    def test_pull_restores_tracked_and_untracked_local_changes(self):
        (self.local / "tracked.txt").write_text("local edit\n", encoding="utf-8")
        (self.local / "untracked.txt").write_text("local untracked\n", encoding="utf-8")
        self.push_remote_change()

        result = UpdateChecker(str(self.local)).pull_updates()

        self.assertTrue(result["success"], result)
        self.assertEqual((self.local / "tracked.txt").read_text(encoding="utf-8"), "local edit\n")
        self.assertEqual((self.local / "untracked.txt").read_text(encoding="utf-8"), "local untracked\n")
        self.assertTrue((self.local / "remote.txt").exists())
        self.assertEqual(self.run_git(self.local, "stash", "list"), "")

    def test_non_fast_forward_update_fails_without_losing_dirty_work(self):
        (self.local / "local-commit.txt").write_text("local commit\n", encoding="utf-8")
        self.run_git(self.local, "add", "local-commit.txt")
        self.run_git(self.local, "commit", "-m", "local divergence")
        (self.local / "tracked.txt").write_text("dirty edit\n", encoding="utf-8")
        self.push_remote_change("remote-divergence.txt")

        result = UpdateChecker(str(self.local)).pull_updates()

        self.assertFalse(result["success"])
        self.assertFalse(result["updated"])
        self.assertEqual((self.local / "tracked.txt").read_text(encoding="utf-8"), "dirty edit\n")
        self.assertTrue((self.local / "local-commit.txt").exists())
        self.assertFalse((self.local / "remote-divergence.txt").exists())
        self.assertEqual(self.run_git(self.local, "stash", "list"), "")


if __name__ == "__main__":
    unittest.main()
