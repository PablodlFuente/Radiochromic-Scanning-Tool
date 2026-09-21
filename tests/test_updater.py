import subprocess
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

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


class UpdateCheckerReleaseTests(unittest.TestCase):
    def test_update_check_uses_published_release_version(self):
        checker = UpdateChecker(current_version="2.0.0")
        checker._request_json = lambda _url: {
            "tag_name": "v2.1.0", "draft": False,
            "html_url": "https://example.invalid/release", "assets": [],
        }

        result = checker.check_for_updates()

        self.assertTrue(result["success"])
        self.assertTrue(result["has_updates"])
        self.assertEqual(result["current_version"], "2.0.0")
        self.assertEqual(result["latest_version"], "2.1.0")

    def test_same_release_is_up_to_date(self):
        checker = UpdateChecker(current_version="2.0.0")
        checker._request_json = lambda _url: {
            "tag_name": "2.0.0", "draft": False, "assets": [],
        }

        self.assertFalse(checker.check_for_updates()["has_updates"])

    def test_installer_asset_selection_requires_the_release_installer(self):
        release = {"assets": [
            {"name": "helper.exe"},
            {"name": "RadiochromicFilmAnalyzer-Setup.exe"},
        ]}

        selected = UpdateChecker._select_installer_asset(release)

        self.assertEqual(selected["name"], "RadiochromicFilmAnalyzer-Setup.exe")

    def test_installer_selection_does_not_accept_an_ambiguous_executable(self):
        release = {"assets": [{"name": "RadiochromicFilmAnalyzer.exe"}]}
        self.assertIsNone(UpdateChecker._select_installer_asset(release))

    def test_installer_download_reports_byte_progress(self):
        payload = b"verified release installer"
        release = {"assets": [{
            "name": "RadiochromicFilmAnalyzer-Setup.exe",
            "browser_download_url": "https://example.invalid/installer",
            "size": len(payload),
        }]}

        class Response(BytesIO):
            headers = {"Content-Length": str(len(payload))}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        updates = []
        checker = UpdateChecker(current_version="2.0.1")
        with patch("app.utils.updater.urllib.request.urlopen", return_value=Response(payload)):
            result = checker.download_release_installer(
                release, progress_callback=lambda downloaded, total: updates.append((downloaded, total))
            )
        try:
            self.assertTrue(result["success"], result)
            self.assertEqual(updates[0], (0, len(payload)))
            self.assertEqual(updates[-1], (len(payload), len(payload)))
        finally:
            if result.get("path"):
                Path(result["path"]).unlink(missing_ok=True)

    def test_startup_update_downloads_and_schedules_a_new_release(self):
        checker = UpdateChecker(current_version="2.1.0")
        checker.check_for_updates = lambda: {
            "success": True, "has_updates": True, "release": {"tag_name": "v2.1.1"}
        }
        checker.download_release_installer = lambda release: {
            "success": True, "path": "C:/temporary/RadiochromicFilmAnalyzer-Setup.exe"
        }
        checker.prepare_installer_update = lambda path: {"success": True, "error": None}

        result = checker.install_latest_published_release()

        self.assertTrue(result["success"])
        self.assertTrue(result["update_started"])

    def test_installer_helper_uses_the_installer_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "RadiochromicFilmAnalyzer.exe"
            installer = root / "RadiochromicFilmAnalyzer-Setup.exe"
            executable.touch()
            installer.touch()
            (root / "unins000.exe").touch()
            checker = UpdateChecker()
            with patch("app.utils.updater.sys.frozen", True, create=True), patch(
                "app.utils.updater.sys.executable", str(executable)
            ), patch("app.utils.updater.subprocess.Popen") as popen:
                result = checker.prepare_installer_update(installer)
        self.assertTrue(result["success"], result)
        self.assertEqual(Path(popen.call_args.kwargs["cwd"]), root)


if __name__ == "__main__":
    unittest.main()
