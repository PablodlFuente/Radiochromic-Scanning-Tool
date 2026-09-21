"""
Updater module for checking and applying updates from GitHub.

This module handles all update-related functionality including:
- Checking for available updates
- Downloading and applying updates via git
- Restarting the application after updates
"""

import os
import sys
import subprocess
import logging
import hashlib
import json
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from app.version import __version__
from app.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Get application root directory
APP_DIR = os.fspath(PROJECT_ROOT)
GITHUB_REPOSITORY = "PablodlFuente/Radiochromic-Scanning-Tool"
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/latest"
RELEASE_INSTALLER_NAME = "RadiochromicFilmAnalyzer-Setup.exe"


class UpdateChecker:
    """Handles checking and applying updates from GitHub."""
    
    def __init__(self, app_dir=None, current_version=None, release_api=None):
        """Initialize the update checker.
        
        Args:
            app_dir: The application root directory. Defaults to auto-detected.
        """
        self.app_dir = app_dir or APP_DIR
        self.current_version = current_version or __version__
        self.release_api = release_api or LATEST_RELEASE_API
        self.local_commit = None
        self.remote_commit = None
        self.commits_behind = 0
        self.has_updates = False
        self.error = None

    @staticmethod
    def _version_key(value):
        """Return a comparable numeric key for conventional release tags."""
        text = str(value or "").strip().lower().lstrip("v")
        core = text.split("-", 1)[0].split("+", 1)[0]
        try:
            parts = tuple(int(part) for part in core.split("."))
        except ValueError:
            return ()
        return parts + (0,) * (3 - len(parts))

    def _request_json(self, url):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": f"RadiochromicFilmAnalyzer/{self.current_version}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)

    def get_latest_release(self):
        """Read the latest published release; commits are intentionally ignored."""
        try:
            release = self._request_json(self.release_api)
            if release.get("draft"):
                raise ValueError("GitHub returned a draft instead of a published release")
            tag = str(release.get("tag_name", "")).strip()
            if not self._version_key(tag):
                raise ValueError("The latest release has no valid version tag")
            return release, None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None, "No published release is available"
            return None, f"GitHub release request failed (HTTP {exc.code})"
        except (urllib.error.URLError, TimeoutError) as exc:
            return None, f"Could not contact GitHub Releases: {exc}"
        except Exception as exc:
            logger.error("Could not read the latest release", exc_info=True)
            return None, str(exc)

    @staticmethod
    def _select_installer_asset(release):
        """Return the uniquely named installer from a published release."""
        assets = release.get("assets", [])
        installers = [
            asset for asset in assets
            if str(asset.get("name", "")).lower() == RELEASE_INSTALLER_NAME.lower()
        ]
        return installers[0] if len(installers) == 1 else None

    def download_release_installer(self, release):
        """Download and verify the installer asset from a published release."""
        asset = self._select_installer_asset(release)
        if asset is None:
            return {"success": False, "error": "The release has no Windows installer asset"}
        url = asset.get("browser_download_url")
        expected_size = int(asset.get("size") or 0)
        if not url:
            return {"success": False, "error": "The executable asset has no download URL"}
        destination_dir = Path(tempfile.gettempdir()) / "RadiochromicFilmAnalyzerUpdate"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / str(asset["name"])
        request = urllib.request.Request(
            url, headers={"User-Agent": f"RadiochromicFilmAnalyzer/{self.current_version}"}
        )
        try:
            digest = hashlib.sha256()
            byte_count = 0
            with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
                    byte_count += len(block)
            if expected_size and byte_count != expected_size:
                destination.unlink(missing_ok=True)
                return {"success": False, "error": "Downloaded installer size does not match the release asset"}
            declared_digest = str(asset.get("digest") or "")
            if declared_digest.startswith("sha256:") and digest.hexdigest() != declared_digest[7:]:
                destination.unlink(missing_ok=True)
                return {"success": False, "error": "Downloaded installer failed SHA-256 verification"}
            return {"success": True, "path": str(destination), "asset": asset, "error": None}
        except Exception as exc:
            destination.unlink(missing_ok=True)
            logger.error("Release download failed", exc_info=True)
            return {"success": False, "error": str(exc)}

    def prepare_installer_update(self, downloaded_path):
        """Run a release installer after the packaged application exits.

        Inno Setup recognizes the existing application ID and installs into the
        existing directory.  It replaces program files while preserving the
        user-owned calibration, log, plugin and configuration directories.
        """
        if not getattr(sys, "frozen", False):
            return {
                "success": False,
                "error": "Automatic installation is only available in the packaged application",
            }
        current_executable = Path(sys.executable).resolve()
        if not (current_executable.parent / "unins000.exe").is_file():
            return {
                "success": False,
                "error": (
                    "Automatic updates require an installed application. "
                    "Run the release installer manually to migrate this portable copy."
                ),
            }
        installer = Path(downloaded_path).resolve()
        helper = installer.parent / "apply_radiochromic_update.cmd"
        helper.write_text(
            "@echo off\n"
            "setlocal\n"
            f"set \"TARGET={current_executable}\"\n"
            f"set \"INSTALLER={installer}\"\n"
            f"set \"APP_PID={os.getpid()}\"\n"
            ":wait_for_exit\n"
            "tasklist /FI \"PID eq %APP_PID%\" 2>NUL | find \"%APP_PID%\" >NUL\n"
            "if not errorlevel 1 (timeout /t 1 /nobreak >NUL & goto wait_for_exit)\n"
            "start \"\" /wait \"%INSTALLER%\" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART\n"
            "if errorlevel 1 exit /b 1\n"
            "start \"\" \"%TARGET%\"\n"
            "del \"%INSTALLER%\"\n"
            "del \"%~f0\"\n",
            encoding="utf-8",
        )
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            ["cmd.exe", "/c", str(helper)],
            cwd=str(downloaded.parent),
            creationflags=creation_flags,
        )
        return {"success": True, "error": None}

    # Compatibility aliases retained for integrations using the previous API.
    _select_executable_asset = _select_installer_asset
    download_release_executable = download_release_installer
    prepare_executable_replacement = prepare_installer_update
    
    def is_git_available(self) -> bool:
        """Check if git is installed and available."""
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True, text=True, cwd=self.app_dir, timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        except Exception as e:
            logger.error(f"Error checking git availability: {e}")
            return False
    
    def is_git_repository(self) -> bool:
        """Check if the app directory is a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True, text=True, cwd=self.app_dir, timeout=10
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except Exception:
            return False
    
    def fetch_updates(self) -> bool:
        """Fetch latest changes from remote repository.
        
        Returns:
            True if fetch was successful, False otherwise.
        """
        try:
            result = subprocess.run(
                ["git", "fetch", "origin"],
                capture_output=True, text=True, cwd=self.app_dir, timeout=30
            )
            if result.returncode != 0:
                self.error = f"Fetch failed: {result.stderr}"
                logger.error(self.error)
                return False
            return True
        except subprocess.TimeoutExpired:
            self.error = "Fetch timed out"
            logger.error(self.error)
            return False
        except Exception as e:
            self.error = str(e)
            logger.error(f"Error fetching updates: {e}")
            return False
    
    def get_local_commit(self) -> str:
        """Get the current local commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=self.app_dir
            )
            if result.returncode == 0:
                self.local_commit = result.stdout.strip()
                return self.local_commit
        except Exception as e:
            logger.error(f"Error getting local commit: {e}")
        return None

    def get_current_branch(self) -> str:
        """Get the current branch name, or HEAD when detached."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=self.app_dir
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.error(f"Error getting current branch: {e}")
        return None
    
    def get_remote_commit(self, branch="master") -> str:
        """Get the latest remote commit hash.
        
        Args:
            branch: The remote branch to check. Defaults to "master".
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", f"origin/{branch}"],
                capture_output=True, text=True, cwd=self.app_dir
            )
            if result.returncode == 0:
                self.remote_commit = result.stdout.strip()
                return self.remote_commit
        except Exception as e:
            logger.error(f"Error getting remote commit: {e}")
        return None
    
    def count_commits_behind(self, branch="master") -> int:
        """Count how many commits behind the local branch is.
        
        Args:
            branch: The remote branch to compare against.
            
        Returns:
            Number of commits behind, or 0 if up to date or error.
        """
        try:
            result = subprocess.run(
                ["git", "rev-list", "--count", f"HEAD..origin/{branch}"],
                capture_output=True, text=True, cwd=self.app_dir
            )
            if result.returncode == 0:
                self.commits_behind = int(result.stdout.strip())
                self.has_updates = self.commits_behind > 0
                return self.commits_behind
        except Exception as e:
            logger.error(f"Error counting commits: {e}")
        return 0
    
    def check_for_updates(self, branch="master") -> dict:
        """Check the latest published GitHub Release, never branch commits.
        
        Args:
            branch: The remote branch to check against.
            
        Returns:
            Dictionary with update information:
            {
                'success': bool,
                'has_updates': bool,
                'local_commit': str,
                'remote_commit': str,
                'commits_behind': int,
                'error': str or None
            }
        """
        self.error = None
        release, error = self.get_latest_release()
        if release is None:
            return {"success": False, "has_updates": False, "error": error}
        latest_version = str(release["tag_name"]).lstrip("v")
        has_updates = self._version_key(latest_version) > self._version_key(self.current_version)
        return {
            "success": True,
            "has_updates": has_updates,
            "current_version": self.current_version,
            "latest_version": latest_version,
            "release_url": release.get("html_url", ""),
            "release": release,
            "error": None,
        }
    
    def has_local_changes(self) -> bool:
        """Check if there are uncommitted local changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=self.app_dir
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def list_remote_versions(self, branch="master", max_count=50) -> dict:
        """List remote commits for a branch, including date and subject."""
        if not self.is_git_available():
            return {
                'success': False,
                'versions': [],
                'error': "Git is not installed or not in PATH"
            }

        if not self.is_git_repository():
            return {
                'success': False,
                'versions': [],
                'error': "Application directory is not a git repository"
            }

        if not self.fetch_updates():
            return {
                'success': False,
                'versions': [],
                'error': self.error or "Failed to fetch updates"
            }

        current_commit = self.get_local_commit()

        try:
            result = subprocess.run(
                [
                    "git", "log", f"origin/{branch}",
                    f"-n{max_count}",
                    "--date=short",
                    "--pretty=format:%H%x09%ad%x09%s",
                ],
                capture_output=True, text=True, cwd=self.app_dir, timeout=30
            )
            if result.returncode != 0:
                error = result.stderr or "Failed to read remote version history"
                logger.error(error)
                return {
                    'success': False,
                    'versions': [],
                    'error': error,
                }

            versions = []
            for line in result.stdout.splitlines():
                parts = line.split("\t", 2)
                if len(parts) != 3:
                    continue
                commit_hash, commit_date, subject = parts
                versions.append({
                    'commit': commit_hash,
                    'short_commit': commit_hash[:8],
                    'date': commit_date,
                    'subject': subject,
                    'is_current': current_commit == commit_hash,
                })

            return {
                'success': True,
                'versions': versions,
                'current_commit': current_commit[:8] if current_commit else None,
                'error': None,
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'versions': [],
                'error': "Reading remote version history timed out",
            }
        except Exception as e:
            logger.error(f"Error listing remote versions: {e}")
            return {
                'success': False,
                'versions': [],
                'error': str(e),
            }
    
    def _stash_local_changes(self):
        """Stash tracked and untracked files and return the exact stash ref/hash."""
        try:
            result = subprocess.run(
                ["git", "stash", "push", "--include-untracked", "-m", "radiochromic-auto-update"],
                capture_output=True, text=True, cwd=self.app_dir
            )
            if result.returncode != 0:
                return None, result.stderr or "Could not stash local changes"
            ref = "stash@{0}"
            revision = subprocess.run(
                ["git", "rev-parse", ref], capture_output=True, text=True, cwd=self.app_dir
            )
            if revision.returncode != 0:
                return None, revision.stderr or "Could not identify the created stash"
            return (ref, revision.stdout.strip()), None
        except Exception as e:
            logger.error(f"Error stashing changes: {e}")
            return None, str(e)

    def stash_changes(self) -> bool:
        """Backward-compatible wrapper for explicitly stashing local changes."""
        stash, error = self._stash_local_changes()
        return stash is not None and error is None

    def _restore_stash(self, stash):
        """Restore only the stash created by this operation; retain it on conflict."""
        if stash is None:
            return None
        ref, expected_revision = stash
        current = subprocess.run(
            ["git", "rev-parse", ref], capture_output=True, text=True, cwd=self.app_dir
        )
        if current.returncode != 0 or current.stdout.strip() != expected_revision:
            return "The update stash changed unexpectedly and was not applied"
        applied = subprocess.run(
            ["git", "stash", "apply", "--index", ref],
            capture_output=True, text=True, cwd=self.app_dir, timeout=60,
        )
        if applied.returncode != 0:
            return (
                "Local changes could not be restored automatically. "
                f"They remain saved in {ref}. {applied.stderr.strip()}"
            )
        dropped = subprocess.run(
            ["git", "stash", "drop", ref], capture_output=True, text=True, cwd=self.app_dir
        )
        if dropped.returncode != 0:
            return f"Local changes were restored, but {ref} could not be removed"
        return None
    
    def pull_updates(self, branch="master") -> dict:
        """Pull the latest updates from remote.
        
        Args:
            branch: The remote branch to pull from.
            
        Returns:
            Dictionary with result information:
            {
                'success': bool,
                'stashed': bool,
                'error': str or None
            }
        """
        stash = None
        original_branch = self.get_current_branch()
        updated = False
        operation_error = None
        try:
            if self.has_local_changes():
                stash, operation_error = self._stash_local_changes()
                if operation_error:
                    return {'success': False, 'stashed': False, 'error': operation_error}

            if original_branch != branch:
                branch_result = subprocess.run(
                    ["git", "checkout", branch],
                    capture_output=True, text=True, cwd=self.app_dir, timeout=30
                )
                if branch_result.returncode != 0:
                    error = branch_result.stderr or f"Could not switch to branch '{branch}'"
                    logger.error(error)
                    operation_error = error

            if operation_error is None:
                result = subprocess.run(
                    ["git", "pull", "--ff-only", "origin", branch],
                    capture_output=True, text=True, cwd=self.app_dir, timeout=60
                )
                if result.returncode == 0:
                    updated = True
                    logger.info("Update pulled successfully")
                else:
                    operation_error = result.stderr or "Unknown error during fast-forward pull"
        except subprocess.TimeoutExpired:
            operation_error = "Update timed out"
        except Exception as e:
            logger.error(f"Error pulling updates: {e}")
            operation_error = str(e)
        finally:
            if operation_error and original_branch and self.get_current_branch() != original_branch:
                subprocess.run(
                    ["git", "checkout", original_branch],
                    capture_output=True, text=True, cwd=self.app_dir, timeout=30,
                )
            restore_error = self._restore_stash(stash)
            if restore_error:
                operation_error = f"{operation_error + '. ' if operation_error else ''}{restore_error}"

        return {
            'success': updated and operation_error is None,
            'updated': updated,
            'stashed': stash is not None,
            'error': operation_error,
        }

    def checkout_version(self, commit_hash: str) -> dict:
        """Checkout a specific commit in detached HEAD mode."""
        stash = None
        operation_error = None
        checked_out = False

        try:
            if self.has_local_changes():
                stash, operation_error = self._stash_local_changes()
                if operation_error:
                    return {'success': False, 'stashed': False, 'error': operation_error}

            verify_result = subprocess.run(
                ["git", "rev-parse", "--verify", commit_hash],
                capture_output=True, text=True, cwd=self.app_dir, timeout=15
            )
            if verify_result.returncode != 0:
                operation_error = verify_result.stderr or f"Commit '{commit_hash}' not found"
                logger.error(operation_error)

            if operation_error is None:
                checkout_result = subprocess.run(
                    ["git", "checkout", "--detach", commit_hash],
                    capture_output=True, text=True, cwd=self.app_dir, timeout=60
                )
                if checkout_result.returncode != 0:
                    operation_error = checkout_result.stderr or f"Could not checkout commit '{commit_hash}'"
                    logger.error(operation_error)
                else:
                    checked_out = True
                    logger.info("Checked out version %s", commit_hash)
        except subprocess.TimeoutExpired:
            operation_error = "Version change timed out"
        except Exception as e:
            logger.error(f"Error checking out version {commit_hash}: {e}")
            operation_error = str(e)
        finally:
            restore_error = self._restore_stash(stash)
            if restore_error:
                operation_error = f"{operation_error + '. ' if operation_error else ''}{restore_error}"

        return {
            'success': checked_out and operation_error is None,
            'stashed': stash is not None,
            'error': operation_error,
        }
    
    @staticmethod
    def restart_application():
        """Restart the application.
        
        This starts a new instance and exits the current one.
        """
        main_script = os.path.join(APP_DIR, "main.py")
        
        try:
            # Start a new instance
            subprocess.Popen([sys.executable, main_script], cwd=APP_DIR)
            logger.info("Started new application instance")
            
            # Exit current instance
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error restarting application: {e}")
            raise


# Convenience function for simple usage
def check_updates() -> dict:
    """Convenience function to check for updates.
    
    Returns:
        Dictionary with update information.
    """
    checker = UpdateChecker()
    return checker.check_for_updates()
