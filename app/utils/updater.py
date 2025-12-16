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

logger = logging.getLogger(__name__)

# Get application root directory
APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class UpdateChecker:
    """Handles checking and applying updates from GitHub."""
    
    def __init__(self, app_dir=None):
        """Initialize the update checker.
        
        Args:
            app_dir: The application root directory. Defaults to auto-detected.
        """
        self.app_dir = app_dir or APP_DIR
        self.local_commit = None
        self.remote_commit = None
        self.commits_behind = 0
        self.has_updates = False
        self.error = None
    
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
        """Check for available updates.
        
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
        
        # Check prerequisites
        if not self.is_git_available():
            return {
                'success': False,
                'has_updates': False,
                'error': "Git is not installed or not in PATH"
            }
        
        if not self.is_git_repository():
            return {
                'success': False,
                'has_updates': False,
                'error': "Application directory is not a git repository"
            }
        
        # Fetch latest from remote
        if not self.fetch_updates():
            return {
                'success': False,
                'has_updates': False,
                'error': self.error or "Failed to fetch updates"
            }
        
        # Get commit information
        self.get_local_commit()
        self.get_remote_commit(branch)
        self.count_commits_behind(branch)
        
        return {
            'success': True,
            'has_updates': self.has_updates,
            'local_commit': self.local_commit[:8] if self.local_commit else None,
            'remote_commit': self.remote_commit[:8] if self.remote_commit else None,
            'commits_behind': self.commits_behind,
            'error': None
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
    
    def stash_changes(self) -> bool:
        """Stash any local changes."""
        try:
            result = subprocess.run(
                ["git", "stash"],
                capture_output=True, text=True, cwd=self.app_dir
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error stashing changes: {e}")
            return False
    
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
        stashed = False
        
        try:
            # Stash local changes if any
            if self.has_local_changes():
                if self.stash_changes():
                    stashed = True
                    logger.info("Stashed local changes before update")
            
            # Pull the latest changes
            result = subprocess.run(
                ["git", "pull", "origin", branch],
                capture_output=True, text=True, cwd=self.app_dir, timeout=60
            )
            
            if result.returncode == 0:
                logger.info("Update pulled successfully")
                return {
                    'success': True,
                    'stashed': stashed,
                    'error': None
                }
            else:
                error = result.stderr or "Unknown error during pull"
                logger.error(f"Pull failed: {error}")
                return {
                    'success': False,
                    'stashed': stashed,
                    'error': error
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stashed': stashed,
                'error': "Update timed out"
            }
        except Exception as e:
            logger.error(f"Error pulling updates: {e}")
            return {
                'success': False,
                'stashed': stashed,
                'error': str(e)
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
