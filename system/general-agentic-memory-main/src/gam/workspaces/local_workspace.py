from typing import Tuple, Optional, Union
from pathlib import Path
from .base import BaseWorkspace, CMD_TIMEOUT
from datetime import datetime
import json
import shutil
import re
import subprocess

class LocalWorkspace(BaseWorkspace):
    """
    Local workspace for executing commands on the local machine.
    WARNING: This executes code directly on your machine. Use with caution.
    """

    def __init__(
        self,
        root_path: Union[str, Path] = None,
        logger=None,
        name: str = "local_workspace",
        description: str = "Local agent workspace for long context task",
        **kwargs,
    ):
        super().__init__(logger)
        
        # Use provided root_path or default to current working directory
        if root_path:
            self.root_path = Path(root_path).resolve()
        else:
            self.root_path = Path.cwd()
            
        # Initialize the agent workspace
        self.setup_workspace(name, description)
        self.logger.info(f"Local agent workspace initialized")
        self.logger.info(f"Working directory: {self.root_path}")

    def setup_workspace(self, name: str, description: str):
        """Setup the local agent workspace."""
        # Ensure working directory exists
        if not self.root_path.exists():
            self.root_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created working directory: {self.root_path}")
        
        # Write gam meta data file
        meta_file = self.root_path / ".gam_meta.json"
        meta_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "name": name,
            "description": description
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)


    def run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        workdir: Union[str, Path] = None,
    ) -> Tuple[str, str]:
        """
        Execute a command locally.

        Returns:
            Tuple of (output, exit_code_or_error).
        """
        exec_workdir = self.root_path if workdir is None else Path(workdir).resolve()
        
        if not exec_workdir.exists():
            return f"Error: Working directory {exec_workdir} does not exist", "-1"

        try:
            # Execute command with timeout
            # shell=True allows using pipes, redirects, etc.
            result = subprocess.run(
                code,
                shell=True,
                cwd=exec_workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr into stdout
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )
            
            output = result.stdout
            exit_code = result.returncode
            
            # Remove ANSI escape codes
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)

            if exit_code != 0:
                return output, f"Error: Exit code {exit_code}"

            return output, str(exit_code)

        except subprocess.TimeoutExpired:
            return f"The command took too long to execute (>{timeout}s)", "-1"
        except Exception as e:
            return f"Error: {repr(e)}", "-1"

    def demux_run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        workdir: Union[str, Path] = None,
    ) -> Tuple[str, str, str]:
        """
        Execute a command locally with separate stdout and stderr.

        Returns:
            Tuple of (stdout, stderr, exit_code_or_error).
        """
        exec_workdir = self.root_path if workdir is None else Path(workdir).resolve()
        
        if not exec_workdir.exists():
            return "", f"Error: Working directory {exec_workdir} does not exist", "-1"

        try:
            result = subprocess.run(
                code,
                shell=True,
                cwd=exec_workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )
            
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode

            # Remove ANSI escape codes
            stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
            stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", stderr)
            
            if exit_code != 0:
                return stdout, stderr, f"Error: Exit code {exit_code}"

            return stdout, stderr, str(exit_code)

        except subprocess.TimeoutExpired:
            return f"The command took too long to execute (>{timeout}s)", "", "-1"
        except Exception as e:
            error_msg = f"Error: {repr(e)}"
            return error_msg, error_msg, "-1"

    def copy_to_workspace(self, src_path: Union[str, Path], dest_path: Union[str, Path]):
        """Copy a file to the gam agent workspace."""
        try:
            src = Path(src_path)
            dest = Path(dest_path)
            
            # Create destination directory if it doesn't exist
            if not dest.parent.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                
            shutil.copy2(src, dest)
            self.logger.info(f"Copied {src} to {dest}")
        except Exception as e:
            self.logger.error(f"Error copying file: {repr(e)}")
            raise

    def copy_dir_to_workspace(self, src_dir: Union[str, Path], dest_dir: Union[str, Path]):
        """Copy a directory to the gam agent workspace."""
        try:
            src = Path(src_dir)
            dest = Path(dest_dir)
            
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            self.logger.info(f"Copied directory {src} to {dest}")
        except Exception as e:
            self.logger.error(f"Error copying directory: {repr(e)}")
            raise

    def copy_from_workspace(self, workspace_path: Union[str, Path], local_path: Union[str, Path]):
        """Copy files from workspace to local path."""
        try:
            src = Path(workspace_path)
            dest = Path(local_path)
            
            # Ensure local directory exists
            if not dest.parent.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)

            if src.is_dir():
                 if dest.exists():
                     shutil.rmtree(dest)
                 shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
            
            self.logger.info(f"Copied {src} to {dest}")
        except Exception as e:
            self.logger.error(f"Error copying from gam: {repr(e)}")
            raise

    def close(self):
        """Cleanup resources."""
        self.logger.info("LocalGAM closed")