import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional

# Default timeout for commands
CMD_TIMEOUT = 120

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class BaseWorkspace(ABC):
    """
    Abstract base class for agent workspace.
    Defines the standard interface for executing commands and managing files.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(self.__class__.__name__)

    @abstractmethod
    def run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        workdir: str = None,
    ) -> Tuple[str, str]:
        """
        Execute a command.

        Returns:
            Tuple of (output, exit_code_or_error).
        """
        pass

    @abstractmethod
    def demux_run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        workdir: str = None,
    ) -> Tuple[str, str, str]:
        """
        Execute a command with separate stdout and stderr.

        Returns:
            Tuple of (stdout, stderr, exit_code_or_error).
        """
        pass

    @abstractmethod
    def copy_to_workspace(self, src_path: str, dest_path: str):
        """Copy a file to the workspace."""
        pass

    @abstractmethod
    def copy_dir_to_workspace(self, src_dir: str, dest_dir: str):
        """Copy a directory to the workspace."""
        pass

    @abstractmethod
    def copy_from_workspace(self, container_path: str, local_path: str):
        """Copy files from the workspace to local path."""
        pass

    @abstractmethod
    def close(self):
        """Cleanup resources."""
        pass