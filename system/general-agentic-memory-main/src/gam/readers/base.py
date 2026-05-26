# -*- coding: utf-8 -*-
"""
Base Reader - 文件读取器基类

所有文件读取器的抽象基类。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


class BaseReader(ABC):
    """
    文件读取器基类
    
    所有文件读取器都应该继承此类并实现 read 方法。
    """
    
    @abstractmethod
    def read(self, path: PathLike) -> str:
        """
        读取文件内容并返回文本
        
        Args:
            path: 文件路径
            
        Returns:
            文件内容文本
        """
        raise NotImplementedError
    
    @staticmethod
    def _to_path(path: PathLike) -> Path:
        """将路径转换为 Path 对象"""
        return Path(path).expanduser()
