# -*- coding: utf-8 -*-
"""
TXT Reader - 文本文件读取器

用于读取纯文本文件。
"""

from __future__ import annotations

from .base import BaseReader, PathLike


class TxtReader(BaseReader):
    """
    文本文件读取器
    
    读取纯文本文件，支持 UTF-8 和 Latin-1 编码。
    
    Usage:
    ```python
    from gam.readers import TxtReader
    
    reader = TxtReader()
    text = reader.read("/path/to/document.txt")
    print(text)
    ```
    """
    
    def read(self, path: PathLike) -> str:
        """
        读取文本文件内容
        
        Args:
            path: 文本文件路径
            
        Returns:
            文件内容
            
        Raises:
            FileNotFoundError: 文件不存在
        """
        file_path = self._to_path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="latin-1")
