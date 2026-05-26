# -*- coding: utf-8 -*-
"""
PDF Reader - PDF 文件读取器

用于读取 PDF 文件并提取文本内容。
"""

from __future__ import annotations

from .base import BaseReader, PathLike


class PdfReader(BaseReader):
    """
    PDF 文件读取器
    
    使用 PyPDF2 提取 PDF 文件中的文本内容。
    
    Usage:
    ```python
    from gam.readers import PdfReader
    
    reader = PdfReader()
    text = reader.read("/path/to/document.pdf")
    print(text)
    ```
    
    Note:
        需要安装 PyPDF2: pip install PyPDF2
    """
    
    def read(self, path: PathLike) -> str:
        """
        读取 PDF 文件并返回文本内容
        
        Args:
            path: PDF 文件路径
            
        Returns:
            PDF 文件中提取的文本内容
            
        Raises:
            FileNotFoundError: 文件不存在
            ImportError: 未安装 PyPDF2
        """
        file_path = self._to_path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            from PyPDF2 import PdfReader as _PdfReader
        except ImportError as exc:
            raise ImportError(
                "Missing dependency PyPDF2. Install with: pip install PyPDF2"
            ) from exc
        
        reader = _PdfReader(str(file_path))
        chunks = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                chunks.append(text)
        return "\n".join(chunks)
