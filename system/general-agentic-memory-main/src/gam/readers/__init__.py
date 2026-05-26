# -*- coding: utf-8 -*-
"""
GAM Readers Module

文件读取器，用于读取不同格式的输入文件。

Usage:
```python
from gam.readers import PdfReader, TxtReader

# 读取 PDF
pdf_reader = PdfReader()
pdf_text = pdf_reader.read("/path/to/document.pdf")

# 读取 TXT
txt_reader = TxtReader()
txt_text = txt_reader.read("/path/to/document.txt")

# 然后可以传给 TextGAMAgent 处理
from gam import TextGAMAgent, GAMTree
from gam.generators import OpenAIGenerator

tree = GAMTree.create(Path("./my_gam"), name="my_memory")
agent = TextGAMAgent(generator, tree)

# 使用 chunk 方法处理文本
chunk_result = agent.chunk(pdf_text)
```
"""

from .base import BaseReader, PathLike
from .pdf_reader import PdfReader
from .txt_reader import TxtReader

__all__ = [
    "BaseReader",
    "PathLike",
    "PdfReader",
    "TxtReader",
]
