# -*- coding: utf-8 -*-
"""
Chunk Schemas

Data schemas for chunking and organization operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field


# ============== Chunking Schemas ==============

@dataclass
class MemorizedChunk:
    """记忆化的 chunk，包含 title、memory、tldr 和原始内容"""
    index: int  # chunk 索引
    title: str  # 生成的标题
    memory: str  # 记忆/摘要
    tldr: str  # 太长不看 (Too Long; Didn't Read) 摘要
    content: str  # 原始内容
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    def to_markdown(self) -> str:
        """转换为 markdown 格式"""
        return f"""# {self.title.replace('_', ' ').title()}

## Chunk Index
{self.index}

## TLDR

{self.tldr}

## Memory

{self.memory}

---

## Original Content

{self.content}
"""

    def save(self, output_dir: Path, filename: Optional[str] = None) -> Path:
        """
        保存到文件
        
        Args:
            output_dir: 输出目录
            filename: 文件名（可选，默认为 chunk_XXX.md）
        
        Returns:
            保存的文件路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"chunk_{self.index:03d}.md"
        
        filepath = output_dir / filename
        filepath.write_text(self.to_markdown(), encoding='utf-8')
        return filepath


@dataclass
class ChunkResult:
    """切分结果"""
    chunks: List[str]  # 切分后的文本块
    metadata: List[Dict[str, Any]]  # 每个块的元信息
    total_tokens: int  # 总 token 数
    num_chunks: int  # 切分数量


@dataclass
class ChunkWithMemoryResult:
    """带 memory 的切分结果"""
    memorized_chunks: List[MemorizedChunk]  # 记忆化的 chunks
    total_tokens: int  # 总 token 数
    num_chunks: int  # 切分数量
    output_dir: Optional[Path] = None  # 输出目录（如果已保存）


# ============== Organization Schemas ==============

@dataclass
class BatchMemorizedChunk:
    """用于批量组织的 memorized chunk"""
    index: int
    title: str
    tldr: str
    memory: str
    original_content: str


@dataclass
class DirectoryNode:
    """目录节点"""
    path: str  # 目录路径
    name: str  # 目录名称
    description: str  # 目录描述
    children: List[str] = field(default_factory=list)  # 子目录路径列表
    chunk_indices: List[int] = field(default_factory=list)  # 该目录下的 chunk 索引


@dataclass
class BatchOrganizationPlan:
    """批量组织计划"""
    directories: List[DirectoryNode] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class BatchProcessingResult:
    """批量处理结果"""
    success: bool = True
    memorized_chunks: List[BatchMemorizedChunk] = field(default_factory=list)
    organization_plan: Optional[BatchOrganizationPlan] = None
    created_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============== Taxonomy Schemas ==============

@dataclass
class TaxonomyNode:
    """目录树节点"""
    path: str
    name: str
    description: str
    children: List[str] = field(default_factory=list)  # 子目录路径


@dataclass
class TaxonomyTree:
    """目录树"""
    nodes: List[TaxonomyNode]
    reasoning: str
    changes_history: List[str] = field(default_factory=list)
    
    def to_tree_string(self) -> str:
        """转换为树形字符串"""
        if not self.nodes:
            return "(empty)"
        
        lines = []
        # 按深度排序
        sorted_nodes = sorted(self.nodes, key=lambda n: (n.path.count('/'), n.path))
        
        for node in sorted_nodes:
            depth = node.path.count('/') - 1
            indent = "  " * depth
            prefix = "├── " if depth > 0 else ""
            lines.append(f"{indent}{prefix}{node.name}/ - {node.description}")
        
        return "\n".join(lines)
    
    def get_all_paths(self) -> List[str]:
        """获取所有路径"""
        return [n.path for n in self.nodes]
    
    def get_leaf_paths(self) -> List[str]:
        """获取所有叶子节点路径"""
        all_children = set()
        for n in self.nodes:
            all_children.update(n.children)
        
        return [n.path for n in self.nodes if n.path not in all_children or not n.children]


@dataclass
class TaxonomyOrganizationPlan:
    """基于 taxonomy 的组织计划"""
    taxonomy: TaxonomyTree
    assignments: List["ChunkAssignmentResult"] = field(default_factory=list)


@dataclass
class ChunkAssignmentResult:
    """Chunk 分配结果"""
    chunk_index: int
    assigned_path: str
    reasoning: str = ""
    confidence: float = 1.0


# ============== Incremental Add Schemas ==============

@dataclass
class ChunkAddAssignment:
    """单个 chunk 的添加分配"""
    chunk_index: int
    chunk_title: str
    action: str  # "use_existing", "create_subdir", "create_toplevel"
    target_path: str
    new_dir_name: Optional[str] = None
    new_dir_description: Optional[str] = None


@dataclass
class NewDirectoryInfo:
    """新目录信息"""
    path: str
    name: str
    description: str
    parent_path: Optional[str] = None


@dataclass
class AddChunksResult:
    """添加 chunks 的结果"""
    success: bool = True
    assignments: List[ChunkAddAssignment] = field(default_factory=list)
    new_directories: List[NewDirectoryInfo] = field(default_factory=list)
    created_files: List[str] = field(default_factory=list)
    affected_paths: List[str] = field(default_factory=list)
    reasoning: str = ""
    errors: List[str] = field(default_factory=list)


@dataclass
class ReorganizeOperation:
    """重组操作"""
    operation: str  # "merge", "split", "rename", "move"
    source_paths: List[str]
    target_path: str
    new_name: Optional[str] = None
    description: Optional[str] = None
    files_to_move: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ReorganizeResult:
    """重组结果"""
    reorganization_needed: bool = False
    reasoning: str = ""
    summary: str = ""
    operations: List[ReorganizeOperation] = field(default_factory=list)
    directories_created: List[str] = field(default_factory=list)
    directories_removed: List[str] = field(default_factory=list)
    files_moved: List[Dict[str, str]] = field(default_factory=list)
    affected_paths: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class IncrementalAddState:
    """增量添加状态"""
    add_count: int = 0
    total_chunks_added: int = 0
    last_reorganization: Optional[str] = None
    recent_additions: List[Dict[str, Any]] = field(default_factory=list)


# ============== JSON Schemas for LLM ==============

MEMORIZE_CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Short descriptive title in snake_case format, e.g., pytorch_autograd_basics"
        },
        "tldr": {
            "type": "string",
            "description": "Structural overview describing what type of content, how it's organized, and what key topics are covered"
        },
        "memory": {
            "type": "string",
            "description": "Memory/summary of chunk content, preserving key information"
        }
    },
    "required": ["title", "memory"]
}

CHUNK_FORMAT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "content_type": {
            "type": "string",
            "description": "Type of content (e.g., 'academic_paper', 'technical_doc', 'code_tutorial')"
        },
        "structure_pattern": {
            "type": "string",
            "description": "Common structure patterns found in the content"
        },
        "chunking_guidance": {
            "type": "string",
            "description": "Guidance for how to chunk this type of content"
        }
    },
    "required": ["content_type", "structure_pattern", "chunking_guidance"]
}

MEMORY_FORMAT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "key_elements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key elements that should be captured in memories"
        },
        "format_guidance": {
            "type": "string",
            "description": "Guidance for how to format memories for this content"
        }
    },
    "required": ["key_elements", "format_guidance"]
}

BATCH_ORGANIZATION_SCHEMA = {
    "type": "object",
    "properties": {
        "directories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "children": {"type": "array", "items": {"type": "string"}},
                    "chunk_indices": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["path", "name"]
            }
        },
        "reasoning": {"type": "string"}
    },
    "required": ["directories"]
}

TAXONOMY_NODE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Full path like /category/subcategory"},
        "name": {"type": "string", "description": "Directory name"},
        "description": {"type": "string", "description": "Brief description"},
        "children": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["path", "name", "description"]
}

GENERATE_TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "taxonomy": {
            "type": "array",
            "items": TAXONOMY_NODE_SCHEMA
        },
        "reasoning": {"type": "string"}
    },
    "required": ["taxonomy"]
}

ADJUST_TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "taxonomy": {
            "type": "array",
            "items": TAXONOMY_NODE_SCHEMA
        },
        "reasoning": {"type": "string"},
        "changes_made": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["taxonomy"]
}

ASSIGN_CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "assigned_path": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["assigned_path"]
}

SHOULD_ADD_TO_EXISTING_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "belongs_to_existing": {"type": "boolean"},
        "suggested_location": {"type": "string"},
        "new_topic_name": {"type": "string"},
        "new_topic_description": {"type": "string"}
    },
    "required": ["belongs_to_existing"]
}

ADD_CHUNKS_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_index": {"type": "integer"},
                    "chunk_title": {"type": "string"},
                    "action": {"type": "string"},
                    "target_path": {"type": "string"},
                    "new_dir_name": {"type": "string"},
                    "new_dir_description": {"type": "string"}
                }
            }
        },
        "new_directories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "parent_path": {"type": "string"}
                }
            }
        },
        "affected_paths": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["assignments"]
}

REORGANIZE_TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "reorganization_needed": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "summary": {"type": "string"},
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "source_paths": {"type": "array", "items": {"type": "string"}},
                    "target_path": {"type": "string"},
                    "new_name": {"type": "string"},
                    "description": {"type": "string"},
                    "files_to_move": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string"},
                                "to": {"type": "string"}
                            }
                        }
                    }
                }
            }
        },
        "affected_paths": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["reorganization_needed"]
}

GAM_GENERATE_README_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "A brief description of what this directory contains"
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "brief": {"type": "string"},
                    "detailed": {"type": "string"}
                },
                "required": ["name", "brief"]
            }
        }
    },
    "required": ["description"]
}

MERGE_ROOT_README_SCHEMA = {
    "type": "object",
    "properties": {
        "merged_readme": {
            "type": "string",
            "description": "The merged README content"
        }
    },
    "required": ["merged_readme"]
}
