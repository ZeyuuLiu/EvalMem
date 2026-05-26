# -*- coding: utf-8 -*-
"""
TextGAMAgent - Merged GAMAgent + ChunkerAgent

A unified agent that handles:
1. Intelligent text chunking (from ChunkerAgent)
2. Memory/summary generation for each chunk
3. Organizing chunks into hierarchical GAM structure (from GAMAgent)
4. README generation for directories
5. Incremental additions to existing GAM

Key feature: `use_chunking` parameter controls whether to:
- True: Split input into multiple chunks intelligently
- False: Treat entire input as a single chunk
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from ..core.tree import GAMTree
from ..core.node import NodeType
from ..generators.base import BaseGenerator
from ..agents.gam_agent import BaseGAMAgent
from ..readers.pdf_reader import PdfReader
from ..readers.txt_reader import TxtReader
from ..workspaces.base import BaseWorkspace
from ..schemas.chunk_schemas import (
    # Chunking
    MemorizedChunk,
    ChunkResult,
    ChunkWithMemoryResult,
    # Organization
    BatchMemorizedChunk,
    DirectoryNode,
    BatchOrganizationPlan,
    BatchProcessingResult,
    # Taxonomy
    TaxonomyNode,
    TaxonomyTree,
    ChunkAssignmentResult,
    # Incremental add
    ChunkAddAssignment,
    NewDirectoryInfo,
    AddChunksResult,
    ReorganizeOperation,
    ReorganizeResult,
    IncrementalAddState,
    # JSON Schemas
    MEMORIZE_CHUNK_SCHEMA,
    CHUNK_FORMAT_ANALYSIS_SCHEMA,
    MEMORY_FORMAT_ANALYSIS_SCHEMA,
    GENERATE_TAXONOMY_SCHEMA,
    ADJUST_TAXONOMY_SCHEMA,
    ASSIGN_CHUNK_SCHEMA,
    SHOULD_ADD_TO_EXISTING_SCHEMA,
    ADD_CHUNKS_SCHEMA,
    REORGANIZE_TAXONOMY_SCHEMA,
    GAM_GENERATE_README_SCHEMA,
    MERGE_ROOT_README_SCHEMA,
)
from ..prompts.chunker_prompts import (
    CHUNK_DECISION_PROMPT,
    MEMORY_GENERATION_PROMPT,
    GENERATE_DOCUMENT_SUMMARY_PROMPT,
    ANALYZE_CHUNK_FORMAT_PROMPT,
    ANALYZE_MEMORY_FORMAT_PROMPT,
    DEFAULT_MEMORIZE_INSTRUCTION,
)
from ..prompts.gam_prompts import (
    GAM_GENERATE_README_PROMPT,
    GENERATE_TAXONOMY_FROM_TLDR_PROMPT,
    ADJUST_TAXONOMY_WITH_TLDR_PROMPT,
    ASSIGN_CHUNK_TO_TAXONOMY_PROMPT,
    SHOULD_ADD_TO_EXISTING_PROMPT,
    ADD_CHUNKS_TO_EXISTING_TAXONOMY_PROMPT,
    REORGANIZE_TAXONOMY_PROMPT,
    MERGE_ROOT_README_PROMPT,
)


class TextGAMAgent(BaseGAMAgent):
    """
    TextGAMAgent - Merged GAMAgent + ChunkerAgent
    
    Handles intelligent text chunking, memory generation, and organizing into GAM.
    
    Usage:
    ```python
    from gam import TextGAMAgent, GAMTree, OpenAIGenerator, LocalWorkspace
    
    generator = OpenAIGenerator(config)
    workspace = LocalWorkspace(root_path="./my_gam")
    tree = GAMTree.create(Path("./my_gam"), name="my_memory")
    agent = TextGAMAgent(generator, tree, workspace, use_chunking=True)
    
    # Add content — automatically creates GAM if empty, or adds incrementally
    result = agent.add(Path("./my_doc.txt"))
    
    # Add more content (incremental)
    result = agent.add(Path("./new_doc.txt"))
    
    tree.to_disk()
    ```
    """
    
    def __init__(
        self,
        generator: BaseGenerator,
        tree: GAMTree,
        workspace: BaseWorkspace,
        use_chunking: bool = True,
        window_size: int = 30000,
        overlap_size: int = 10000,
        encoding_name: str = "o200k_base",
        auto_save: bool = True,
        verbose: bool = True,
        memory_workers: int = 4,
        max_analysis_tokens: int | None = None,
    ):
        """
        Initialize TextGAMAgent
        
        Args:
            generator: LLM generator instance
            tree: GAM tree instance (used for reading/viewing structure)
            workspace: Workspace instance for executing Linux commands
            use_chunking: Whether to chunk input (True) or treat as single chunk (False)
            window_size: Sliding window size in tokens (for chunking)
            overlap_size: Overlap size in tokens (for chunking)
            encoding_name: tiktoken encoding name (default: o200k_base for GPT-4o-mini)
            auto_save: Whether to auto-save tree to disk
            verbose: Whether to print detailed info
            memory_workers: Number of parallel workers for memory generation
            max_analysis_tokens: Max tokens for global analysis (document summary / memory
                format analysis). None means no truncation (pass full content to LLM).
                Default is None (no truncation).
        """
        super().__init__(generator, tree, workspace)
        
        # workspace is now set by parent class
        self.use_chunking = use_chunking
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.encoding_name = encoding_name
        self.auto_save = auto_save
        self.verbose = verbose
        self.memory_workers = memory_workers
        self.max_analysis_tokens = max_analysis_tokens
        
        # History and state
        self.history: List[Dict[str, Any]] = []
        self.incremental_state = IncrementalAddState()
        self.reorganize_threshold = 5
        
        # Timestamp tracking: {path: "YYYY-MM-DD HH:MM:SS"}
        self._path_timestamps: Dict[str, str] = {}
        
        # Chunker state
        self.document_summary: str = ""
        self.chunk_format_guidance: str = ""
        self.memory_format_guidance: str = ""
        
        # Initialize tokenizer
        if not HAS_TIKTOKEN:
            raise ImportError(
                "tiktoken is required for TextGAMAgent. "
                "Install with: pip install tiktoken"
            )
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Initialize readers
        self.pdf_reader = PdfReader()
        self.txt_reader = TxtReader()
    
    # ========== Helper Methods ==========
    
    def _mkdir(self, path: str, readme: str = "") -> bool:
        """
        Create directory using workspace (Linux command)
        
        Args:
            path: Directory path (relative to workspace root)
            readme: Optional README content
            
        Returns:
            True if successful
        """
        # Create directory using Linux command
        abs_path = str(Path(self.workspace.root_path) / path.lstrip('/'))
        output, exit_code = self.workspace.run(f'mkdir -p "{abs_path}"')
        
        if exit_code != "0":
            if self.verbose:
                print(f"✗ mkdir failed for {path}: {output}")
            return False
        
        # Record creation timestamp (only if first time)
        if path not in self._path_timestamps:
            self._path_timestamps[path] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create README.md if provided
        if readme:
            readme_path = str(Path(abs_path) / "README.md")
            # Use cat with heredoc for multiline content
            # Escape the delimiter to prevent variable expansion
            delimiter = "EOF_README_CONTENT"
            cmd = f'cat > "{readme_path}" << \'{delimiter}\'\n{readme}\n{delimiter}'
            output, exit_code = self.workspace.run(cmd)
            if exit_code != "0":
                if self.verbose:
                    print(f"✗ Failed to write README for {path}: {output}")
                return False
        
        return True
    
    def _write_file(self, path: str, content: str) -> bool:
        """
        Write file using workspace (Linux command)
        
        Args:
            path: File path (relative to workspace root)
            content: File content
            
        Returns:
            True if successful
        """
        abs_path = str(Path(self.workspace.root_path) / path.lstrip('/'))
        
        # Create parent directory if needed
        parent_dir = str(Path(abs_path).parent)
        output, exit_code = self.workspace.run(f'mkdir -p "{parent_dir}"')
        if exit_code != "0":
            if self.verbose:
                print(f"✗ Failed to create parent directory for {path}: {output}")
            return False
        
        # Write file content using cat with heredoc for proper multiline handling
        # Escape the delimiter to prevent variable expansion
        delimiter = "EOF_FILE_CONTENT"
        cmd = f'cat > "{abs_path}" << \'{delimiter}\'\n{content}\n{delimiter}'
        output, exit_code = self.workspace.run(cmd)
        
        if exit_code != "0":
            if self.verbose:
                print(f"✗ Failed to write file {path}: {output}")
            return False
        
        # Record creation timestamp (only if first time)
        if path not in self._path_timestamps:
            self._path_timestamps[path] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return True
    
    def _mv(self, src: str, dest: str) -> bool:
        """
        Move file/directory using workspace (Linux command)
        
        Args:
            src: Source path
            dest: Destination path
            
        Returns:
            True if successful
        """
        abs_src = str(Path(self.workspace.root_path) / src.lstrip('/'))
        abs_dest = str(Path(self.workspace.root_path) / dest.lstrip('/'))
        
        # Create parent directory of destination
        parent_dir = str(Path(abs_dest).parent)
        self.workspace.run(f'mkdir -p "{parent_dir}"')
        
        output, exit_code = self.workspace.run(f'mv "{abs_src}" "{abs_dest}"')
        
        if exit_code != "0":
            if self.verbose:
                print(f"✗ Failed to move {src} to {dest}: {output}")
            return False
        
        return True
    
    def _rm(self, path: str) -> bool:
        """
        Remove file/directory using workspace (Linux command)
        
        Args:
            path: Path to remove
            
        Returns:
            True if successful
        """
        path = path.lstrip('/')
        abs_path = str(Path(self.workspace.root_path) / path)
        output, exit_code = self.workspace.run(f'rm -rf "{abs_path}"')
        
        if exit_code != "0":
            if self.verbose:
                print(f"✗ Failed to remove {path}: {output}")
            return False
        
        return True
    
    def _update_readme(self, dir_path: str, readme_content: str) -> bool:
        """
        Update README.md in a directory using workspace (Linux command)
        
        Args:
            dir_path: Directory path
            readme_content: README content
            
        Returns:
            True if successful
        """
        if dir_path == "/":
            readme_path = "README.md"
        else:
            readme_path = f"{dir_path.rstrip('/')}/README.md"
        
        return self._write_file(readme_path, readme_content)
    
    def _read_dir_structure(self) -> str:
        """
        Read directory structure using tree command (for viewing only)
        
        Returns:
            Tree structure string
        """
        output, exit_code = self.workspace.run('tree . -L 3 -I ".gam_meta.json"')
        
        if exit_code != "0":
            output, exit_code = self.workspace.run(
                'find . -type d | head -20'
            )
        
        return output if exit_code == "0" else "(Unable to read directory structure)"
    
    def _get_path_created_time(self, path: str) -> str:
        """
        Get the creation timestamp for a path.
        
        Priority:
        1. In-memory _path_timestamps (set during this session)
        2. Filesystem mtime via stat (fallback for pre-existing files)
        3. Current time (last resort)
        
        Args:
            path: Relative path (e.g., "/experimental_studies")
            
        Returns:
            Timestamp string in 'YYYY-MM-DD HH:MM:SS' format
        """
        # Check in-memory cache first
        if path in self._path_timestamps:
            return self._path_timestamps[path]
        
        # Fallback: read filesystem mtime
        abs_path = str(Path(self.workspace.root_path) / path.lstrip('/'))
        try:
            output, exit_code = self.workspace.run(f"stat -c '%Y' \"{abs_path}\" 2>/dev/null")
            if exit_code == "0" and output.strip():
                epoch = int(output.strip())
                ts = datetime.fromtimestamp(epoch).strftime('%Y-%m-%d %H:%M:%S')
                self._path_timestamps[path] = ts  # cache it
                return ts
        except (ValueError, OSError):
            pass
        
        # Last resort: current time
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def _read_file(self, input_file: Path) -> str:
        """
        根据文件扩展名自动选择合适的 reader 读取文件内容
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            文件内容文本
            
        Raises:
            ValueError: 不支持的文件类型
        """
        file_ext = input_file.suffix.lower()
        
        if file_ext == '.pdf':
            return self.pdf_reader.read(input_file)
        elif file_ext in ['.txt', '.md', '.text']:
            return self.txt_reader.read(input_file)
        else:
            # 默认尝试作为文本文件读取
            try:
                return self.txt_reader.read(input_file)
            except Exception:
                raise ValueError(
                    f"Unsupported file type: {file_ext}. "
                    f"Supported types: .pdf, .txt, .md, .text"
                )
    
    def _resolve_inputs(
        self,
        input_file: Path | List[Path] | None,
        content: str | List[str] | None,
    ) -> List[tuple[str, str]]:
        """
        Resolve input_file and/or content into a list of (text_content, source_name) pairs.
        
        Both input_file and content can be single values or lists.
        They can also be provided together (all will be included).
        
        Args:
            input_file: Path(s) to input file(s)
            content: Direct text content(s)
            
        Returns:
            List of (text_content, source_name) tuples
            
        Raises:
            ValueError: If neither input_file nor content is provided
        """
        items: List[tuple[str, str]] = []
        
        # Process content inputs
        if content is not None:
            if isinstance(content, str):
                content_list = [content]
            else:
                content_list = list(content)
            for i, c in enumerate(content_list):
                name = f"<content_{i+1}>" if len(content_list) > 1 else "<direct_content>"
                items.append((c, name))
        
        # Process file inputs
        if input_file is not None:
            if isinstance(input_file, (str, Path)):
                file_list = [Path(input_file)]
            else:
                file_list = [Path(f) for f in input_file]
            for f in file_list:
                text = self._read_file(f)
                items.append((text, f.name))
        
        if not items:
            raise ValueError("Either input_file or content must be provided.")
        
        return items
    
    # ========== Main Entry Point ==========
    
    def _create(
        self,
        input_file: Path | List[Path] | None = None,
        content: str | List[str] | None = None,
        context: str = "",
        use_chunking: Optional[bool] = None,
        memorize_instruction: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
        max_splits: int = 30,
        callback: Optional[Callable] = None,
        taxonomy_batch_size: int = 50,
    ) -> BatchProcessingResult:
        """
        Internal: Create GAM from input file(s) and/or text content(s).
        
        Called automatically by add() when the GAM is empty.
        
        This method:
        1. Resolves inputs (files and/or direct content, single or list)
        2. Optionally chunks each input (based on use_chunking)
        3. Generates memories for each chunk
        4. Organizes all chunks into directory structure
        5. Generates README files
        
        Args:
            input_file: Path(s) to input file(s), single Path or list of Paths
            content: Direct text content(s), single string or list of strings
            context: Optional context information
            use_chunking: Whether to chunk inputs; if None, uses self.use_chunking
            memorize_instruction: Custom instruction for memory generation
            output_dir: Optional directory to save raw chunks
            max_splits: Maximum split points per window
            callback: Optional callback function
            taxonomy_batch_size: Batch size for taxonomy generation
        
        Returns:
            BatchProcessingResult
            
        Raises:
            ValueError: If neither input_file nor content is provided
        """
        # Resolve all inputs into (text, source_name) pairs
        text_items = self._resolve_inputs(input_file, content)
        should_chunk = use_chunking if use_chunking is not None else self.use_chunking
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TextGAMAgent: Creating from {len(text_items)} input(s)")
            for i, (_, name) in enumerate(text_items):
                print(f"  Input {i+1}: {name}")
            print(f"Use chunking: {should_chunk}")
            print(f"{'='*60}\n")
        
        memorized_chunks: List[MemorizedChunk] = []
        chunk_index_offset = 0
        
        if should_chunk:
            # Chunk each input independently, collect all memorized chunks
            for text_content, source_name in text_items:
                if self.verbose:
                    print(f"\n--- Chunking input: {source_name} ---")
                
                chunk_result = self._chunk_with_memory_internal(
                    text=text_content,
                    context=context,
                    memorize_instruction=memorize_instruction,
                    output_dir=output_dir,
                    max_splits=max_splits,
                    on_memory_generated=lambda mc: callback("memory_generated", {"title": mc.title}) if callback else None,
                )
                
                # Re-index chunks with global offset
                for mc in chunk_result.memorized_chunks:
                    mc.index = chunk_index_offset + mc.index
                    memorized_chunks.append(mc)
                
                chunk_index_offset = len(memorized_chunks)
        else:
            # Each input becomes a single chunk
            if self.verbose:
                print(f"  Processing {len(text_items)} input(s) as individual chunks.")
            
            # Analyze format from combined content for global consistency
            combined_text = "\n\n---\n\n".join(text for text, _ in text_items)
            self.document_summary = self._get_document_summary(combined_text)
            
            content_to_analyze = self._get_content_for_analysis(combined_text)
            self.memory_format_guidance = self._analyze_memory_format(
                content_to_analyze,
                self.document_summary,
            )
            
            instruction = memorize_instruction or DEFAULT_MEMORIZE_INSTRUCTION
            
            for i, (text_content, source_name) in enumerate(text_items):
                single_mc = self._generate_single_memory(
                    index=i,
                    content=text_content,
                    instruction=instruction,
                    metadata={"original_source": source_name},
                )
                memorized_chunks.append(single_mc)
                
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    stem = re.sub(r'[<>:"/\\|?*]', '_', source_name.rsplit('.', 1)[0] if '.' in source_name else source_name)
                    single_mc.save(Path(output_dir), filename=f"{stem}.md")
                    if self.verbose:
                        print(f"  ✓ Saved chunk: {stem}.md - {single_mc.title}")
                
                if callback:
                    callback("memory_generated", {"title": single_mc.title})
        
        if not memorized_chunks:
            return BatchProcessingResult(success=False, errors=["No memorized chunks generated."])
        
        # Convert to BatchMemorizedChunk
        batch_memorized = []
        for mc in memorized_chunks:
            tldr = getattr(mc, 'tldr', '') or ''
            memory = mc.memory
            if not isinstance(memory, str):
                memory = json.dumps(memory, ensure_ascii=False)
            bmc = BatchMemorizedChunk(
                index=mc.index,
                title=mc.title,
                tldr=tldr,
                memory=memory,
                original_content=mc.content,
            )
            batch_memorized.append(bmc)
        
        # Organize into GAM
        result = self._organize_from_memorized_chunks_internal(
            memorized_chunks=batch_memorized,
            context=context,
            callback=callback,
            taxonomy_batch_size=taxonomy_batch_size,
        )
        
        return result

    def add(
        self,
        input_file: Path | List[Path] | None = None,
        content: str | List[str] | None = None,
        context: str = "",
        use_chunking: Optional[bool] = None,
        memorize_instruction: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
        max_splits: int = 120,
        callback: Optional[Callable] = None,
        taxonomy_batch_size: int = 50,
        force_reorganize: bool = False,
    ) -> Any:
        """
        Add memory from input file(s) or text content(s).
        
        This is the single unified entry point:
        - If the GAM is empty (no existing content), performs a full creation
          (generates taxonomy, organizes into directory structure).
        - If the GAM already has content, performs an incremental addition
          (checks whether new content belongs to existing structure or needs expansion).
        
        Args:
            input_file: Path(s) to input file(s), single Path or list of Paths
            content: Direct text content(s), single string or list of strings
            context: Optional context/instruction information
            use_chunking: Whether to chunk inputs; if None, uses self.use_chunking
            memorize_instruction: Custom instruction for memory generation (used in create mode)
            output_dir: Optional directory to save raw chunks to disk
            max_splits: Maximum split points per window (used in create mode)
            callback: Optional callback function
            taxonomy_batch_size: Batch size for taxonomy generation (used in create mode)
            force_reorganize: Force reorganization regardless of threshold (used in incremental mode)
        
        Returns:
            BatchProcessingResult (when creating) or AddChunksResult (when adding incrementally)
            
        Raises:
            ValueError: If neither input_file nor content is provided
        """
        # Reload tree from disk to get the latest state
        self.tree = GAMTree.from_disk(self.tree.root_path, self.workspace)
        
        if self.tree.is_empty:
            # GAM is empty — perform full creation
            if self.verbose:
                print(f"  GAM is empty, performing full creation...")
            return self._create(
                input_file=input_file,
                content=content,
                context=context,
                use_chunking=use_chunking,
                memorize_instruction=memorize_instruction,
                output_dir=output_dir,
                max_splits=max_splits,
                callback=callback,
                taxonomy_batch_size=taxonomy_batch_size,
            )
        else:
            # GAM has existing content — perform incremental addition
            if self.verbose:
                print(f"  GAM has existing content, performing incremental addition...")
            return self._add_incremental(
                input_file=input_file,
                content=content,
                context=context,
                use_chunking=use_chunking,
                force_reorganize=force_reorganize,
                callback=callback,
                output_dir=output_dir,
            )
    
    def _add_incremental(
        self,
        input_file: Path | List[Path] | None = None,
        content: str | List[str] | None = None,
        context: str = "",
        use_chunking: Optional[bool] = None,
        force_reorganize: bool = False,
        callback: Optional[Callable] = None,
        output_dir: Optional[str | Path] = None,
    ) -> AddChunksResult:
        """
        Internal: Add memory incrementally without overwriting existing data.
        
        Called automatically by add() when the GAM already has content.
        
        This method resolves inputs (files and/or direct content, single or list),
        optionally chunks them, generates memories, and intelligently adds them to
        the existing tree structure.
        
        It checks whether the new content belongs to the existing structure:
        - If yes: adds to existing directories
        - If no: creates a new topic directory and reorganizes
        
        Args:
            input_file: Path(s) to input file(s), single Path or list of Paths
            content: Direct text content(s), single string or list of strings
            context: Optional context/instruction for segmentation
            use_chunking: Whether to chunk inputs; if None, uses self.use_chunking
            force_reorganize: Force reorganization regardless of threshold
            callback: Optional callback function
            output_dir: Optional directory to save raw chunks to disk
        
        Returns:
            AddChunksResult
            
        Raises:
            ValueError: If neither input_file nor content is provided
        """
        # Resolve all inputs into (text, source_name) pairs
        text_items = self._resolve_inputs(input_file, content)
        should_chunk = use_chunking if use_chunking is not None else self.use_chunking
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TextGAMAgent: Adding {len(text_items)} input(s)")
            for i, (_, name) in enumerate(text_items):
                print(f"  Input {i+1}: {name}")
            print(f"Use chunking: {should_chunk}")
            print(f"Add count before: {self.incremental_state.add_count}")
            print(f"{'='*60}\n")
        
        memorized_chunks: List[MemorizedChunk] = []
        chunk_index_offset = 0

        chunk_output_path = Path(output_dir) if output_dir else None
        if chunk_output_path:
            chunk_output_path.mkdir(parents=True, exist_ok=True)
        
        if should_chunk:
            # Chunk each input independently, collect all memorized chunks
            for text_content, source_name in text_items:
                if self.verbose:
                    print(f"\n--- Chunking input: {source_name} ---")
                
                chunk_result = self._chunk_with_memory_internal(
                    text=text_content,
                    context=context,
                    memorize_instruction=None,
                    output_dir=chunk_output_path,
                    max_splits=120,
                )
                
                # Re-index chunks with global offset
                for mc in chunk_result.memorized_chunks:
                    mc.index = chunk_index_offset + mc.index
                    memorized_chunks.append(mc)
                
                chunk_index_offset = len(memorized_chunks)
        else:
            # Each input becomes a single chunk
            if self.verbose:
                print(f"  Processing {len(text_items)} input(s) as individual chunks.")
            
            # Analyze format from combined content for global consistency
            combined_text = "\n\n---\n\n".join(text for text, _ in text_items)
            self.document_summary = self._get_document_summary(combined_text)
            
            content_to_analyze = self._get_content_for_analysis(combined_text)
            self.memory_format_guidance = self._analyze_memory_format(
                content_to_analyze,
                self.document_summary,
            )
            
            for i, (text_content, source_name) in enumerate(text_items):
                single_mc = self._generate_single_memory(
                    index=i,
                    content=text_content,
                    instruction=DEFAULT_MEMORIZE_INSTRUCTION,
                    metadata={"original_source": source_name},
                )
                memorized_chunks.append(single_mc)
                
                if chunk_output_path:
                    stem = re.sub(r'[<>:"/\\|?*]', '_', source_name.rsplit('.', 1)[0] if '.' in source_name else source_name)
                    single_mc.save(chunk_output_path, filename=f"{stem}.md")
                    if self.verbose:
                        print(f"  ✓ Saved chunk: {stem}.md - {single_mc.title}")
        
        if not memorized_chunks:
            result = AddChunksResult()
            result.success = False
            result.errors.append("No memorized chunks generated.")
            return result
        
        # Convert to BatchMemorizedChunk
        batch_memorized = []
        for mc in memorized_chunks:
            tldr = getattr(mc, 'tldr', '') or ''
            memory = mc.memory
            if not isinstance(memory, str):
                memory = json.dumps(memory, ensure_ascii=False)
            bmc = BatchMemorizedChunk(
                index=mc.index,
                title=mc.title,
                tldr=tldr,
                memory=memory,
                original_content=mc.content,
            )
            batch_memorized.append(bmc)
        
        # Step 1: Get root README and structure
        root_readme = self.tree.root.summary or ""
        previous_root_readme = root_readme
        existing_tree = self.get_tree_view()
        
        if self.verbose:
            print(f"\n--- Current Directory Structure ---")
            print(existing_tree)
        
        new_content_summary = self._build_content_summary(batch_memorized)
        
        # Step 2: Check if new content belongs to existing structure
        if self.verbose:
            print(f"\n--- Checking if new content belongs to existing structure ---")
        
        decision = self._check_belongs_to_existing(
            root_readme=root_readme,
            existing_tree=existing_tree,
            new_content_summary=new_content_summary,
        )
        
        belongs_to_existing = decision.get("belongs_to_existing", True)
        
        if self.verbose:
            print(f"  Decision: {'belongs to existing' if belongs_to_existing else 'needs separate directory'}")
            print(f"  Reasoning: {decision.get('reasoning', '')[:100]}...")
        
        if callback:
            callback("decision_made", {"belongs_to_existing": belongs_to_existing})
        
        # Step 3: Add to structure based on decision
        if belongs_to_existing:
            result = self._add_to_existing_structure(
                memorized_chunks=memorized_chunks,
                batch_memorized=batch_memorized,
                suggested_location=decision.get("suggested_location", "/"),
                context=context,
            )
        else:
            result = self._expand_with_new_topic(
                memorized_chunks=memorized_chunks,
                batch_memorized=batch_memorized,
                new_topic_name=decision.get("new_topic_name", "new_content"),
                new_topic_description=decision.get("new_topic_description", "New content"),
                context=context,
            )
        
        # Step 4: Merge root README
        self._merge_root_readme(previous_root_readme)
        
        # Step 5: Update incremental state
        self.incremental_state.add_count += 1
        self.incremental_state.total_chunks_added += len(memorized_chunks)
        self.incremental_state.recent_additions.append({
            "timestamp": datetime.now().isoformat(),
            "chunks_added": len(memorized_chunks),
            "affected_paths": result.affected_paths,
            "expanded": not belongs_to_existing,
        })
        
        if len(self.incremental_state.recent_additions) > 10:
            self.incremental_state.recent_additions = self.incremental_state.recent_additions[-10:]
        
        # Step 6: Check if reorganization needed
        should_reorganize = force_reorganize or (self.incremental_state.add_count >= self.reorganize_threshold)
        if should_reorganize:
            if self.verbose:
                print(f"\n--- Triggering Taxonomy Reorganization (add_count={self.incremental_state.add_count}) ---")
            
            reorganize_result = self._reorganize_taxonomy()
            if reorganize_result.reorganization_needed:
                self._update_affected_readmes(reorganize_result.affected_paths)
                self.incremental_state.add_count = 0
                self.incremental_state.last_reorganization = datetime.now().isoformat()
        
        # Save
        if self.auto_save:
            # Reload tree from disk to sync with workspace changes
            self.tree = GAMTree.from_disk(self.tree.root_path, self.workspace)
            if self.verbose:
                print(f"\n✓ GAM synced from: {self.tree.root_path}")
        
        # Record history
        self.history.append({
            "mode": "add",
            "chunks_count": len(memorized_chunks),
            "expanded": not belongs_to_existing,
            "result": {
                "files_created": len(result.created_files),
                "directories_created": len(result.new_directories),
                "affected_paths": len(result.affected_paths),
            },
        })
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Add Complete!")
            print(f"Mode: {'Added to existing' if belongs_to_existing else 'Expanded with new topic'}")
            print(f"Files created: {len(result.created_files)}")
            print(f"New directories: {len(result.new_directories)}")
            print(f"READMEs updated: {len(result.affected_paths)}")
            print(f"{'='*60}\n")
        
        return result

    def organize_from_memorized_chunks(
        self,
        memorized_chunks: List[MemorizedChunk],
        context: str = "",
        callback: Optional[Callable] = None,
        taxonomy_batch_size: int = 50,
    ) -> BatchProcessingResult:
        """
        Organize pre-generated memorized chunks into GAM.
        
        Use this method when you already have memorized chunks
        (e.g., from a previous chunking operation).
        
        Args:
            memorized_chunks: List of MemorizedChunk objects
            context: Optional context information
            callback: Optional callback function
            taxonomy_batch_size: Batch size for taxonomy generation
        
        Returns:
            BatchProcessingResult
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("TextGAMAgent: Organize from Memorized Chunks")
            print(f"Total chunks: {len(memorized_chunks)}")
            print(f"{'='*60}\n")
        
        # Convert to BatchMemorizedChunk
        batch_memorized = []
        for mc in memorized_chunks:
            tldr = getattr(mc, 'tldr', '') or ''
            memory = mc.memory
            if not isinstance(memory, str):
                memory = json.dumps(memory, ensure_ascii=False)
            bmc = BatchMemorizedChunk(
                index=mc.index,
                title=mc.title,
                tldr=tldr,
                memory=memory,
                original_content=mc.content,
            )
            batch_memorized.append(bmc)
        
        return self._organize_from_memorized_chunks_internal(
            memorized_chunks=batch_memorized,
            context=context,
            callback=callback,
            taxonomy_batch_size=taxonomy_batch_size,
        )
    
    # ========== Chunking Methods (from ChunkerAgent) ==========
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def _truncate_by_tokens(self, text: str, max_tokens: int, from_end: bool = False) -> str:
        """Truncate text by token count"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        if from_end:
            truncated_tokens = tokens[-max_tokens:]
        else:
            truncated_tokens = tokens[:max_tokens]
        
        return self.encoding.decode(truncated_tokens)
    
    def _get_content_for_analysis(self, text: str, max_tokens: int | None = None) -> str:
        """Get content for global analysis (head + tail if too long).

        If max_tokens is None and self.max_analysis_tokens is None, the full
        text is returned without any truncation.
        """
        effective_max = max_tokens if max_tokens is not None else self.max_analysis_tokens
        if effective_max is None:
            return text

        half_max_tokens = effective_max // 2
        text_tokens = self._count_tokens(text)

        if text_tokens <= effective_max:
            return text

        if self.verbose:
            print(f"  Content length {text_tokens} tokens exceeds {effective_max} tokens, using head/tail")

        head_part = self._truncate_by_tokens(text, half_max_tokens, from_end=False)
        tail_part = self._truncate_by_tokens(text, half_max_tokens, from_end=True)
        return head_part + "\n\n[... middle part omitted ...]\n\n" + tail_part
    
    def _get_document_summary(self, text: str) -> str:
        """Generate document summary"""
        content_to_summarize = self._get_content_for_analysis(text)
        prompt = GENERATE_DOCUMENT_SUMMARY_PROMPT.format(content=content_to_summarize)
        
        try:
            if self.verbose:
                print(f"  Generating document summary...")
            response = self.generator.generate_single(prompt=prompt)
            summary = response.get("text", "").strip()
            if self.verbose:
                print(f"  Document summary generated ({len(summary)} chars)")
            return summary
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Error generating document summary: {e}")
            return ""
    
    def _analyze_chunk_format(self, content_to_analyze: str) -> str:
        """Analyze how content should be chunked"""
        prompt = ANALYZE_CHUNK_FORMAT_PROMPT.format(content=content_to_analyze)
        
        try:
            if self.verbose:
                print(f"  Analyzing chunk format from global view...")
            response = self.generator.generate_single(
                prompt=prompt,
                schema=CHUNK_FORMAT_ANALYSIS_SCHEMA,
            )
            
            data = response.get("json")
            if data is None:
                data = self._extract_json(response.get("text", ""))
            
            if data:
                guidance_parts = []
                if data.get("content_type"):
                    guidance_parts.append(f"- **Content Type**: {data['content_type']}")
                if data.get("natural_structure"):
                    guidance_parts.append(f"- **Natural Structure**: {data['natural_structure']}")
                if data.get("recommended_granularity"):
                    guidance_parts.append(f"- **Recommended Granularity**: {data['recommended_granularity']}")
                if data.get("boundaries_to_respect"):
                    guidance_parts.append(f"- **Boundaries to Respect**: {data['boundaries_to_respect']}")
                if data.get("special_considerations"):
                    guidance_parts.append(f"- **Special Considerations**: {data['special_considerations']}")
                
                guidance = "\n".join(guidance_parts)
                if self.verbose:
                    print(f"  Chunk format analysis complete")
                return guidance
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Error analyzing chunk format: {e}")
        
        return "No specific chunking guidance available. Use default chunking rules."
    
    def _analyze_memory_format(self, content_to_analyze: str, document_summary: str) -> str:
        """Analyze how memories should be formatted"""
        prompt = ANALYZE_MEMORY_FORMAT_PROMPT.format(
            content=content_to_analyze,
            document_summary=document_summary,
        )
        
        try:
            if self.verbose:
                print(f"  Analyzing memory format from global view...")
            response = self.generator.generate_single(
                prompt=prompt,
                schema=MEMORY_FORMAT_ANALYSIS_SCHEMA,
            )
            
            data = response.get("json")
            if data is None:
                data = self._extract_json(response.get("text", ""))
            
            if data:
                guidance_parts = []
                if data.get("key_information_types"):
                    guidance_parts.append(f"- **Key Information to Preserve**: {data['key_information_types']}")
                if data.get("recommended_structure"):
                    guidance_parts.append(f"- **Recommended Structure**: {data['recommended_structure']}")
                if data.get("tldr_focus"):
                    guidance_parts.append(f"- **TLDR Focus**: {data['tldr_focus']}")
                if data.get("abstraction_level"):
                    guidance_parts.append(f"- **Abstraction Level**: {data['abstraction_level']}")
                if data.get("special_instructions"):
                    guidance_parts.append(f"- **Special Instructions**: {data['special_instructions']}")
                
                guidance = "\n".join(guidance_parts)
                if self.verbose:
                    print(f"  Memory format analysis complete")
                return guidance
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Error analyzing memory format: {e}")
        
        return "No specific memory guidance available. Use default memorization rules."
    
    def _count_meaningful_lines(self, text: str) -> int:
        """Count meaningful lines (non-empty)"""
        lines = text.split('\n')
        meaningful_lines = [line for line in lines if line.strip()]
        return len(meaningful_lines)
    
    def _merge_tiny_chunks(
        self,
        chunks: List[str],
        metadata: List[Dict[str, Any]],
        min_lines: int = 5
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Merge tiny chunks into adjacent ones"""
        if len(chunks) <= 1:
            return chunks, metadata
        
        merged_chunks = []
        merged_metadata = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            current_meta = metadata[i].copy()
            line_count = self._count_meaningful_lines(current_chunk)
            
            if line_count < min_lines and i < len(chunks) - 1:
                if self.verbose:
                    print(f"  Merging tiny chunk {i} ({line_count} lines) with chunk {i+1}")
                
                next_chunk = chunks[i + 1]
                next_meta = metadata[i + 1]
                
                merged_chunk = current_chunk + "\n" + next_chunk
                merged_meta = {
                    "index": len(merged_chunks),
                    "type": "merged",
                    "merged_from": [current_meta.get("index", i), next_meta.get("index", i+1)],
                    "tokens": current_meta.get("tokens", 0) + next_meta.get("tokens", 0),
                }
                
                merged_chunks.append(merged_chunk)
                merged_metadata.append(merged_meta)
                i += 2
                
            elif line_count < min_lines and i == len(chunks) - 1 and len(merged_chunks) > 0:
                if self.verbose:
                    print(f"  Merging tiny final chunk {i} ({line_count} lines) with previous chunk")
                
                last_chunk = merged_chunks[-1]
                last_meta = merged_metadata[-1]
                
                merged_chunk = last_chunk + "\n" + current_chunk
                merged_meta = {
                    "index": last_meta["index"],
                    "type": "merged",
                    "merged_from": last_meta.get("merged_from", [last_meta.get("index", len(merged_chunks)-1)]) + [current_meta.get("index", i)],
                    "tokens": last_meta.get("tokens", 0) + current_meta.get("tokens", 0),
                }
                
                merged_chunks[-1] = merged_chunk
                merged_metadata[-1] = merged_meta
                i += 1
                
            else:
                current_meta["index"] = len(merged_chunks)
                merged_chunks.append(current_chunk)
                merged_metadata.append(current_meta)
                i += 1
        
        if self.verbose and len(merged_chunks) < len(chunks):
            print(f"\n  Post-processing: Merged {len(chunks)} chunks into {len(merged_chunks)} chunks")
        
        return merged_chunks, merged_metadata
    
    def chunk(self, text: str, context: str = "", max_splits: int = 120) -> ChunkResult:
        """
        Intelligently chunk long text.
        
        Args:
            text: Text to chunk
            context: Optional context information
            max_splits: Maximum split points per window
        
        Returns:
            ChunkResult containing all chunks
        """
        # Prepare content for analysis
        content_to_analyze = self._get_content_for_analysis(text)
        
        # Generate context if not provided
        if not context:
            context = self._get_document_summary(text)
        self.document_summary = context
        
        # Analyze chunk format
        if self.verbose:
            print(f"\n--- Global Format Analysis ---")
        self.chunk_format_guidance = self._analyze_chunk_format(content_to_analyze)
        
        total_tokens = self._count_tokens(text)
        total_chars = len(text)
        
        chars_per_token = total_chars / total_tokens if total_tokens > 0 else 4.0
        window_chars = int(self.window_size * chars_per_token)
        overlap_chars = int(self.overlap_size * chars_per_token)
        
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"TextGAMAgent: Processing text")
            print(f"Text length: {total_chars} chars")
            print(f"Total tokens: {total_tokens} (using {self.encoding_name})")
            print(f"Chars per token: {chars_per_token:.2f}")
            print(f"Window size: {self.window_size} tokens (~{window_chars} chars)")
            print(f"{'='*50}\n")
        
        # Short text handling
        if total_tokens <= self.window_size:
            if self.verbose:
                print(f"Text fits in window, asking LLM for split decision...")
            decision = self._decide_split(text, max_splits=max_splits)
            if self.verbose:
                print(f"Short text LLM decision: {decision}")
            
            if decision["should_split"] and decision["split_lines"]:
                split_char_positions = self._find_split_char_positions_by_lines(
                    text, decision["split_lines"]
                )
                
                if split_char_positions:
                    chunks: List[str] = []
                    metadata: List[Dict[str, Any]] = []
                    start_pos = 0
                    
                    for split_pos in split_char_positions:
                        chunk_content = text[start_pos:split_pos].strip()
                        if chunk_content:
                            chunk_tokens = self._count_tokens(chunk_content)
                            chunks.append(chunk_content)
                            metadata.append({
                                "index": len(chunks) - 1,
                                "type": "split",
                                "char_start": start_pos,
                                "char_end": split_pos,
                                "tokens": chunk_tokens,
                            })
                        start_pos = split_pos
                    
                    tail_content = text[start_pos:].strip()
                    if tail_content:
                        tail_tokens = self._count_tokens(tail_content)
                        chunks.append(tail_content)
                        metadata.append({
                            "index": len(chunks) - 1,
                            "type": "final",
                            "char_start": start_pos,
                            "char_end": total_chars,
                            "tokens": tail_tokens,
                        })
                    
                    if chunks:
                        chunks, metadata = self._merge_tiny_chunks(chunks, metadata, min_lines=25)
                        return ChunkResult(
                            chunks=chunks,
                            metadata=metadata,
                            total_tokens=total_tokens,
                            num_chunks=len(chunks),
                        )
            
            if self.verbose:
                print("Text is short enough, no chunking needed.")
            return ChunkResult(
                chunks=[text],
                metadata=[{"index": 0, "type": "full", "tokens": total_tokens}],
                total_tokens=total_tokens,
                num_chunks=1,
            )
        
        # Sliding window processing
        chunks: List[str] = []
        metadata: List[Dict[str, Any]] = []
        pending_start = 0
        position = 0
        
        while pending_start < total_chars:
            window_end = min(position + window_chars, total_chars)
            window_text = text[position:window_end]
            window_tokens = self._count_tokens(window_text)
            
            is_last_window = window_end >= total_chars
            pending_tokens = self._count_tokens(text[pending_start:window_end])
            
            if self.verbose:
                print(f"\n--- Window [char {position}:{window_end}] / {total_chars} ---")
                print(f"Window size: {len(window_text)} chars (~{window_tokens} tokens)")
                print(f"Pending tokens: {pending_tokens} (from char {pending_start})")
            
            if is_last_window and pending_tokens <= self.window_size * 1.2:
                remaining_text = text[pending_start:].strip()
                if remaining_text:
                    remaining_tokens = self._count_tokens(remaining_text)
                    if self.verbose:
                        print(f"Outputting final chunk ({remaining_tokens} tokens).")
                    chunks.append(remaining_text)
                    metadata.append({
                        "index": len(chunks) - 1,
                        "type": "final",
                        "char_start": pending_start,
                        "char_end": total_chars,
                        "tokens": remaining_tokens,
                    })
                break
            
            decision = self._decide_split(window_text, max_splits=max_splits)
            
            if self.verbose:
                print(f"LLM decision: {decision}")
            
            split_char_positions = []
            if not (decision["should_split"] and decision["split_lines"]) and pending_tokens > self.window_size * 1.5:
                if self.verbose:
                    print(f"  Warning: Pending content too large ({pending_tokens} tokens), forcing split.")
                decision["should_split"] = True
                lines = window_text.split('\n')
                split_idx = int(len(lines) * 0.75)
                decision["split_lines"] = [split_idx] if split_idx > 0 else [len(lines) // 2 or 1]
            
            if decision["should_split"] and decision["split_lines"]:
                if self.verbose:
                    print(f"Decision: SPLIT - {decision['reasoning'][:100]}...")
                
                split_char_positions = self._find_split_char_positions_by_lines(
                    window_text, decision["split_lines"]
                )
                
                # Fallback: if line-based split failed (e.g., single-line text with no '\n'),
                # force a character-position-based split at the window midpoint.
                if not split_char_positions and pending_tokens > self.window_size:
                    pending_text = text[pending_start:window_end]
                    target_tokens = self.window_size
                    fallback_split_pos = len(self._truncate_by_tokens(pending_text, target_tokens))
                    if fallback_split_pos > 0 and fallback_split_pos < len(pending_text):
                        abs_split = pending_start + fallback_split_pos
                        split_char_positions = [abs_split - position]  # relative to window
                        if self.verbose:
                            print(f"  Fallback: character-based split at char {abs_split} ({target_tokens} tokens target)")
                
                if split_char_positions:
                    first_split_abs = position + split_char_positions[0]
                    chunk_content = text[pending_start:first_split_abs].strip()
                    if chunk_content:
                        chunk_tokens = self._count_tokens(chunk_content)
                        chunks.append(chunk_content)
                        metadata.append({
                            "index": len(chunks) - 1,
                            "type": "split",
                            "char_start": pending_start,
                            "char_end": first_split_abs,
                            "tokens": chunk_tokens,
                        })
                        if self.verbose:
                            print(f"  Output chunk {len(chunks)}: [char {pending_start}:{first_split_abs}] ({chunk_tokens} tokens)")
                    
                    for i in range(len(split_char_positions) - 1):
                        start_abs = position + split_char_positions[i]
                        end_abs = position + split_char_positions[i + 1]
                        chunk_content = text[start_abs:end_abs].strip()
                        if chunk_content:
                            chunk_tokens = self._count_tokens(chunk_content)
                            chunks.append(chunk_content)
                            metadata.append({
                                "index": len(chunks) - 1,
                                "type": "split",
                                "char_start": start_abs,
                                "char_end": end_abs,
                                "tokens": chunk_tokens,
                            })
                            if self.verbose:
                                print(f"  Output chunk {len(chunks)}: [char {start_abs}:{end_abs}] ({chunk_tokens} tokens)")
                    
                    pending_start = position + split_char_positions[-1]
                    position = pending_start
                    
                    if self.verbose:
                        print(f"  Next window will start from last split point: char {position}")
                else:
                    if self.verbose:
                        print("  Split points not found, moving forward with overlap")
                    position = window_end - overlap_chars
            else:
                if self.verbose:
                    print(f"Decision: NO SPLIT. Moving forward, keeping overlap. Pending from char {pending_start}")
                position = window_end - overlap_chars
            
            if position <= pending_start:
                position = pending_start + (window_chars // 2)
                if position >= total_chars:
                    position = total_chars - 1
            
            # Safety: if we're at the last window and can't make progress,
            # force output remaining content to avoid infinite loop.
            if is_last_window and pending_start < total_chars:
                new_window_end = min(position + window_chars, total_chars)
                if new_window_end >= total_chars:
                    remaining_text = text[pending_start:].strip()
                    if remaining_text:
                        remaining_tokens = self._count_tokens(remaining_text)
                        if self.verbose:
                            print(f"  Safety: forcing final chunk output ({remaining_tokens} tokens) to avoid stuck loop.")
                        chunks.append(remaining_text)
                        metadata.append({
                            "index": len(chunks) - 1,
                            "type": "forced_final",
                            "char_start": pending_start,
                            "char_end": total_chars,
                            "tokens": remaining_tokens,
                        })
                    break
        
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Chunking complete!")
            print(f"Total chunks: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                chunk_tokens = metadata[i].get("tokens", self._count_tokens(chunk))
                print(f"  Chunk {i+1}: {chunk_tokens} tokens ({len(chunk)} chars)")
            print(f"{'='*50}\n")
        
        chunks, metadata = self._merge_tiny_chunks(chunks, metadata, min_lines=25)
        
        return ChunkResult(
            chunks=chunks,
            metadata=metadata,
            total_tokens=total_tokens,
            num_chunks=len(chunks),
        )
    
    def _chunk_with_memory_internal(
        self,
        text: str,
        context: str = "",
        memorize_instruction: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
        max_splits: int = 120,
        on_chunk_created: Optional[Callable[[int, str], None]] = None,
        on_memory_generated: Optional[Callable[[MemorizedChunk], None]] = None,
    ) -> ChunkWithMemoryResult:
        """Internal method for chunk with memory generation"""
        if self.verbose:
            print(f"\n{'='*60}")
            print("TextGAMAgent: Chunk with Memory (Streaming)")
            print(f"{'='*60}\n")
        
        output_path = Path(output_dir) if output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        
        instruction = memorize_instruction or DEFAULT_MEMORIZE_INSTRUCTION
        content_to_analyze = self._get_content_for_analysis(text)
        
        memorized_chunks: List[MemorizedChunk] = []
        memorized_chunks_lock = Lock()
        
        def generate_and_save_memory(index: int, chunk_content: str, chunk_metadata: Dict) -> MemorizedChunk:
            mc = self._generate_single_memory(
                index=index,
                content=chunk_content,
                instruction=instruction,
                metadata=chunk_metadata,
            )
            
            if output_path:
                mc.save(output_path)
                if self.verbose:
                    print(f"  ✓ Saved: chunk_{index:03d}.md - {mc.title}")
            
            if on_memory_generated:
                on_memory_generated(mc)
            
            return mc
        
        total_tokens = self._count_tokens(text)
        chunk_data: List[tuple[int, str, Dict]] = []
        
        chunk_result = self.chunk(text, context, max_splits=max_splits)
        
        for i, (chunk_content, chunk_metadata) in enumerate(zip(chunk_result.chunks, chunk_result.metadata)):
            if on_chunk_created:
                on_chunk_created(i, chunk_content)
            chunk_data.append((i, chunk_content, chunk_metadata))
        
        if self.verbose:
            print(f"\n--- Memory Format Analysis ---")
        self.memory_format_guidance = self._analyze_memory_format(
            content_to_analyze,
            getattr(self, "document_summary", ""),
        )
        
        if self.verbose:
            print(f"\n--- Starting Memory Generation ({len(chunk_data)} chunks) ---\n")
        
        with ThreadPoolExecutor(max_workers=self.memory_workers) as executor:
            futures = {
                executor.submit(generate_and_save_memory, idx, content, meta): idx
                for idx, content, meta in chunk_data
            }
            
            for future in as_completed(futures):
                try:
                    mc = future.result()
                    with memorized_chunks_lock:
                        memorized_chunks.append(mc)
                except Exception as e:
                    idx = futures[future]
                    if self.verbose:
                        print(f"  ✗ Error generating memory for chunk {idx}: {e}")
        
        memorized_chunks.sort(key=lambda x: x.index)
        
        if output_path:
            readme_lines = [
                f"# Chunks Directory",
                f"",
                f"This directory contains {len(memorized_chunks)} intelligently split chunks with generated memories.",
                f"",
                f"## Chunk Index",
                f""
            ]
            for mc in memorized_chunks:
                readme_lines.append(f"- **chunk_{mc.index:03d}.md**: {mc.title}")
            
            readme_path = output_path / "README.md"
            readme_path.write_text("\n".join(readme_lines), encoding='utf-8')
            if self.verbose:
                print(f"  ✓ Generated: README.md index")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Memory Generation Complete!")
            print(f"Total memorized chunks: {len(memorized_chunks)}")
            if output_path:
                print(f"Saved to: {output_path}")
            print(f"{'='*60}\n")
        
        return ChunkWithMemoryResult(
            memorized_chunks=memorized_chunks,
            total_tokens=total_tokens,
            num_chunks=len(memorized_chunks),
            output_dir=output_path,
        )
    
    def _generate_single_memory(
        self,
        index: int,
        content: str,
        instruction: str,
        metadata: Optional[Dict] = None,
    ) -> MemorizedChunk:
        """Generate memory for a single chunk"""
        prompt = MEMORY_GENERATION_PROMPT.format(
            document_summary=getattr(self, "document_summary", ""),
            memory_format_guidance=getattr(self, "memory_format_guidance", "No specific guidance."),
            chunk_index=index,
            chunk_content=content[:100000],
            memorize_instruction=instruction,
        )
        
        try:
            response = self.generator.generate_single(
                prompt=prompt,
                schema=MEMORIZE_CHUNK_SCHEMA,
            )
            
            data = response.get("json")
            if data is None:
                data = self._extract_json(response.get("text", ""))
            
            if data:
                title = data.get("title", f"chunk_{index}")
                memory = data.get("memory", content[:500])
                tldr = data.get("tldr", content.splitlines()[0][:100] if content else "")
            else:
                first_line = content.split('\n')[0][:50]
                title = re.sub(r'[^\w\-]', '_', first_line.lower().strip())
                title = re.sub(r'_+', '_', title).strip('_') or f"chunk_{index}"
                memory = content[:500] + "..." if len(content) > 500 else content
                tldr = content.splitlines()[0][:100] if content else ""
                
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Error generating memory for chunk {index}: {e}")
            title = f"chunk_{index}"
            memory = content[:500] + "..." if len(content) > 500 else content
            tldr = content.splitlines()[0][:100] if content else ""
        
        return MemorizedChunk(
            index=index,
            title=title,
            memory=memory,
            tldr=tldr,
            content=content,
            metadata=metadata or {},
        )
    
    def _add_line_numbers(self, text: str) -> tuple[str, List[str]]:
        """Add line numbers to text"""
        lines = text.split('\n')
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            numbered_lines.append(f"line {i}: {line}")
        return '\n'.join(numbered_lines), lines
    
    def _decide_split(self, text: str, max_splits: int = 120, context: str = "") -> Dict[str, Any]:
        """Let LLM decide if and where to split"""
        token_count = self._count_tokens(text)
        numbered_text, original_lines = self._add_line_numbers(text)
        
        preview_text = numbered_text
        
        prompt = CHUNK_DECISION_PROMPT.format(
            document_summary=context or getattr(self, "document_summary", ""),
            chunk_format_guidance=getattr(self, "chunk_format_guidance", "No specific guidance."),
            token_count=token_count,
            text_segment=preview_text,
        )
        
        max_retries = 3
        last_oversized_split_lines = None
        
        for attempt in range(max_retries):
            if self.verbose:
                preview_tokens = self._count_tokens(preview_text)
                retry_msg = f" (Attempt {attempt + 1}/{max_retries})" if attempt > 0 else ""
                print(f"  Sending {preview_tokens} tokens to LLM for split decision...{retry_msg}")
            
            try:
                response = self.generator.generate_single(prompt=prompt)
                if self.verbose:
                    print(f"  LLM response received.")
                result_text = response.get("text", "")
                
                data = self._extract_json(result_text)
                
                if data:
                    split_lines = data.get("split_lines", [])
                    if isinstance(split_lines, list):
                        split_lines = [int(x) for x in split_lines if isinstance(x, (int, float, str)) and str(x).isdigit()]
                    else:
                        split_lines = []
                    
                    if len(split_lines) > max_splits:
                        if self.verbose:
                            print(f"  Warning: Too many splits ({len(split_lines)} > {max_splits}). Retrying...")
                        last_oversized_split_lines = sorted(split_lines)
                        continue

                    return {
                        "should_split": data.get("should_split", False),
                        "reasoning": data.get("reasoning", ""),
                        "split_lines": split_lines,
                    }
            except Exception as e:
                if self.verbose:
                    print(f"Error in split decision attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    break
        
        # If all retries failed due to too many splits, subsample evenly
        if last_oversized_split_lines and len(last_oversized_split_lines) > max_splits:
            step = len(last_oversized_split_lines) / max_splits
            subsampled = [last_oversized_split_lines[int(i * step)] for i in range(max_splits)]
            if self.verbose:
                print(f"  Subsampling {len(last_oversized_split_lines)} splits down to {len(subsampled)}")
            return {
                "should_split": True,
                "reasoning": f"Subsampled from {len(last_oversized_split_lines)} LLM-suggested splits",
                "split_lines": subsampled,
            }
        
        return {
            "should_split": False,
            "reasoning": "Failed to get a valid LLM decision after retries, defaulting to no split",
            "split_lines": [],
        }
    
    def _find_split_char_positions_by_lines(self, window_text: str, split_lines: List[int]) -> List[int]:
        """Find character positions for split lines"""
        lines = window_text.split('\n')
        positions = []
        
        for line_num in split_lines:
            line_idx = line_num - 1
            
            if line_idx < 0 or line_idx >= len(lines):
                if self.verbose:
                    print(f"  Warning: Line {line_num} is outside window range (window has {len(lines)} lines)")
                continue
            
            char_pos = 0
            for i in range(line_idx):
                char_pos += len(lines[i]) + 1
            
            if char_pos > 0:
                positions.append(char_pos)
                if self.verbose:
                    print(f"  Line {line_num} -> char position {char_pos}")
            elif self.verbose:
                print(f"  Warning: Line {line_num} is at position 0, skipping")
        
        return sorted(set(positions))
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from text"""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return None
    
    # ========== Organization Methods (from GAMAgent) ==========
    
    def _organize_from_memorized_chunks_internal(
        self,
        memorized_chunks: List[BatchMemorizedChunk],
        context: str = "",
        callback: Optional[Callable] = None,
        taxonomy_batch_size: int = 50,
    ) -> BatchProcessingResult:
        """Internal method for organizing memorized chunks into GAM"""
        if self.verbose:
            print(f"\n{'='*40}")
            print("Stage 1: Organize Directory Structure (Taxonomy-based)")
            print(f"{'='*40}")
        
        result = BatchProcessingResult()
        result.memorized_chunks = memorized_chunks
        
        if callback:
            callback("memorized_chunks_loaded", {"count": len(memorized_chunks)})
        
        # Stage 1: Organize Directory Structure
        try:
            organization_plan = self._batch_organize_structure(
                memorized_chunks=memorized_chunks,
                context=context,
                batch_size=taxonomy_batch_size,
            )
            result.organization_plan = organization_plan
            
            if self.verbose:
                print(f"  ✓ Directories planned: {len(organization_plan.directories)}")
                for d in organization_plan.directories:
                    if d.chunk_indices:
                        print(f"    - {d.path} (leaf, chunks: {d.chunk_indices})")
                    else:
                        print(f"    - {d.path} (parent)")
            
            if callback:
                callback("stage_1_complete", {"directories": len(organization_plan.directories)})
                
        except Exception as e:
            result.errors.append(f"Stage 1 error: {e}")
            result.success = False
            if self.verbose:
                print(f"  ✗ Error: {e}")
            return result
        
        # Stage 2: Execute Organization Plan
        if self.verbose:
            print(f"\n{'='*40}")
            print("Stage 2: Execute Organization Plan (Parallel README)")
            print(f"{'='*40}")
        
        try:
            created_files = self._execute_batch_organization_parallel(
                memorized_chunks=memorized_chunks,
                organization_plan=organization_plan,
            )
            result.created_files = created_files
            
            if self.verbose:
                print(f"  ✓ Created {len(created_files)} files")
                for f in created_files[:10]:
                    print(f"    - {f}")
                if len(created_files) > 10:
                    print(f"    ... and {len(created_files) - 10} more")
            
            if callback:
                callback("stage_2_complete", {"created_files": len(created_files)})
                
        except Exception as e:
            result.errors.append(f"Stage 2 error: {e}")
            result.success = False
            if self.verbose:
                print(f"  ✗ Error: {e}")
            return result
        
        # Save
        if self.auto_save:
            # Reload tree from disk to sync with workspace changes
            self.tree = GAMTree.from_disk(self.tree.root_path, self.workspace)
            if self.verbose:
                print(f"\n✓ GAM synced from: {self.tree.root_path}")
        
        # Record history
        self.history.append({
            "mode": "organize_from_memorized",
            "chunks_count": len(memorized_chunks),
            "result": {
                "directories_count": len(result.organization_plan.directories) if result.organization_plan else 0,
                "files_created": len(result.created_files),
            },
        })
        
        return result
    
    def _execute_batch_organization_parallel(
        self,
        memorized_chunks: List[BatchMemorizedChunk],
        organization_plan: BatchOrganizationPlan,
    ) -> List[str]:
        """Execute organization plan with parallel README generation"""
        created_files = []
        chunk_map = {mc.index: mc for mc in memorized_chunks}
        dir_files: Dict[str, List[tuple]] = {}
        
        # Sort directories by depth
        sorted_dirs = sorted(
            organization_plan.directories,
            key=lambda d: d.path.count("/")
        )
        
        # Collect directories that have chunks or are ancestors of dirs with chunks
        dirs_with_chunks = set()
        for dir_node in sorted_dirs:
            if dir_node.chunk_indices:
                dirs_with_chunks.add(dir_node.path)
        
        needed_dirs = set()
        for path in dirs_with_chunks:
            needed_dirs.add(path)
            # Add all ancestors
            parts = path.split('/')
            for i in range(1, len(parts)):
                parent = '/'.join(parts[:i])
                if parent:
                    needed_dirs.add(parent)
        
        # Also include dirs that have children in needed_dirs (parent dirs)
        for dir_node in sorted_dirs:
            if dir_node.children:
                for child_name in dir_node.children:
                    child_path = f"{dir_node.path}/{child_name}"
                    if child_path in needed_dirs:
                        needed_dirs.add(dir_node.path)
                        break
        
        # Filter sorted_dirs to only needed ones
        sorted_dirs = [d for d in sorted_dirs if d.path in needed_dirs]
        
        # Create only needed directories
        for dir_node in sorted_dirs:
            dir_path = dir_node.path
            success = self._mkdir(dir_path, "")
            if self.verbose:
                status = "✓" if success else "✗"
                print(f"  {status} mkdir {dir_path}")
            dir_files[dir_path] = []
        
        # Create all chunk files (track created paths for dedup within this batch)
        batch_created_paths: set = set()
        for dir_node in sorted_dirs:
            if not dir_node.chunk_indices:
                continue
                
            dir_path = dir_node.path
            
            for chunk_idx in dir_node.chunk_indices:
                mc = chunk_map.get(chunk_idx)
                if mc is None:
                    continue
                
                chunk_title = re.sub(r'[^\w\-]', '_', mc.title.lower().strip())
                chunk_title = re.sub(r'_+', '_', chunk_title).strip('_') or f"chunk_{chunk_idx}"
                chunk_filename = f"{chunk_title}.md"
                chunk_filepath = f"{dir_path}/{chunk_filename}"
                
                # Check both tree and in-batch created paths for collisions
                def _path_exists(p: str) -> bool:
                    return self.tree.get_node(p) is not None or p in batch_created_paths
                
                if _path_exists(chunk_filepath):
                    counter = 1
                    while _path_exists(f"{dir_path}/{chunk_title}_{counter}.md"):
                        counter += 1
                    chunk_filename = f"{chunk_title}_{counter}.md"
                    chunk_filepath = f"{dir_path}/{chunk_filename}"
                
                now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                chunk_content = f"""# {mc.title.replace('_', ' ').title()}

> Created: {now_str}

{mc.original_content}
"""
                
                success = self._write_file(chunk_filepath, chunk_content)
                if self.verbose:
                    status = "✓" if success else "✗"
                    print(f"  {status} write {chunk_filepath}")
                
                if success:
                    created_files.append(chunk_filepath)
                    batch_created_paths.add(chunk_filepath)
                    dir_files[dir_path].append((chunk_filename, mc.title, mc.memory))
        
        # Build directory hierarchy
        dir_children: Dict[str, List[str]] = {d.path: d.children for d in organization_plan.directories}
        
        # Generate READMEs (bottom-up)
        sorted_dirs_deep_first = sorted(
            organization_plan.directories,
            key=lambda d: d.path.count("/"),
            reverse=True
        )
        
        depth_groups: Dict[int, List[DirectoryNode]] = {}
        for d in sorted_dirs_deep_first:
            depth = d.path.count("/")
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(d)
        
        dir_readmes: Dict[str, str] = {}
        
        if self.verbose:
            print("\n  Generating READMEs (parallel, bottom-up)...")
        
        for depth in sorted(depth_groups.keys(), reverse=True):
            dirs_at_depth = depth_groups[depth]
            
            if self.verbose:
                print(f"\n  Processing depth {depth}: {len(dirs_at_depth)} directories")
            
            if len(dirs_at_depth) > 1:
                readme_results = self._generate_readmes_batch(
                    dirs=dirs_at_depth,
                    dir_files=dir_files,
                    dir_children=dir_children,
                    dir_readmes=dir_readmes,
                )
                
                for dir_node, readme_content in zip(dirs_at_depth, readme_results):
                    dir_path = dir_node.path
                    dir_readmes[dir_path] = readme_content
                    
                    success = self._update_readme(dir_path, readme_content)
                    if self.verbose:
                        status = "✓" if success else "✗"
                        print(f"    {status} README {dir_path}")
                    
                    if success:
                        created_files.append(f"{dir_path}/README.md")
            else:
                for dir_node in dirs_at_depth:
                    dir_path = dir_node.path
                    files_in_dir = dir_files.get(dir_path, [])
                    
                    subdirs_info = []
                    for child_path in dir_children.get(dir_path, []):
                        child_name = child_path.split("/")[-1]
                        child_readme = dir_readmes.get(child_path, "")
                        subdirs_info.append((child_name, child_path, child_readme))
                    
                    readme_content = self._generate_directory_readme(
                        dir_name=dir_node.name,
                        dir_path=dir_path,
                        files_info=files_in_dir,
                        subdirs_info=subdirs_info,
                    )
                    
                    dir_readmes[dir_path] = readme_content
                    
                    success = self._update_readme(dir_path, readme_content)
                    if self.verbose:
                        status = "✓" if success else "✗"
                        print(f"    {status} README {dir_path}")
                    
                    if success:
                        created_files.append(f"{dir_path}/README.md")
        
        # Generate root README
        root_files = dir_files.get("/", [])
        root_subdirs = []
        for dir_node in organization_plan.directories:
            if dir_node.path.count("/") == 1:
                root_subdirs.append((dir_node.name, dir_node.path, dir_readmes.get(dir_node.path, "")))
        
        root_readme = self._generate_directory_readme(
            dir_name=self.tree.root.name or "Root",
            dir_path="/",
            files_info=root_files,
            subdirs_info=root_subdirs,
        )
        
        success = self._update_readme("/", root_readme)
        if self.verbose:
            status = "✓" if success else "✗"
            print(f"    {status} Root README")
        if success:
            created_files.append("/README.md")
        
        return created_files
    
    def _generate_readmes_batch(
        self,
        dirs: List[DirectoryNode],
        dir_files: Dict[str, List[tuple]],
        dir_children: Dict[str, List[str]],
        dir_readmes: Dict[str, str],
    ) -> List[str]:
        """Batch generate READMEs for multiple directories"""
        prompts = []
        
        for dir_node in dirs:
            dir_path = dir_node.path
            files_in_dir = dir_files.get(dir_path, [])
            
            subdirs_info = []
            for child_path in dir_children.get(dir_path, []):
                child_name = child_path.split("/")[-1]
                child_readme = dir_readmes.get(child_path, "")
                subdirs_info.append((child_name, child_path, child_readme))
            
            if files_in_dir:
                files_content_parts = []
                for filename, title, memory in files_in_dir:
                    files_content_parts.append(f"- **{filename}** (title: {title})\n  Memory: {memory}")
                files_content = "\n".join(files_content_parts)
            else:
                files_content = "(No files in this directory)"
            
            if subdirs_info:
                subdirs_content_parts = []
                for subdir_name, subdir_path, subdir_readme in subdirs_info:
                    subdir_desc = self._extract_description_from_readme(subdir_readme)
                    subdirs_content_parts.append(f"- **{subdir_name}/**\n  Description: {subdir_desc}")
                subdirs_content = "\n".join(subdirs_content_parts)
            else:
                subdirs_content = "(No subdirectories)"
            
            prompt = GAM_GENERATE_README_PROMPT.format(
                directory_name=dir_node.name.replace('_', ' ').title(),
                files_content=files_content,
                subdirs_content=subdirs_content,
            )
            prompts.append(prompt)
        
        try:
            results = self.generator.generate_batch(
                prompts=prompts,
                schema=GAM_GENERATE_README_SCHEMA,
            )
            
            readme_contents = []
            for dir_node, response in zip(dirs, results):
                data = response.get("json")
                if data is None:
                    data = self._extract_json(response.get("text", ""))
                
                files_in_dir = dir_files.get(dir_node.path, [])
                subdirs_info = []
                for child_path in dir_children.get(dir_node.path, []):
                    child_name = child_path.split("/")[-1]
                    child_readme = dir_readmes.get(child_path, "")
                    subdirs_info.append((child_name, child_path, child_readme))
                
                readme_content = self._build_readme_from_data(
                    dir_name=dir_node.name,
                    dir_path=dir_node.path,
                    files_info=files_in_dir,
                    subdirs_info=subdirs_info,
                    llm_data=data,
                )
                readme_contents.append(readme_content)
            
            return readme_contents
            
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Batch README generation failed: {e}, falling back to sequential")
            return [
                self._generate_directory_readme(
                    dir_name=d.name,
                    dir_path=d.path,
                    files_info=dir_files.get(d.path, []),
                    subdirs_info=[
                        (cp.split("/")[-1], cp, dir_readmes.get(cp, ""))
                        for cp in dir_children.get(d.path, [])
                    ],
                )
                for d in dirs
            ]
    
    def _build_readme_from_data(
        self,
        dir_name: str,
        dir_path: str,
        files_info: List[tuple],
        subdirs_info: List[tuple],
        llm_data: Optional[Dict],
    ) -> str:
        """Build README from LLM response data"""
        if llm_data:
            description = llm_data.get("description", "")
            items = llm_data.get("items", [])
        else:
            description = f"This directory contains organized content related to {dir_name}."
            items = []
        
        readme_parts = []
        
        display_name = dir_name.replace('_', ' ').title()
        readme_parts.append(f"# {display_name}\n")
        readme_parts.append(f"{description}\n")
        
        if subdirs_info or files_info:
            readme_parts.append("## Contents\n")
            
            item_lookup = {item.get("name", ""): item for item in items}
            
            if subdirs_info:
                readme_parts.append("### Subdirectories\n")
                for subdir_name, subdir_path, _ in subdirs_info:
                    item_info = self._fuzzy_lookup_item(subdir_name, item_lookup)
                    brief = item_info.get("brief", f"Contains content related to {subdir_name}")
                    detailed = item_info.get("detailed", "")
                    
                    readme_parts.append(f"#### [{subdir_name}/](./{subdir_name}/)\n")
                    readme_parts.append(f"**{brief}**\n")
                    if detailed:
                        readme_parts.append(f"\n{detailed}\n")
                    readme_parts.append("")
            
            if files_info:
                readme_parts.append("### Files\n")
                for filename, title, memory in files_info:
                    item_info = self._fuzzy_lookup_item(filename, item_lookup)
                    brief = item_info.get("brief", memory)
                    detailed = item_info.get("detailed", "")
                    
                    display_title = title.replace('_', ' ').title()
                    readme_parts.append(f"#### [{display_title}](./{filename})\n")
                    readme_parts.append(f"**{brief}**\n")
                    if detailed:
                        readme_parts.append(f"\n{detailed}\n")
                    readme_parts.append("")
        
        # For root directory, show full recursive tree; for subdirs, show flat structure
        if dir_path == "/":
            structure = self._generate_full_recursive_tree(
                dir_path="/", prefix="", show_timestamps=True,
            )
        else:
            structure = self._generate_structure_tree(
                dir_path, files_info, subdirs_info,
            )
        if structure:
            readme_parts.append("## Directory Structure\n")
            readme_parts.append(f"```\n{structure}\n```\n")
        
        return "\n".join(readme_parts)
    
    def _batch_organize_structure(
        self,
        memorized_chunks: List[BatchMemorizedChunk],
        context: str = "",
        batch_size: int = 50,
    ) -> BatchOrganizationPlan:
        """Organize chunks into directory structure using taxonomy approach"""
        if self.verbose:
            print(f"\n  --- Taxonomy-based Organization ---")
            print(f"  Total chunks: {len(memorized_chunks)}")
            print(f"  Batch size: {batch_size}")
        
        # Generate taxonomy from TLDRs
        taxonomy = self._generate_taxonomy_from_tldrs(
            memorized_chunks=memorized_chunks,
            context=context,
            batch_size=batch_size,
        )
        
        if self.verbose:
            print(f"\n  Taxonomy generated with {len(taxonomy.nodes)} directories:")
            print(taxonomy.to_tree_string())
        
        # Assign chunks to taxonomy
        assignments = self._assign_chunks_to_taxonomy(
            memorized_chunks=memorized_chunks,
            taxonomy=taxonomy,
        )
        
        if self.verbose:
            print(f"\n  Assigned {len(assignments)} chunks to taxonomy")
        
        # Build directory nodes
        path_to_chunks: Dict[str, List[int]] = {}
        for assignment in assignments:
            path = assignment.assigned_path
            if path not in path_to_chunks:
                path_to_chunks[path] = []
            path_to_chunks[path].append(assignment.chunk_index)
        
        directories = []
        for node in taxonomy.nodes:
            directories.append(DirectoryNode(
                path=node.path,
                name=node.name,
                description=node.description,
                children=node.children,
                chunk_indices=path_to_chunks.get(node.path, []),
            ))
        
        # Cleanup empty directories
        directories = self._cleanup_empty_directories(directories)
        
        if self.verbose:
            print(f"\n  Final structure has {len(directories)} directories")
            for d in directories:
                if d.chunk_indices:
                    print(f"    - {d.path} (chunks: {d.chunk_indices})")
                else:
                    print(f"    - {d.path} (parent)")
        
        return BatchOrganizationPlan(
            directories=directories,
            reasoning=taxonomy.reasoning,
        )
    
    def _generate_taxonomy_from_tldrs(
        self,
        memorized_chunks: List[BatchMemorizedChunk],
        context: str = "",
        batch_size: int = 50,
    ) -> TaxonomyTree:
        """Generate taxonomy from TLDRs in batches"""
        tldrs = []
        for mc in memorized_chunks:
            tldr = mc.tldr if mc.tldr else mc.memory.split('\n')[0][:200]
            tldrs.append({
                'index': mc.index,
                'title': mc.title,
                'tldr': tldr,
            })
        
        total_batches = (len(tldrs) + batch_size - 1) // batch_size
        taxonomy = TaxonomyTree(nodes=[], reasoning="", changes_history=[])
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(tldrs))
            batch_tldrs = tldrs[start_idx:end_idx]
            
            if self.verbose:
                print(f"\n  Processing TLDR batch {batch_num + 1}/{total_batches} (chunks {start_idx}-{end_idx - 1})")
            
            tldr_list_text = "\n".join([
                f"- **Chunk #{t['index']}** ({t['title']}): {t['tldr']}"
                for t in batch_tldrs
            ])
            
            if batch_num == 0:
                taxonomy = self._generate_initial_taxonomy(
                    tldr_list=tldr_list_text,
                    context=context,
                    batch_num=batch_num + 1,
                    total_batches=total_batches,
                )
            else:
                taxonomy = self._adjust_taxonomy(
                    existing_taxonomy=taxonomy,
                    tldr_list=tldr_list_text,
                    context=context,
                    batch_num=batch_num + 1,
                    total_batches=total_batches,
                )
        
        return taxonomy
    
    def _generate_initial_taxonomy(
        self,
        tldr_list: str,
        context: str,
        batch_num: int,
        total_batches: int,
    ) -> TaxonomyTree:
        """Generate initial taxonomy from first batch"""
        prompt = GENERATE_TAXONOMY_FROM_TLDR_PROMPT.format(
            tldr_list=tldr_list,
            context=context or "No additional context.",
            batch_num=batch_num,
            total_batches=total_batches,
        )
        
        try:
            response = self.generator.generate_single(
                prompt=prompt,
                schema=GENERATE_TAXONOMY_SCHEMA,
            )
            
            data = response.get("json")
            if data is None:
                data = self._extract_json(response.get("text", ""))
            
            if data:
                nodes = []
                for node_data in data.get("taxonomy", []):
                    nodes.append(TaxonomyNode(
                        path=node_data.get("path", "/"),
                        name=node_data.get("name", "unnamed"),
                        description=node_data.get("description", ""),
                        children=node_data.get("children", []),
                    ))
                
                return TaxonomyTree(
                    nodes=nodes,
                    reasoning=data.get("reasoning", ""),
                    changes_history=[f"Initial taxonomy created with {len(nodes)} directories"],
                )
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Error generating initial taxonomy: {e}")
        
        return TaxonomyTree(
            nodes=[TaxonomyNode(
                path="/content",
                name="content",
                description="All content",
                children=[],
            )],
            reasoning="Fallback taxonomy",
            changes_history=["Fallback: created single content directory"],
        )
    
    def _adjust_taxonomy(
        self,
        existing_taxonomy: TaxonomyTree,
        tldr_list: str,
        context: str,
        batch_num: int,
        total_batches: int,
    ) -> TaxonomyTree:
        """Adjust existing taxonomy with new TLDRs"""
        prompt = ADJUST_TAXONOMY_WITH_TLDR_PROMPT.format(
            existing_taxonomy=existing_taxonomy.to_tree_string(),
            tldr_list=tldr_list,
            context=context or "No additional context.",
            batch_num=batch_num,
            total_batches=total_batches,
        )
        
        try:
            response = self.generator.generate_single(
                prompt=prompt,
                schema=ADJUST_TAXONOMY_SCHEMA,
            )
            
            data = response.get("json")
            if data is None:
                data = self._extract_json(response.get("text", ""))
            
            if data:
                nodes = []
                for node_data in data.get("taxonomy", []):
                    nodes.append(TaxonomyNode(
                        path=node_data.get("path", "/"),
                        name=node_data.get("name", "unnamed"),
                        description=node_data.get("description", ""),
                        children=node_data.get("children", []),
                    ))
                
                changes_made = data.get("changes_made", [])
                new_history = existing_taxonomy.changes_history + [
                    f"Batch {batch_num}: {', '.join(changes_made) if changes_made else 'no changes'}"
                ]
                
                return TaxonomyTree(
                    nodes=nodes,
                    reasoning=data.get("reasoning", existing_taxonomy.reasoning),
                    changes_history=new_history,
                )
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Error adjusting taxonomy: {e}")
        
        existing_taxonomy.changes_history.append(f"Batch {batch_num}: error, no changes made")
        return existing_taxonomy
    
    def _assign_chunks_to_taxonomy(
        self,
        memorized_chunks: List[BatchMemorizedChunk],
        taxonomy: TaxonomyTree,
    ) -> List[ChunkAssignmentResult]:
        """Assign chunks to taxonomy directories in parallel"""
        if self.verbose:
            print(f"\n  Assigning {len(memorized_chunks)} chunks to taxonomy (parallel)...")
        
        taxonomy_tree_str = taxonomy.to_tree_string()
        
        prompts = []
        for mc in memorized_chunks:
            tldr = mc.tldr if mc.tldr else mc.memory.split('\n')[0][:200]
            memory_preview = mc.memory[:500] + "..." if len(mc.memory) > 500 else mc.memory
            
            prompt = ASSIGN_CHUNK_TO_TAXONOMY_PROMPT.format(
                taxonomy_tree=taxonomy_tree_str,
                chunk_index=mc.index,
                chunk_title=mc.title,
                chunk_tldr=tldr,
                chunk_memory_preview=memory_preview,
            )
            prompts.append(prompt)
        
        try:
            results = self.generator.generate_batch(
                prompts=prompts,
                schema=ASSIGN_CHUNK_SCHEMA,
            )
            
            assignments = []
            valid_paths = set(taxonomy.get_all_paths())
            
            for mc, response in zip(memorized_chunks, results):
                data = response.get("json")
                if data is None:
                    data = self._extract_json(response.get("text", ""))
                
                if data:
                    assigned_path = data.get("assigned_path", "/content")
                    if assigned_path not in valid_paths:
                        if self.verbose:
                            print(f"    Warning: Invalid path '{assigned_path}' for chunk {mc.index}, using fallback")
                        assigned_path = self._find_best_matching_path(assigned_path, valid_paths)
                    
                    assignments.append(ChunkAssignmentResult(
                        chunk_index=mc.index,
                        assigned_path=assigned_path,
                        reasoning=data.get("reasoning", ""),
                        confidence=data.get("confidence", 1.0),
                    ))
                else:
                    leaf_paths = taxonomy.get_leaf_paths()
                    fallback_path = leaf_paths[0] if leaf_paths else "/content"
                    assignments.append(ChunkAssignmentResult(
                        chunk_index=mc.index,
                        assigned_path=fallback_path,
                        reasoning="Fallback assignment",
                        confidence=0.5,
                    ))
            
            return assignments
            
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Batch assignment failed: {e}, using sequential fallback")
            
            leaf_paths = taxonomy.get_leaf_paths()
            fallback_path = leaf_paths[0] if leaf_paths else "/content"
            
            return [
                ChunkAssignmentResult(
                    chunk_index=mc.index,
                    assigned_path=fallback_path,
                    reasoning="Fallback due to batch error",
                    confidence=0.3,
                )
                for mc in memorized_chunks
            ]
    
    def _find_best_matching_path(self, invalid_path: str, valid_paths: Set[str]) -> str:
        """Find best matching valid path for invalid path"""
        invalid_parts = invalid_path.strip('/').split('/')
        
        best_match = None
        best_score = 0
        
        for valid_path in valid_paths:
            valid_parts = valid_path.strip('/').split('/')
            score = sum(1 for p in invalid_parts if p in valid_parts)
            if score > best_score:
                best_score = score
                best_match = valid_path
        
        if best_match:
            return best_match
        
        for p in valid_paths:
            return p
        
        return "/content"
    
    def _cleanup_empty_directories(
        self,
        directories: List[DirectoryNode],
    ) -> List[DirectoryNode]:
        """Remove empty directories"""
        if self.verbose:
            print(f"\n  Cleaning up empty directories...")
        
        path_map = {d.path: d for d in directories}
        has_content: Dict[str, bool] = {}
        
        sorted_dirs = sorted(directories, key=lambda d: -d.path.count('/'))
        
        for d in sorted_dirs:
            if d.chunk_indices:
                has_content[d.path] = True
            else:
                children_have_content = any(
                    has_content.get(child_path, False)
                    for child_path in d.children
                )
                has_content[d.path] = children_have_content
        
        result = []
        removed_paths = set()
        
        for d in directories:
            if has_content.get(d.path, False):
                updated_children = [
                    cp for cp in d.children
                    if has_content.get(cp, False)
                ]
                result.append(DirectoryNode(
                    path=d.path,
                    name=d.name,
                    description=d.description,
                    children=updated_children,
                    chunk_indices=d.chunk_indices,
                ))
            else:
                removed_paths.add(d.path)
        
        if self.verbose and removed_paths:
            print(f"    Removed {len(removed_paths)} empty directories: {removed_paths}")
        
        return result
    
    def _generate_directory_readme(
        self,
        dir_name: str,
        dir_path: str,
        files_info: List[tuple],
        subdirs_info: List[tuple],
    ) -> str:
        """Generate README for a directory using LLM"""
        if files_info:
            files_content_parts = []
            for filename, title, memory in files_info:
                files_content_parts.append(f"- **{filename}** (title: {title})\n  Memory: {memory}")
            files_content = "\n".join(files_content_parts)
        else:
            files_content = "(No files in this directory)"
        
        if subdirs_info:
            subdirs_content_parts = []
            for subdir_name, subdir_path, subdir_readme in subdirs_info:
                subdir_desc = self._extract_description_from_readme(subdir_readme)
                subdirs_content_parts.append(f"- **{subdir_name}/**\n  Description: {subdir_desc}")
            subdirs_content = "\n".join(subdirs_content_parts)
        else:
            subdirs_content = "(No subdirectories)"
        
        prompt = GAM_GENERATE_README_PROMPT.format(
            directory_name=dir_name.replace('_', ' ').title(),
            files_content=files_content,
            subdirs_content=subdirs_content,
        )
        
        try:
            response = self.generator.generate_single(
                prompt=prompt,
                schema=GAM_GENERATE_README_SCHEMA,
            )
            
            data = response.get("json")
            if data is None:
                data = self._extract_json(response.get("text", ""))
            
            if data:
                description = data.get("description", "")
                items = data.get("items", [])
            else:
                description = f"This directory contains organized content related to {dir_name}."
                items = []
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Error generating README for {dir_path}: {e}")
            description = f"This directory contains organized content related to {dir_name}."
            items = []
        
        readme_parts = []
        
        display_name = dir_name.replace('_', ' ').title()
        readme_parts.append(f"# {display_name}\n")
        readme_parts.append(f"{description}\n")
        
        if subdirs_info or files_info:
            readme_parts.append("## Contents\n")
            
            item_lookup = {item.get("name", ""): item for item in items}
            
            if subdirs_info:
                readme_parts.append("### Subdirectories\n")
                for subdir_name, subdir_path, _ in subdirs_info:
                    item_info = self._fuzzy_lookup_item(subdir_name, item_lookup)
                    brief = item_info.get("brief", f"Contains content related to {subdir_name}")
                    detailed = item_info.get("detailed", "")
                    
                    readme_parts.append(f"#### [{subdir_name}/](./{subdir_name}/)\n")
                    readme_parts.append(f"**{brief}**\n")
                    if detailed:
                        readme_parts.append(f"\n{detailed}\n")
                    readme_parts.append("")
            
            if files_info:
                readme_parts.append("### Files\n")
                for filename, title, memory in files_info:
                    item_info = self._fuzzy_lookup_item(filename, item_lookup)
                    brief = item_info.get("brief", memory)
                    detailed = item_info.get("detailed", "")
                    
                    display_title = title.replace('_', ' ').title()
                    readme_parts.append(f"#### [{display_title}](./{filename})\n")
                    readme_parts.append(f"**{brief}**\n")
                    if detailed:
                        readme_parts.append(f"\n{detailed}\n")
                    readme_parts.append("")
        
        # For root directory, show full recursive tree; for subdirs, show flat structure
        if dir_path == "/":
            structure = self._generate_full_recursive_tree(
                dir_path="/", prefix="", show_timestamps=True,
            )
        else:
            structure = self._generate_structure_tree(
                dir_path, files_info, subdirs_info,
            )
        if structure:
            readme_parts.append("## Directory Structure\n")
            readme_parts.append(f"```\n{structure}\n```\n")
        
        return "\n".join(readme_parts)
    
    def _fuzzy_lookup_item(self, name: str, item_lookup: Dict[str, Dict]) -> Dict:
        """
        模糊匹配 item_lookup 中的条目。
        
        LLM 返回的 name 可能与实际文件名不完全匹配（如缺少 .md 后缀、
        使用 title 格式等），需要依次尝试多种匹配策略。
        
        匹配优先级：
        1. 精确匹配
        2. 去掉 .md 后缀匹配
        3. 大小写不敏感匹配
        4. 将 name 标准化（下划线/空格统一、去后缀）后匹配
        """
        # 1. 精确匹配
        if name in item_lookup:
            return item_lookup[name]
        
        # 2. 去掉 .md 后缀匹配（LLM 可能返回不带后缀的名字）
        name_no_ext = name.rsplit('.', 1)[0] if '.' in name else name
        if name_no_ext in item_lookup:
            return item_lookup[name_no_ext]
        
        # 3. 大小写不敏感匹配
        name_lower = name.lower()
        for key, value in item_lookup.items():
            if key.lower() == name_lower:
                return value
        
        # 4. 标准化匹配：统一下划线/空格/连字符，去后缀，再比较
        def _normalize(s: str) -> str:
            s = s.rsplit('.', 1)[0] if '.' in s else s
            return re.sub(r'[\s_\-]+', '_', s.strip()).lower()
        
        name_norm = _normalize(name)
        for key, value in item_lookup.items():
            if _normalize(key) == name_norm:
                return value
        
        return {}
    
    def _extract_description_from_readme(self, readme: str) -> str:
        """Extract description from README"""
        if not readme:
            return ""
        
        lines = readme.strip().split("\n")
        description_lines = []
        started = False
        
        for line in lines:
            if line.startswith("# "):
                started = True
                continue
            if started and line.startswith("## "):
                break
            if started and line.strip():
                description_lines.append(line.strip())
            elif started and description_lines:
                break
        
        return " ".join(description_lines)
    
    def _generate_structure_tree(
        self,
        dir_path: str,
        files_info: List[tuple],
        subdirs_info: List[tuple],
        show_timestamps: bool = False,
    ) -> str:
        """
        Generate directory structure tree (flat, single level)
        
        Args:
            dir_path: Current directory path
            files_info: List of (filename, title, memory)
            subdirs_info: List of (subdir_name, subdir_path, subdir_readme)
            show_timestamps: Whether to show creation timestamps next to each entry
        """
        if not files_info and not subdirs_info:
            return ""
        
        lines = []
        items = []
        
        for subdir_name, _, _ in subdirs_info:
            items.append((subdir_name + "/", True, subdir_name))
        
        for filename, _, _ in files_info:
            items.append((filename, False, filename))
        
        items.sort(key=lambda x: (not x[1], x[0].lower()))
        
        for i, (name, is_dir, entry_name) in enumerate(items):
            is_last = i == len(items) - 1
            prefix = "└── " if is_last else "├── "
            line = f"{prefix}{name}"
            
            if show_timestamps:
                # Construct full path for timestamp lookup
                if dir_path == "/":
                    entry_path = f"/{entry_name}"
                else:
                    entry_path = f"{dir_path}/{entry_name}"
                ts = self._get_path_created_time(entry_path)
                # Pad for alignment
                line = f"{line:<40} Created: {ts}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def _generate_full_recursive_tree(
        self,
        dir_path: str = "/",
        prefix: str = "",
        show_timestamps: bool = True,
    ) -> str:
        """
        Recursively generate the full directory structure tree from the filesystem.
        Walks the actual filesystem to include ALL nested files and subdirectories.
        
        Args:
            dir_path: Directory path relative to workspace root
            prefix: Indentation prefix for tree drawing characters
            show_timestamps: Whether to show creation timestamps next to each entry
        
        Returns:
            Full recursive tree structure string
        """
        abs_path = str(Path(self.workspace.root_path) / dir_path.lstrip('/'))
        
        # List directory contents (exclude hidden files and README.md)
        output, exit_code = self.workspace.run(
            f"ls -1 '{abs_path}' 2>/dev/null | grep -v '^\\..*' | grep -v '^README\\.md$' | sort"
        )
        
        if exit_code != "0" or not output.strip():
            return ""
        
        entries = [e.strip() for e in output.strip().split('\n') if e.strip()]
        if not entries:
            return ""
        
        # Classify entries as dirs or files
        items = []
        for entry in entries:
            entry_abs = str(Path(abs_path) / entry)
            check_output, _ = self.workspace.run(f"test -d '{entry_abs}' && echo 'dir' || echo 'file'")
            is_dir = 'dir' in check_output
            items.append((entry, is_dir))
        
        # Sort: directories first, then files, alphabetically
        items.sort(key=lambda x: (not x[1], x[0].lower()))
        
        lines = []
        for i, (entry, is_dir) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            
            display_name = entry + "/" if is_dir else entry
            line = f"{prefix}{connector}{display_name}"
            
            if show_timestamps:
                if dir_path == "/":
                    entry_path = f"/{entry}"
                else:
                    entry_path = f"{dir_path.rstrip('/')}/{entry}"
                ts = self._get_path_created_time(entry_path)
                line = f"{line:<60} Created: {ts}"
            
            lines.append(line)
            
            # Recurse into subdirectories
            if is_dir:
                child_prefix = prefix + ("    " if is_last else "│   ")
                if dir_path == "/":
                    child_dir = f"/{entry}"
                else:
                    child_dir = f"{dir_path.rstrip('/')}/{entry}"
                subtree = self._generate_full_recursive_tree(
                    child_dir, child_prefix, show_timestamps
                )
                if subtree:
                    lines.append(subtree)
        
        return "\n".join(lines)
    
    # ========== Incremental Add Helper Methods ==========
    
    def _build_content_summary(self, batch_memorized: List[BatchMemorizedChunk]) -> str:
        """Build summary of new content"""
        summaries = []
        for mc in batch_memorized[:10]:
            tldr = mc.tldr if mc.tldr else mc.memory.split('\n')[0][:150]
            summaries.append(f"- **{mc.title}**: {tldr}")
        
        if len(batch_memorized) > 10:
            summaries.append(f"... and {len(batch_memorized) - 10} more chunks")
        
        return "\n".join(summaries)
    
    def _check_belongs_to_existing(
        self,
        root_readme: str,
        existing_tree: str,
        new_content_summary: str,
    ) -> Dict[str, Any]:
        """Check if new content belongs to existing structure"""
        prompt = SHOULD_ADD_TO_EXISTING_PROMPT.format(
            existing_readme=root_readme or "(No README yet)",
            existing_tree=existing_tree or "(Empty tree)",
            new_content_summary=new_content_summary,
        )
        
        try:
            response = self.generator.generate_single(
                prompt=prompt,
                schema=SHOULD_ADD_TO_EXISTING_SCHEMA,
            )
            
            data = response.get("json")
            if data is None:
                data = self._extract_json(response.get("text", ""))
            
            if data:
                return data
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Error checking content belonging: {e}")
        
        return {
            "reasoning": "Default: assuming content belongs to existing structure",
            "belongs_to_existing": True,
            "suggested_location": "/",
        }
    
    def _add_to_existing_structure(
        self,
        memorized_chunks: List[MemorizedChunk],
        batch_memorized: List[BatchMemorizedChunk],
        suggested_location: str,
        context: str,
    ) -> AddChunksResult:
        """Add new content to existing structure"""
        result = AddChunksResult()
        
        if self.verbose:
            print(f"\n--- Adding to Existing Structure ---")
            print(f"  Suggested location: {suggested_location}")
        
        existing_tree = self.get_tree_view()
        directory_descriptions = self._get_directory_descriptions()
        
        try:
            add_plan = self._plan_chunk_additions(
                batch_memorized=batch_memorized,
                existing_tree=existing_tree,
                directory_descriptions=directory_descriptions,
            )
            result.assignments = add_plan.get("assignments", [])
            result.new_directories = add_plan.get("new_directories", [])
            result.affected_paths = add_plan.get("affected_paths", [])
            result.reasoning = add_plan.get("reasoning", "")
        except Exception as e:
            result.errors.append(f"Planning error: {e}")
            result.success = False
            return result
        
        # Create new directories
        for new_dir in result.new_directories:
            success = self._mkdir(new_dir.path, "")
            if self.verbose:
                status = "✓" if success else "✗"
                print(f"  {status} mkdir {new_dir.path}")
        
        # Create chunk files (track created paths for dedup within this batch)
        chunk_map = {mc.index: mc for mc in batch_memorized}
        batch_created_files: set = set()
        
        for assignment in result.assignments:
            mc = chunk_map.get(assignment.chunk_index)
            if mc is None:
                continue
            
            target_path = assignment.target_path
            if assignment.action in ["create_subdir", "create_toplevel"] and assignment.new_dir_name:
                target_path = f"{assignment.target_path}/{assignment.new_dir_name}"
            
            filepath = self._create_chunk_file(mc, target_path, created_files=batch_created_files)
            if filepath:
                result.created_files.append(filepath)
                if target_path not in result.affected_paths:
                    result.affected_paths.append(target_path)
        
        # Reload tree from disk before generating READMEs so that the tree
        # reflects the newly created directories and files.
        self.tree = GAMTree.from_disk(self.tree.root_path, self.workspace)
        
        # Update READMEs
        if self.verbose:
            print(f"\n--- Updating Affected READMEs ---")
        self._update_affected_readmes(result.affected_paths)
        
        return result
    
    def _expand_with_new_topic(
        self,
        memorized_chunks: List[MemorizedChunk],
        batch_memorized: List[BatchMemorizedChunk],
        new_topic_name: str,
        new_topic_description: str,
        context: str,
    ) -> AddChunksResult:
        """Expand with new topic directory"""
        result = AddChunksResult()
        
        if self.verbose:
            print(f"\n--- Expanding with New Topic ---")
            print(f"  New topic: {new_topic_name}")
        
        existing_children = [child.name for child in self.tree.root.children]
        
        old_content_dir = None
        if existing_children:
            old_topic_name = self._extract_topic_name_from_readme(self.tree.root.summary or "")
            if not old_topic_name:
                old_topic_name = "existing_content"
            
            old_content_dir = f"/{old_topic_name}"
            
            if self.verbose:
                print(f"  Moving existing content to: {old_content_dir}")
            
            self._mkdir(old_content_dir, self.tree.root.summary or "")
            result.new_directories.append(NewDirectoryInfo(
                path=old_content_dir,
                name=old_topic_name,
                description="Previous content",
            ))
            
            for child_name in existing_children:
                old_path = f"/{child_name}"
                new_path = f"{old_content_dir}/{child_name}"
                self._mv(old_path, new_path)
                if self.verbose:
                    print(f"    Moved {old_path} -> {new_path}")
        
        # Create new topic directory
        new_topic_path = f"/{new_topic_name}"
        self._mkdir(new_topic_path, "")
        result.new_directories.append(NewDirectoryInfo(
            path=new_topic_path,
            name=new_topic_name,
            description=new_topic_description,
        ))
        
        if self.verbose:
            print(f"  Created new topic directory: {new_topic_path}")
        
        # Generate taxonomy for new topic
        taxonomy = self._generate_taxonomy_from_tldrs(
            memorized_chunks=batch_memorized,
            context=context,
            batch_size=50,
        )
        
        assignments = self._assign_chunks_to_taxonomy(
            memorized_chunks=batch_memorized,
            taxonomy=taxonomy,
        )
        
        # Build structure
        path_to_chunks: Dict[str, List[int]] = {}
        for assignment in assignments:
            adjusted_path = f"{new_topic_path}{assignment.assigned_path}"
            if adjusted_path not in path_to_chunks:
                path_to_chunks[adjusted_path] = []
            path_to_chunks[adjusted_path].append(assignment.chunk_index)
        
        # Collect all needed directories: paths that have chunks + their ancestors
        needed_dirs = set()
        for path in path_to_chunks:
            needed_dirs.add(path)
            # Also add all parent directories up to new_topic_path
            parts = path.split('/')
            for i in range(1, len(parts)):
                parent = '/'.join(parts[:i])
                if parent and parent != new_topic_path:
                    needed_dirs.add(parent)
        needed_dirs.add(new_topic_path)
        
        # Create only needed directories and files
        created_dirs = set()
        chunk_map = {mc.index: mc for mc in batch_memorized}
        
        for node in taxonomy.nodes:
            adjusted_path = f"{new_topic_path}{node.path}"
            if adjusted_path in needed_dirs and adjusted_path not in created_dirs:
                self._mkdir(adjusted_path, "")
                created_dirs.add(adjusted_path)
        
        batch_created_files: set = set()
        for path, chunk_indices in path_to_chunks.items():
            if path not in created_dirs:
                self._mkdir(path, "")
                created_dirs.add(path)
            
            for chunk_idx in chunk_indices:
                mc = chunk_map.get(chunk_idx)
                if mc:
                    filepath = self._create_chunk_file(mc, path, created_files=batch_created_files)
                    if filepath:
                        result.created_files.append(filepath)
        
        # Reload tree from disk before generating READMEs so that the tree
        # reflects all the filesystem changes (moved dirs, new dirs/files).
        # Without this reload, _update_affected_readmes uses a stale in-memory
        # tree and generates incorrect READMEs (missing the old topic or new topic).
        self.tree = GAMTree.from_disk(self.tree.root_path, self.workspace)
        
        # Update READMEs
        new_topic_paths = list(created_dirs)
        new_topic_paths.sort(key=lambda p: -p.count('/'))
        
        if self.verbose:
            print(f"\n--- Updating New Topic READMEs ---")
        self._update_affected_readmes(new_topic_paths)
        
        result.affected_paths = ["/"]
        if old_content_dir:
            result.affected_paths.append(old_content_dir)
        result.affected_paths.append(new_topic_path)
        
        if self.verbose:
            print(f"\n--- Updating Root README ---")
        self._update_affected_readmes(["/"])
        
        return result
    
    def _extract_topic_name_from_readme(self, readme: str) -> str:
        """Extract topic name from README"""
        if not readme:
            return ""
        
        lines = readme.strip().split('\n')
        for line in lines:
            if line.startswith('# '):
                title = line[2:].strip()
                topic_name = re.sub(r'[^\w\s]', '', title.lower())
                topic_name = re.sub(r'\s+', '_', topic_name)
                return topic_name[:50]
        
        return ""
    
    def _create_chunk_file(
        self,
        mc: BatchMemorizedChunk,
        target_path: str,
        created_files: Optional[set] = None,
    ) -> Optional[str]:
        """Create a single chunk file
        
        Args:
            mc: The memorized chunk to create a file for
            target_path: Directory path to create file in
            created_files: Optional set tracking files created in current batch,
                          used for dedup when tree hasn't been reloaded yet.
                          Will be updated in-place if provided.
        """
        chunk_title = re.sub(r'[^\w\-]', '_', mc.title.lower().strip())
        chunk_title = re.sub(r'_+', '_', chunk_title).strip('_') or f"chunk_{mc.index}"
        chunk_filename = f"{chunk_title}.md"
        chunk_filepath = f"{target_path}/{chunk_filename}"
        
        # Check both tree and in-batch created_files set for collisions
        def _path_exists(path: str) -> bool:
            if self.tree.get_node(path) is not None:
                return True
            if created_files is not None and path in created_files:
                return True
            return False
        
        if _path_exists(chunk_filepath):
            counter = 1
            while _path_exists(f"{target_path}/{chunk_title}_{counter}.md"):
                counter += 1
            chunk_filename = f"{chunk_title}_{counter}.md"
            chunk_filepath = f"{target_path}/{chunk_filename}"
        
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        chunk_content = f"""# {mc.title.replace('_', ' ').title()}

> Created: {now_str}

{mc.original_content}
"""
        
        success = self._write_file(chunk_filepath, chunk_content)
        if self.verbose:
            status = "✓" if success else "✗"
            print(f"  {status} write {chunk_filepath}")
        
        if success and created_files is not None:
            created_files.add(chunk_filepath)
        
        return chunk_filepath if success else None
    
    def _plan_chunk_additions(
        self,
        batch_memorized: List[BatchMemorizedChunk],
        existing_tree: str,
        directory_descriptions: str,
    ) -> Dict[str, Any]:
        """Plan how to add new chunks"""
        new_chunks_info = []
        for mc in batch_memorized:
            tldr = mc.tldr if mc.tldr else mc.memory.split('\n')[0][:200]
            new_chunks_info.append(f"""- **Chunk #{mc.index}**: {mc.title}
  TLDR: {tldr}
  Memory: {mc.memory[:300]}...""")
        
        prompt = ADD_CHUNKS_TO_EXISTING_TAXONOMY_PROMPT.format(
            existing_tree=existing_tree,
            directory_descriptions=directory_descriptions,
            new_chunks_info="\n".join(new_chunks_info),
        )
        
        try:
            response = self.generator.generate_single(
                prompt=prompt,
                schema=ADD_CHUNKS_SCHEMA,
            )
            
            data = response.get("json")
            if data is None:
                data = self._extract_json(response.get("text", ""))
            
            if data:
                assignments = []
                for a in data.get("assignments", []):
                    assignments.append(ChunkAddAssignment(
                        chunk_index=a.get("chunk_index", 0),
                        chunk_title=a.get("chunk_title", ""),
                        action=a.get("action", "use_existing"),
                        target_path=a.get("target_path", "/"),
                        new_dir_name=a.get("new_dir_name"),
                        new_dir_description=a.get("new_dir_description"),
                    ))
                
                new_directories = []
                for d in data.get("new_directories", []):
                    new_directories.append(NewDirectoryInfo(
                        path=d.get("path", "/"),
                        name=d.get("name", ""),
                        description=d.get("description", ""),
                        parent_path=d.get("parent_path"),
                    ))
                
                return {
                    "reasoning": data.get("reasoning", ""),
                    "assignments": assignments,
                    "new_directories": new_directories,
                    "affected_paths": data.get("affected_paths", []),
                }
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Error planning chunk additions: {e}")
        
        return {
            "reasoning": "Fallback: placing all chunks in /content",
            "assignments": [
                ChunkAddAssignment(
                    chunk_index=mc.index,
                    chunk_title=mc.title,
                    action="use_existing",
                    target_path="/content",
                )
                for mc in batch_memorized
            ],
            "new_directories": [],
            "affected_paths": ["/content"],
        }
    
    def _get_directory_descriptions(self) -> str:
        """Get descriptions of all directories"""
        descriptions = []
        
        def collect_descriptions(node, path="/"):
            if node.summary:
                desc = self._extract_description_from_readme(node.summary)
                descriptions.append(f"- **{path}**: {desc[:200]}...")
            
            for child_node in node.children:
                child_name = child_node.name
                child_path = f"{path}/{child_name}" if path != "/" else f"/{child_name}"
                collect_descriptions(child_node, child_path)
        
        collect_descriptions(self.tree.root)
        return "\n".join(descriptions) if descriptions else "(No directory descriptions available)"
    
    def _reorganize_taxonomy(self) -> ReorganizeResult:
        """Reorganize directory structure"""
        result = ReorganizeResult()
        
        current_tree = self.get_tree_view()
        directory_stats = self._get_directory_stats()
        
        recent_additions_text = "\n".join([
            f"- {a['timestamp']}: Added {a['chunks_added']} chunks to {a['affected_paths']}"
            for a in self.incremental_state.recent_additions[-5:]
        ])
        
        prompt = REORGANIZE_TAXONOMY_PROMPT.format(
            current_tree=current_tree,
            directory_stats=directory_stats,
            add_count=self.incremental_state.add_count,
            recent_additions=recent_additions_text or "No recent additions",
        )
        
        try:
            response = self.generator.generate_single(
                prompt=prompt,
                schema=REORGANIZE_TAXONOMY_SCHEMA,
            )
            
            data = response.get("json")
            if data is None:
                data = self._extract_json(response.get("text", ""))
            
            if data:
                result.reorganization_needed = data.get("reorganization_needed", False)
                result.reasoning = data.get("reasoning", "")
                result.summary = data.get("summary", "")
                result.affected_paths = data.get("affected_paths", [])
                
                if result.reorganization_needed:
                    for op_data in data.get("operations", []):
                        op = ReorganizeOperation(
                            operation=op_data.get("operation", ""),
                            source_paths=op_data.get("source_paths", []),
                            target_path=op_data.get("target_path", ""),
                            new_name=op_data.get("new_name"),
                            description=op_data.get("description"),
                            files_to_move=op_data.get("files_to_move", []),
                        )
                        result.operations.append(op)
                        self._execute_reorganize_operation(op, result)
                
                if self.verbose:
                    if result.reorganization_needed:
                        print(f"  Reorganization performed: {result.summary}")
                    else:
                        print(f"  No reorganization needed: {result.reasoning}")
                        
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Error during reorganization: {e}")
            result.errors.append(str(e))
        
        return result
    
    def _execute_reorganize_operation(
        self,
        operation: ReorganizeOperation,
        result: ReorganizeResult,
    ):
        """Execute a single reorganization operation"""
        if operation.operation == "merge":
            self._mkdir(operation.target_path, "")
            result.directories_created.append(operation.target_path)
            
            for file_move in operation.files_to_move:
                from_path = file_move.get("from", "")
                to_path = file_move.get("to", "")
                if from_path and to_path:
                    self._mv(from_path, to_path)
                    result.files_moved.append({"from": from_path, "to": to_path})
            
            for source_path in operation.source_paths:
                if source_path != operation.target_path:
                    self._rm(source_path)
                    result.directories_removed.append(source_path)
        
        elif operation.operation == "split":
            for file_move in operation.files_to_move:
                to_path = file_move.get("to", "")
                if to_path:
                    target_dir = "/".join(to_path.split("/")[:-1])
                    if target_dir and target_dir not in result.directories_created:
                        self._mkdir(target_dir, "")
                        result.directories_created.append(target_dir)
                    
                    from_path = file_move.get("from", "")
                    if from_path:
                        self._mv(from_path, to_path)
                        result.files_moved.append({"from": from_path, "to": to_path})
        
        elif operation.operation == "rename":
            if operation.source_paths:
                self._mv(operation.source_paths[0], operation.target_path)
                result.files_moved.append({
                    "from": operation.source_paths[0],
                    "to": operation.target_path
                })
        
        elif operation.operation == "move":
            for file_move in operation.files_to_move:
                from_path = file_move.get("from", "")
                to_path = file_move.get("to", "")
                if from_path and to_path:
                    self._mv(from_path, to_path)
                    result.files_moved.append({"from": from_path, "to": to_path})
    
    def _get_directory_stats(self) -> str:
        """Get directory statistics"""
        stats = []
        
        def collect_stats(node, path="/"):
            file_count = len([c for c in node.children if c.is_file])
            dir_count = len([c for c in node.children if c.is_dir])
            stats.append(f"- **{path}**: {file_count} files, {dir_count} subdirectories")
            
            for child_node in node.children:
                if child_node.is_dir:
                    child_name = child_node.name
                    child_path = f"{path}/{child_name}" if path != "/" else f"/{child_name}"
                    collect_stats(child_node, child_path)
        
        collect_stats(self.tree.root)
        return "\n".join(stats) if stats else "(No directory stats available)"
    
    def _update_affected_readmes(self, affected_paths: List[str]):
        """Update READMEs for affected directories (bottom-up)"""
        if not affected_paths:
            return
        
        all_paths = set(affected_paths)
        for path in affected_paths:
            parts = path.strip('/').split('/')
            for i in range(len(parts)):
                parent_path = '/' + '/'.join(parts[:i+1])
                all_paths.add(parent_path)
        all_paths.add('/')
        
        sorted_paths = sorted(all_paths, key=lambda p: -p.count('/'))
        readme_cache: Dict[str, str] = {}
        
        if self.verbose:
            print(f"  Updating {len(sorted_paths)} READMEs (bottom-up)...")
        
        for dir_path in sorted_paths:
            node = self.tree.get_node(dir_path)
            if node is None:
                continue
            
            files_info = []
            subdirs_info = []
            
            for child_node in node.children:
                child_name = child_node.name
                child_path = f"{dir_path}/{child_name}" if dir_path != "/" else f"/{child_name}"
                
                if child_node.is_dir:
                    subdir_readme = readme_cache.get(child_path, child_node.summary or "")
                    subdirs_info.append((child_name, child_path, subdir_readme))
                else:
                    content = child_node.content or ""
                    title = child_name.replace('.md', '').replace('_', ' ')
                    memory = content[:500] if content else ""
                    files_info.append((child_name, title, memory))
            
            dir_name = dir_path.split('/')[-1] or self.tree.root.name or "Root"
            readme_content = self._generate_directory_readme(
                dir_name=dir_name,
                dir_path=dir_path,
                files_info=files_info,
                subdirs_info=subdirs_info,
            )
            
            success = self._update_readme(dir_path, readme_content)
            readme_cache[dir_path] = readme_content
            
            if self.verbose:
                status = "✓" if success else "✗"
                print(f"    {status} README {dir_path}")
    
    def _merge_root_readme(self, previous_readme: str) -> None:
        """Merge previous root README with new one"""
        if not previous_readme:
            return
        
        # Read current README from disk rather than self.tree.root.summary,
        # because the in-memory tree may be stale (not reflecting the README
        # that was just written by _update_affected_readmes).
        readme_abs = str(Path(self.workspace.root_path) / "README.md")
        output, exit_code = self.workspace.run(f'cat "{readme_abs}"')
        current_readme = output.strip() if exit_code == "0" else ""
        
        if not current_readme:
            return
        if previous_readme.strip() in current_readme:
            return
        
        merged_readme = None
        try:
            prompt = MERGE_ROOT_README_PROMPT.format(
                previous_readme=previous_readme,
                new_readme=current_readme,
            )
            response = self.generator.generate_single(
                prompt=prompt,
                schema=MERGE_ROOT_README_SCHEMA,
            )
            data = response.get("json")
            if data is None:
                data = self._extract_json(response.get("text", ""))
            if data:
                merged_readme = data.get("merged_readme")
        except Exception as e:
            if self.verbose:
                print(f"    Warning: README merge failed, fallback to append: {e}")
        
        if not merged_readme:
            merged_readme = (
                f"{current_readme}\n\n"
                f"---\n\n"
                f"## Previous README\n\n"
                f"{previous_readme}"
            )
        
        self._update_readme("/", merged_readme)
    
    # ========== Utility Methods ==========
    
    def get_tree_view(self) -> str:
        """Get tree view using workspace tree command"""
        return self._read_dir_structure()
    
    def get_structure_summary(self) -> str:
        """Get structure summary"""
        # Reload tree from disk to get the latest structure
        self.tree = GAMTree.from_disk(self.tree.root_path, self.workspace)
        return self.tree.get_structure_summary()
    
    def add_file(self, path: str, content: str) -> bool:
        """Add file directly (without LLM)"""
        success = self._write_file(path, content)
        if success and self.auto_save:
            # Reload tree from disk to sync
            self.tree = GAMTree.from_disk(self.tree.root_path, self.workspace)
        return success
    
    def add_directory(self, path: str, readme: str = "") -> bool:
        """Add directory directly (without LLM)"""
        success = self._mkdir(path, readme)
        if success and self.auto_save:
            # Reload tree from disk to sync
            self.tree = GAMTree.from_disk(self.tree.root_path, self.workspace)
        return success
