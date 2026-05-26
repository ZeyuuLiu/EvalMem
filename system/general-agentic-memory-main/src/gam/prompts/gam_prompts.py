# -*- coding: utf-8 -*-
"""
GAM Agent Prompts

Prompts for directory organization, README generation, and taxonomy management.
"""

# ========== Batch Processing Prompts ==========

GAM_BATCH_MEMORIZE_CHUNK_PROMPT = """You are a memory specialist. Your task is to memorize the given content and create a concise memory record.

## Content to Memorize (Chunk #{chunk_index})
```
{chunk_content}
```

## Memorization Instruction
{memorize_instruction}

## Task
Based on the content and instruction above, create a memory record:

1. **Title**: A short, descriptive title in `snake_case` format (English only)
   - Should capture the main topic/concept
   - Examples: `pytorch_autograd_basics`, `fastapi_project_setup`, `debugging_memory_leaks`

2. **Memory**: A concise summary that:
   - Preserves key information and concepts
   - Captures important details, examples, code patterns
   - Is useful for future retrieval and understanding
   - Follows the memorization instruction if provided

## Output Format
```json
{{
  "title": "descriptive_title_in_snake_case",
  "memory": "Concise summary preserving key information..."
}}
```
"""


GAM_BATCH_ORGANIZE_PROMPT = """You are an intelligent file system organizer. Your task is to organize multiple memorized chunks into a hierarchical General Agentic Memory (GAM).

## All Memorized Chunks
{all_memories}

## Context (if any)
{context}

## Task
Analyze ALL the memories above and design a complete, hierarchical directory structure to organize them.

**Note**: You only need to design the directory structure. READMEs will be generated automatically in a bottom-up manner based on the actual content.

## Directory Structure Rules

1. **Parent Directories**: Non-leaf directories should:
   - Have meaningful, descriptive names
   - Group related content logically
   - Have `chunk_indices` as empty array `[]`

2. **Leaf Directories**: The deepest directories should:
   - Contain actual chunks (specify chunk indices in `chunk_indices`)
   - Each chunk stored in appropriate leaf directory
   - One directory can hold multiple related chunks

3. **Structure Guidelines**:
   - Don't over-nest (max 3-4 levels recommended)
   - Keep related content together
   - Use meaningful directory names in snake_case
   - Balance breadth vs depth

## Output Format
```json
{{
  "reasoning": "<analysis of all memories and why this organization makes sense>",
  "directories": [
    {{
      "path": "<absolute path starting with />",
      "name": "<directory name in snake_case>",
      "description": "<brief description of directory purpose>",
      "children": ["<child path 1>", "<child path 2>", ...],
      "chunk_indices": []  // empty for parent directories
    }},
    {{
      "path": "<leaf directory path>",
      "name": "<directory name>",
      "description": "<description>",
      "children": [],  // empty for leaf directories
      "chunk_indices": [0, 1, ...]  // chunk indices for leaf directories
    }}
  ]
}}
```

## Important Notes
- Every chunk index (0 to N-1) must appear in exactly ONE leaf directory's chunk_indices
- Parent directories have empty chunk_indices: []
- Leaf directories have non-empty chunk_indices: [index1, index2, ...]
- All paths start with /
- Ensure all directories form a valid tree (children paths match parent path prefix)
"""


GAM_BATCH_DEFAULT_MEMORIZE_INSTRUCTION = """Summarize the content, preserving:
- Main concepts and ideas
- Key facts, numbers, and data
- Important code patterns or examples
- Actionable information
Keep the summary concise but comprehensive enough to understand without the original content."""


# ========== README Generation Prompts ==========

GAM_GENERATE_README_PROMPT = """You are a technical documentation specialist. Your task is to generate a README for a directory based on its contents.

## Directory Name
{directory_name}

## Directory Contents

### Files in this directory
{files_content}

### Subdirectories in this directory
{subdirs_content}

## Task
Generate a README for this directory with the following structure:

1. **Directory Description**: A concise description (2-4 sentences) of what this directory contains and its purpose.

2. **Brief Introductions**: For each file/subdirectory, provide a brief one-line introduction.

3. **Detailed Introductions**: For each file/subdirectory, provide a more detailed description (2-4 sentences).

## Output Format
```json
{{
  "description": "A concise description of the directory's purpose and contents...",
  "items": [
    {{
      "name": "item_name",
      "type": "file|directory",
      "brief": "One-line brief introduction",
      "detailed": "More detailed description (2-4 sentences)"
    }}
  ]
}}
```

## Important Notes
- The `name` field in each item MUST be the EXACT filename (e.g. `my_file.md`) or directory name (e.g. `my_dir`) as shown in the Directory Contents above. Do NOT modify, rename, or reformat the name.
- Keep descriptions clear, informative, and well-organized
- Focus on what each item contains and why it's useful
- Use technical language appropriate for documentation
- Maintain consistency in tone and style
"""


GAM_GENERATE_README_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "Directory description (2-4 sentences)"
        },
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["file", "directory"]},
                    "brief": {"type": "string"},
                    "detailed": {"type": "string"}
                },
                "required": ["name", "type", "brief", "detailed"]
            }
        }
    },
    "required": ["description", "items"]
}

# ========== README Merge Prompts ==========

MERGE_ROOT_README_PROMPT = """You are a documentation editor. Merge the previous README with the newly generated README into a single coherent README.

## Previous README
```
{previous_readme}
```

## New README
```
{new_readme}
```

## Task
Produce a merged README that:
1. Preserves important information from both versions
2. Removes duplication and outdated/conflicting content
3. Keeps a clean structure suitable for a root GAM README

## Output Format
```json
{{
  "merged_readme": "The merged README content"
}}
```
"""

MERGE_ROOT_README_SCHEMA = {
    "type": "object",
    "properties": {
        "merged_readme": {
            "type": "string",
            "description": "Merged README content"
        }
    },
    "required": ["merged_readme"]
}


# ========== Taxonomy-based Organization Prompts ==========

# Prompt for generating initial taxonomy from TLDRs
GENERATE_TAXONOMY_FROM_TLDR_PROMPT = """You are an intelligent taxonomy designer. Your task is to create a hierarchical directory structure (taxonomy) based on the TLDRs of content chunks.

## Content TLDRs (Batch {batch_num}/{total_batches})
{tldr_list}

## Context
{context}

## Task
Design a hierarchical taxonomy that can organize the above content. Focus on:
1. **Logical grouping**: Group related content together
2. **Clear hierarchy**: Create parent categories for broader topics, child categories for specific topics
3. **Meaningful names**: Use descriptive snake_case names
4. **Balanced structure**: Avoid too deep (max 3-4 levels) or too flat structures

## Output Format
```json
{{
  "reasoning": "Brief explanation of the taxonomy design logic",
  "taxonomy": [
    {{
      "path": "/category_name",
      "name": "category_name",
      "description": "What this category contains",
      "children": ["/category_name/subcategory1", "/category_name/subcategory2"]
    }},
    {{
      "path": "/category_name/subcategory1",
      "name": "subcategory1",
      "description": "What this subcategory contains",
      "children": []
    }}
  ]
}}
```

## Rules
- All paths start with /
- Parent directories must list their children paths
- Leaf directories have empty children: []
- Use snake_case for names
- Keep descriptions concise but informative
"""


# Prompt for adjusting existing taxonomy with new TLDRs
ADJUST_TAXONOMY_WITH_TLDR_PROMPT = """You are an intelligent taxonomy maintainer. Your task is to adjust an existing taxonomy based on new content TLDRs.

## Existing Taxonomy
```
{existing_taxonomy}
```

## New Content TLDRs (Batch {batch_num}/{total_batches})
{tldr_list}

## Context
{context}

## Task
Analyze the new TLDRs and adjust the existing taxonomy if needed:
1. **Add new categories**: If new content doesn't fit existing categories
2. **Merge categories**: If categories are too similar or redundant
3. **Split categories**: If a category is becoming too broad
4. **Rename categories**: If names no longer reflect content accurately
5. **Keep stable**: Don't remove or modify categories that are unrelated to new content

## IMPORTANT
- Do NOT remove existing categories unless they need to be merged or renamed
- Preserve the overall structure when possible
- Only make necessary changes to accommodate new content

## Output Format
```json
{{
  "reasoning": "Brief explanation of changes made (or why no changes needed)",
  "changes_made": ["added /new_category", "merged /old1 and /old2 into /merged", ...],
  "taxonomy": [
    {{
      "path": "/category_name",
      "name": "category_name", 
      "description": "What this category contains",
      "children": ["/category_name/subcategory1"]
    }}
  ]
}}
```
"""


# Prompt for assigning a single chunk to taxonomy
ASSIGN_CHUNK_TO_TAXONOMY_PROMPT = """You are a content classifier. Your task is to assign a content chunk to the most appropriate directory in the taxonomy.

## Taxonomy Structure
```
{taxonomy_tree}
```

## Chunk to Assign
**Index**: {chunk_index}
**Title**: {chunk_title}
**TLDR**: {chunk_tldr}
**Memory Preview**: {chunk_memory_preview}

## Task
Determine the best directory path to place this chunk. Consider:
1. **Content relevance**: Match the chunk's topic to the most relevant category
2. **Specificity**: Prefer more specific (deeper) directories when appropriate
3. **Leaf preference**: Chunks should ideally go into leaf directories (no children)

## Output Format
```json
{{
  "reasoning": "Brief explanation of why this directory was chosen",
  "assigned_path": "/path/to/directory",
  "confidence": 0.95
}}
```

## Rules
- assigned_path must be an existing path in the taxonomy
- confidence is a float between 0 and 1
- If no good match exists, choose the closest match and note it in reasoning
"""


# JSON Schemas for taxonomy operations
GENERATE_TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Explanation of taxonomy design logic"
        },
        "taxonomy": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path starting with /"},
                    "name": {"type": "string", "description": "Directory name in snake_case"},
                    "description": {"type": "string", "description": "What this directory contains"},
                    "children": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Child directory paths"
                    }
                },
                "required": ["path", "name", "description", "children"]
            }
        }
    },
    "required": ["reasoning", "taxonomy"]
}


ADJUST_TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Explanation of changes made"
        },
        "changes_made": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of changes made to taxonomy"
        },
        "taxonomy": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "children": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["path", "name", "description", "children"]
            }
        }
    },
    "required": ["reasoning", "taxonomy"]
}


ASSIGN_CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Why this directory was chosen"
        },
        "assigned_path": {
            "type": "string",
            "description": "Directory path to assign chunk to"
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score 0-1"
        }
    },
    "required": ["reasoning", "assigned_path"]
}


# ========== Incremental Add Prompts ==========

# Prompt for deciding whether new content belongs to existing taxonomy
SHOULD_ADD_TO_EXISTING_PROMPT = """You are a content classifier. Analyze whether new content should be added under the existing directory structure, or if it represents a different topic that needs a separate top-level directory.

## Existing Directory README
```
{existing_readme}
```

## Existing Directory Structure
```
{existing_tree}
```

## New Content Summary
{new_content_summary}

## Task
Decide if the new content belongs under the existing directory structure:

1. **belongs_to_existing**: true if new content is related to the existing content and should be added under the existing structure
2. **belongs_to_existing**: false if new content is a different topic and needs its own separate directory

## Output Format
```json
{{
  "reasoning": "Brief explanation of the decision",
  "belongs_to_existing": true|false,
  "suggested_location": "/path/to/suggested/directory (if belongs_to_existing is true)",
  "new_topic_name": "name_for_new_topic (if belongs_to_existing is false)",
  "new_topic_description": "description for new topic directory (if belongs_to_existing is false)"
}}
```
"""

SHOULD_ADD_TO_EXISTING_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "belongs_to_existing": {"type": "boolean"},
        "suggested_location": {"type": "string"},
        "new_topic_name": {"type": "string"},
        "new_topic_description": {"type": "string"}
    },
    "required": ["reasoning", "belongs_to_existing"]
}


# Prompt for adding new chunks to existing taxonomy
ADD_CHUNKS_TO_EXISTING_TAXONOMY_PROMPT = """You are a content organizer. Your task is to add new content chunks to an existing directory structure.

## Existing Directory Structure
```
{existing_tree}
```

## Existing Directory Descriptions
{directory_descriptions}

## New Chunks to Add
{new_chunks_info}

## Task
For each new chunk, determine the best placement:
1. **Use existing directory**: If the chunk fits well in an existing directory
2. **Create new subdirectory**: If the chunk needs a new subcategory under an existing directory
3. **Create new top-level directory**: Only if the content is truly different from all existing categories

## Output Format
```json
{{
  "reasoning": "Brief explanation of placement decisions",
  "assignments": [
    {{
      "chunk_index": 0,
      "chunk_title": "chunk_title",
      "action": "use_existing|create_subdir|create_toplevel",
      "target_path": "/path/to/directory",
      "new_dir_name": "new_directory_name",
      "new_dir_description": "Description for new directory (if creating new)"
    }}
  ],
  "new_directories": [
    {{
      "path": "/path/to/new_dir",
      "name": "new_dir_name",
      "description": "What this directory contains",
      "parent_path": "/parent/path"
    }}
  ],
  "affected_paths": ["/path1", "/path2"]
}}
```

## Rules
- Prefer using existing directories when content fits
- Only create new directories when truly necessary
- new_directories should list ALL new directories to create (in order from parent to child)
- affected_paths should list all directories that need README updates (including parents of new dirs)
"""


# Prompt for reorganizing taxonomy after multiple adds
REORGANIZE_TAXONOMY_PROMPT = """You are a taxonomy optimizer. After multiple content additions, the directory structure may need reorganization.

## Current Directory Structure
```
{current_tree}
```

## Directory Statistics
{directory_stats}

## Recent Additions (last {add_count} operations)
{recent_additions}

## Task
Analyze the current structure and suggest reorganizations:
1. **Merge**: Combine directories that are too similar or have few items
2. **Split**: Divide directories that have grown too large or cover too many topics
3. **Rename**: Update names that no longer accurately describe contents
4. **Move**: Relocate content to better-fitting locations
5. **Keep**: Leave well-organized directories unchanged

## Guidelines
- Only suggest changes that clearly improve organization
- Preserve the overall structure when possible
- Consider user navigation experience
- Aim for balanced directory sizes (not too few, not too many items per directory)

## Output Format
```json
{{
  "reasoning": "Analysis of current structure and why changes are needed",
  "reorganization_needed": true|false,
  "operations": [
    {{
      "operation": "merge|split|rename|move",
      "source_paths": ["/path1", "/path2"],
      "target_path": "/new/path",
      "new_name": "new_directory_name",
      "description": "New description",
      "files_to_move": [
        {{"from": "/old/path/file.md", "to": "/new/path/file.md"}}
      ]
    }}
  ],
  "affected_paths": ["/path1", "/path2", "/path3"],
  "summary": "Brief summary of all changes"
}}
```

## Rules
- source_paths: directories involved in the operation
- target_path: resulting directory path
- files_to_move: specific files that need to be relocated
- affected_paths: ALL directories needing README updates after reorganization
"""


# JSON Schemas for incremental operations
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
                    "action": {"type": "string", "enum": ["use_existing", "create_subdir", "create_toplevel"]},
                    "target_path": {"type": "string"},
                    "new_dir_name": {"type": "string"},
                    "new_dir_description": {"type": "string"}
                },
                "required": ["chunk_index", "action", "target_path"]
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
                },
                "required": ["path", "name", "description"]
            }
        },
        "affected_paths": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["reasoning", "assignments", "affected_paths"]
}


REORGANIZE_TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "reorganization_needed": {"type": "boolean"},
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["merge", "split", "rename", "move"]},
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
                },
                "required": ["operation", "source_paths", "target_path"]
            }
        },
        "affected_paths": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"}
    },
    "required": ["reasoning", "reorganization_needed"]
}
