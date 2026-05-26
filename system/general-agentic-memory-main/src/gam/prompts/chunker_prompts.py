# -*- coding: utf-8 -*-
"""
Chunker Prompts

Prompt templates for intelligent text chunking operations.
"""

# Prompt for chunk decision
CHUNK_DECISION_PROMPT = """You are a document structure analyzer. Split the text into chunks that are semantically complete and reasonably uniform in granularity (avoid some chunks being tiny while others are huge).

## Document Summary
{document_summary}

## Chunking Format Guidance (from global analysis)
{chunk_format_guidance}

## Text Segment (approximately {token_count} tokens)
Each line is prefixed with its line number (e.g., "line 1: content...").

{text_segment}

## Goal
Follow the chunking format guidance above. Return split points that produce chunks that are:
- **Semantically complete**: each chunk can be understood on its own (has the needed context).
- **Consistent granularity**: avoid a mix of very small and very large chunks.
- **Contentful**: avoid chunks that are just page numbers, figure/table captions, or stray headings.
- **No header-only chunks**: never create a chunk that is only a heading/title with no real body.

## Two-pass decision process (IMPORTANT)
1) **Propose candidate boundaries**:
   - **Top-level section headers** (highest priority): lines like "1 Introduction", "2 Preliminaries", "3 Form: ...", "4.2 ...".
     - These are usually the best split points *unless* splitting there would create a tiny preceding chunk.
   - Major section headers (e.g., "3.2 Parametric Memory", "## Results", "Introduction")
   - Subsection headers (e.g., "3.1.1 ...", "4.2.3 ...") *ONLY IF* the resulting chunks will be contentful
   - Clear topic shifts where the subject meaningfully changes
2) **Refine for size + completeness**:
   - **Merge away tiny chunks**: if a candidate split would create a chunk that is too short or incomplete, do NOT split there.
   - **Split oversized chunks**: if no split is chosen but the segment contains multiple clear subsections, add splits at the best subsection boundaries.
   - Prefer boundaries that keep definitions/examples/discussion together.
   - Enforce **no header-only chunks**: if a split would isolate a heading (e.g., just "1 Introduction") as its own chunk, do NOT split there; attach the heading to the following content.

## When NOT to Split
- **Noise / artifacts** (do NOT create standalone chunks for these; keep them with adjacent content):
  - Page numbers (lines that are only digits like "74")
  - Figure/table captions, "Figure X", "Table Y", "Continued on next page", footers/headers
  - Isolated section numbers or dangling headings with no real body
- **Inside** a table, list, bibliography/reference block, or code block (keep them whole with surrounding context)
- Between a heading and its immediate content (heading must stay with its body)
- Between a section and its immediate Summary/Conclusion paragraph (keep summary with what it summarizes)
- Inside a references/bibliography list: do not split citation-by-citation. If you must split (too large), split into a small number of large groups.

## Chunk Size Guidance (soft constraints, use judgement)
- **Minimum**: avoid creating chunks with < ~20 lines of meaningful content (unless it's a complete standalone section with real substance).
- **Target**: aim for chunks that feel like a coherent section/subsection (often ~40-160 lines of meaningful content).
- **Maximum**: if a chunk would exceed ~250+ lines *and* contains clear subsection boundaries, prefer splitting at those subsection boundaries.
- If a subsection is very short (definitions-only, a tiny bullet list, a single paragraph), keep it with its parent/adjacent related content.
- **Heading-only rule (hard)**: a heading/title line (e.g., "1 Introduction", "## Methods") must be grouped with the content that follows it. Never output a split that makes the heading the last line of a chunk or the only real content of a chunk.

## Key Principle
Each chunk should be a **self-contained unit** covering ONE topic (or a tightly related set of subtopics) with enough detail to be useful on its own. Avoid fragments that lack context, and avoid mega-chunks that mix multiple independent topics.

## Output Format
```json
{{
  "should_split": true/false,
  "reasoning": "Briefly explain the boundaries chosen (and any merges to avoid tiny chunks / splits to avoid oversized chunks).",
  "split_lines": [line_number1, line_number2, ...]
}}
```

**Rules for split_lines**:
- Each number is the LINE NUMBER where a new section BEGINS
- Only include split points where BOTH resulting chunks will have substantial content and be semantically complete
- Order them ascending
- If should_split is false, split_lines must be empty []
"""


# Memory generation prompt
MEMORY_GENERATION_PROMPT = """You are a memory specialist. Your task is to memorize the given content and create a concise memory record.

## Document Summary
{document_summary}

## Memory Format Guidance (from global analysis)
{memory_format_guidance}

## Content to Memorize (Chunk #{chunk_index})
```
{chunk_content}
```

## Memorization Instruction
{memorize_instruction}

## Task
Follow the memory format guidance above. Based on the content and instruction, create a memory record:

1. **Title**: A short, descriptive title in `snake_case` format (English only)
   - Should capture the main topic/concept
   - Examples: `pytorch_autograd_basics`, `fastapi_project_setup`, `debugging_memory_leaks`

2. **TLDR**: A structural overview (1-3 sentences) that describes:
   - **What structure**: What type of content is this? (e.g., code implementation, API documentation, theoretical explanation, configuration guide, algorithm description, etc.)
   - **How organized**: How is the content organized? (e.g., "contains 3 main functions with helper utilities", "structured as problem-solution pairs", "organized by component hierarchy", etc.)
   - **What content**: What are the key topics/components covered? (e.g., "covers authentication flow, session management, and token refresh")

3. **Memory**: A concise summary that:
   - Preserves key information and concepts
   - Captures important details, examples, code patterns
   - Is useful for future retrieval and understanding
   - Follows the memorization instruction if provided

## Output Format
```json
{{
  "title": "descriptive_title_in_snake_case",
  "tldr": "Structural overview describing what type of content, how it's organized, and what key topics are covered",
  "memory": "Concise summary preserving key information..."
}}
```
"""

# Prompt for generating document summary
GENERATE_DOCUMENT_SUMMARY_PROMPT = """You are a document analyzer. Your task is to provide a brief, high-level summary of the entire content provided below. 
This summary will be used as background context to help another model intelligently split the text into chunks and generate memories for each chunk.

Focus on:
- The overall topic or purpose of the document.
- Main sections or key themes.
- Any important entities or terminology.

Keep it concise.

## Content to Summarize
{content}

## Task
Generate a concise introduction/summary for this document.
"""


# ========== Global Format Analysis Prompts ==========

# Prompt for analyzing how content should be chunked
ANALYZE_CHUNK_FORMAT_PROMPT = """You are a document structure expert. Analyze the content below and provide guidance on how it should be split into chunks.

## Content Overview
{content}

## Task
Based on this content's nature and structure, provide chunking guidance:

1. **Content Type**: What type of content is this? (e.g., academic paper, code documentation, tutorial, API reference, configuration guide, narrative text, etc.)

2. **Natural Structure**: What is the natural structure of this content?
   - What are the main organizational units? (chapters, sections, functions, classes, topics, etc.)
   - Are there clear hierarchical levels? (e.g., Chapter > Section > Subsection)
   - Are there recurring patterns? (e.g., problem-solution pairs, Q&A format, step-by-step instructions)

3. **Recommended Chunk Granularity**: Based on the content type and structure, what granularity should chunks have?
   - Should chunks be at section level, subsection level, or paragraph level?
   - What makes a semantically complete unit for this specific content?
   - Any special considerations? (e.g., keep code with its explanation, keep tables with their descriptions)

4. **Boundaries to Respect**: What boundaries should NOT be crossed when chunking?
   - What elements should always stay together?
   - What would be broken if split in the middle?

## Output Format
```json
{{
  "content_type": "Brief description of content type",
  "natural_structure": "Description of the document's natural organizational structure",
  "recommended_granularity": "Specific guidance on chunk size and what constitutes a complete unit",
  "boundaries_to_respect": "What should not be split apart",
  "special_considerations": "Any other chunking advice specific to this content"
}}
```
"""


# Prompt for analyzing how memory should be formatted
ANALYZE_MEMORY_FORMAT_PROMPT = """You are a memory and summarization expert. Analyze the content below and provide guidance on how memories/summaries should be structured for each chunk.

## Content Overview
{content}

## Document Summary
{document_summary}

## Task
Based on this content's nature, provide memory generation guidance:

1. **Key Information Types**: What types of information are most important to preserve?
   - What should definitely be captured in memories? (concepts, code patterns, formulas, steps, relationships, etc.)
   - What can be omitted or heavily summarized?

2. **Recommended Memory Structure**: How should each chunk's memory be structured?
   - Should it follow the original structure or reorganize?
   - What format works best? (bullet points, prose, code snippets, tables, etc.)
   - Should it include examples? References? Code?

3. **TLDR Focus**: What should the TLDR emphasize for this type of content?
   - What structural aspects are most relevant?
   - How should the organization be described?

4. **Abstraction Level**: How detailed should memories be?
   - Should they be high-level summaries or detailed notes?
   - What level of technical detail should be preserved?

5. **Cross-references**: How should relationships between chunks be handled?
   - Should memories reference related concepts from other chunks?
   - Should dependencies be noted?

## Output Format
```json
{{
  "key_information_types": "What information to prioritize preserving",
  "recommended_structure": "How to structure the memory content",
  "tldr_focus": "What the TLDR should emphasize",
  "abstraction_level": "How detailed memories should be",
  "special_instructions": "Any other memory generation advice specific to this content"
}}
```
"""


# JSON Schemas for format analysis
CHUNK_FORMAT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "content_type": {"type": "string"},
        "natural_structure": {"type": "string"},
        "recommended_granularity": {"type": "string"},
        "boundaries_to_respect": {"type": "string"},
        "special_considerations": {"type": "string"}
    },
    "required": ["content_type", "recommended_granularity"]
}

MEMORY_FORMAT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "key_information_types": {"type": "string"},
        "recommended_structure": {"type": "string"},
        "tldr_focus": {"type": "string"},
        "abstraction_level": {"type": "string"},
        "special_instructions": {"type": "string"}
    },
    "required": ["key_information_types", "recommended_structure"]
}

DEFAULT_MEMORIZE_INSTRUCTION = """Summarize the content, preserving:
- Main concepts and ideas
- Key facts, numbers, and data
- Important code patterns or examples
- Actionable information
Keep the summary concise but comprehensive enough to understand without the original content.

For the TLDR, provide a structural overview that describes:
- What type of content this is (code, documentation, theory, config, etc.)
- How the content is organized (sections, components, flow, etc.)
- What key topics or components are covered"""
