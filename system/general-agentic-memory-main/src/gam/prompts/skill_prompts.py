# -*- coding: utf-8 -*-
"""
GAM Skill Prompt

Skill description text to be injected into an external Agent's system prompt,
so the Agent knows it can call GAM capabilities via shell commands.

The caller pre-configures the GAM directory path — the agent does NOT choose it.

Usage:
    from gam import get_skill_prompt
    
    # Pre-configure the GAM directory and get the skill prompt
    skill_text = get_skill_prompt(gam_dir="/data/research_memory")
    
    # Inject into your agent's system prompt
    system_prompt = f"You are a research agent.\\n\\n{skill_text}"
    
    # The agent also needs a shell/bash tool to execute commands.
    # Most agent frameworks provide one (e.g. OpenAI code_interpreter,
    # LangChain BashTool, custom run_command tool, etc.)
"""

GAM_SKILL_PROMPT = """## SKILL: GAM (General Agentic Memory) — Structured Memory Management

You have access to **GAM**, a hierarchical intelligent memory file system.
You can use it to organize files/text into structured, searchable memory, and perform Q&A retrieval over that memory.

**The GAM workspace is pre-configured at: `{gam_dir}`**
Always use this path — do NOT change or create a different GAM directory.

---

### Tool 1: `gam-add` — Build / Update Memory

Add file or text content to the GAM knowledge base. Use `--type text` for text content, `--type video` for video content.

```bash
# Add a file
gam-add --type text --gam-dir {gam_dir} --input /path/to/file.pdf

# Add multiple files
gam-add --type text --gam-dir {gam_dir} -i file1.pdf -i file2.txt

# Add text content directly
gam-add --type text --gam-dir {gam_dir} --content "Text to memorize"

# Add file + text together
gam-add --type text --gam-dir {gam_dir} -i paper.pdf -c "Additional notes about this paper"

# Short text (skip chunking)
gam-add --type text --gam-dir {gam_dir} -c "Brief note" --no-chunking
```

Parameters:
- `--type text/video` / `-t text/video`: **(Required)** GAM type.
- `--input <file>` / `-i <file>`: Input file (PDF/TXT/MD). Repeatable.
- `--content <text>` / `-c <text>`: Direct text. Repeatable.
- `--no-chunking`: Treat each input as one unit (skip intelligent splitting).
- `--no-verbose`: Reduce output.

At least one of `--input` or `--content` is required (for text type).

The system automatically: chunks text intelligently → generates memory summaries → organizes into a hierarchical directory structure. First call creates the GAM; subsequent calls add incrementally.

---

### Tool 2: `gam-request` — Q&A over Memory

Query the GAM knowledge base. Returns an answer with sources and confidence.

```bash
# Ask a question
gam-request --type text --gam-dir {gam_dir} -q "What is the main conclusion?"

# Ask with JSON output (structured, easier to parse)
gam-request --type text --gam-dir {gam_dir} -q "What methods were used?" --json

# Custom system prompt
gam-request --type text --gam-dir {gam_dir} -q "Summarize findings" -s "You are a research assistant"

# More exploration rounds for complex questions
gam-request --type text --gam-dir {gam_dir} -q "Compare all experimental results" --max-iter 20
```

Parameters:
- `--type text/video` / `-t text/video`: **(Required)** GAM type.
- `--question <text>` / `-q <text>`: **(Required)** The question to ask.
- `--json`: Output JSON with fields: answer, sources, confidence.
- `--system-prompt <text>` / `-s <text>`: Custom instruction for the Q&A agent.
- `--max-iter <n>`: Max exploration rounds (default: 10).
- `--no-verbose`: Reduce output.

---

### Workflow

1. **Build**: `gam-add --type text` to ingest files/text into the GAM
2. **Query**: `gam-request --type text` to ask questions over the built memory
3. **Append**: `gam-add --type text` again to add more content (no rebuild needed)

### When to Use

- Understanding long documents (papers, reports, books)
- Retrieving information across multiple files
- Organizing research content into structured knowledge
- Q&A over an existing knowledge base
"""


def get_skill_prompt(gam_dir: str = "/tmp/gam_workspace") -> str:
    """
    Get the GAM skill prompt with a pre-configured GAM directory.
    
    The returned text is ready to be injected into an Agent's system prompt.
    The GAM directory path is baked into the prompt — the agent will always
    use this fixed path and cannot change it.
    
    Args:
        gam_dir: Pre-configured GAM directory path. This path will be
                 hard-coded into the skill prompt. The agent cannot override it.
        
    Returns:
        Complete skill prompt text with the GAM directory baked in.
        
    Example:
        from gam import get_skill_prompt
        
        # 1. Get skill prompt with your chosen directory
        skill = get_skill_prompt(gam_dir="/data/project_memory")
        
        # 2. Inject into your agent's system prompt
        system_prompt = f'''You are a research agent with shell access.
        
        {skill}
        '''
        
        # 3. Your agent also needs a shell/bash tool to execute commands.
        #    Example with a simple run_command tool:
        #    
        #    tools = [
        #        {"type": "function", "function": {
        #            "name": "run_command",
        #            "description": "Execute a shell command",
        #            "parameters": {
        #                "type": "object",
        #                "properties": {"command": {"type": "string"}},
        #                "required": ["command"]
        #            }
        #        }}
        #    ]
        #
        #    The agent will generate commands like:
        #      run_command("gam-add --type text --gam-dir /data/project_memory -i paper.pdf")
        #      run_command("gam-request --type text --gam-dir /data/project_memory -q 'What is X?'")
        #
        #    Your tool executor runs them in a shell and returns stdout to the agent.
    """
    return GAM_SKILL_PROMPT.replace("{gam_dir}", gam_dir)
