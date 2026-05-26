import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import openai
from rich import print as rprint
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from gam import (
    TextGAMAgent, GAMTree,
    OpenAIGenerator, OpenAIGeneratorConfig,
    LocalWorkspace, TextChatAgent
)

# --- BM25 Searcher for Example ---

class BM25Searcher:
    """A searcher using rank_bm25 for more realistic retrieval."""
    def __init__(self):
        # Expanded corpus to make the task more complex
        self.corpus_data = [
            {"id": "doc1", "text": "The General Agentic Memory (GAM) is a hierarchical memory system designed for LLM agents. It mimics a traditional Linux file system structure but is optimized for agentic workflows."},
            {"id": "doc2", "text": "Long horizontal tasks, also known as long-horizon tasks, require agents to maintain state and context over a large number of steps and interactions. Context window limits are a major bottleneck for these tasks."},
            {"id": "doc3", "text": "The 'memorize' tool in GAM is a core component for context management. It takes raw, often verbose search results and uses an AI agent to compress them into refined, structured knowledge."},
            {"id": "doc4", "text": "When an agent calls 'memorize', the GAM system replaces the original long tool outputs in the conversation history with a shorter '[GAM Memory Result]' tag, significantly saving tokens."},
            {"id": "doc5", "text": "The 'recall' tool allows agents to perform semantic or keyword-based searches within their own GAM memory. This is crucial for retrieving information stored many steps ago."},
            {"id": "doc6", "text": "GAM stands for Generative Agent Memory - General Agentic Memory. It is the implementation of the GAM concept in the GAM framework."},
            {"id": "doc7", "text": "GAM uses a 'Taxonomy-based Organization' strategy. It analyzes the content of new memories and automatically places them into appropriate directories like /research/papers/ or /project/notes/."},
            {"id": "doc8", "text": "The GAM tree structure is stored on disk and can be reloaded. Each directory in GAM contains a README.md file that summarizes its contents, helping the agent navigate the memory hierarchy."},
            {"id": "doc9", "text": "In long horizontal tasks, an agent might perform 20+ searches. Without GAM, the context window would overflow. GAM allows the agent to 'offload' information to disk and only keep summaries in context."},
            {"id": "doc10", "text": "The 'memorize' process involves: 1. Intelligent chunking of input text. 2. Summary generation for each chunk. 3. Placement into the GAM tree. 4. Updating parent directory READMEs."},
            {"id": "doc11", "text": "GAM supports different types of workspaces, including LocalWorkspace for local file operations and DockerWorkspace for isolated execution environments."},
            {"id": "doc12", "text": "The 'TextGAMAgent' is responsible for building and updating the GAM, while the 'TextChatAgent' is used for exploring the GAM and answering questions."},
            {"id": "doc13", "text": "Context window overflow leads to 'lost in the middle' phenomena where LLMs forget information in the middle of long prompts. GAM mitigates this by keeping only relevant summaries in the prompt."},
            {"id": "doc14", "text": "A typical long-horizon workflow with GAM: Search -> Memorize -> Clear Context -> Search More -> Recall -> Final Answer."},
            {"id": "doc15", "text": "The 'memorize' tool can take multiple search indices at once, allowing for batch processing of information gathered in a single reasoning step."},
        ]
        self.documents = [d["text"] for d in self.corpus_data]
        self.tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, k=3):
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]
        
        results = []
        for i in top_n_indices:
            if doc_scores[i] > 0:
                doc = self.corpus_data[i]
                results.append({
                    "docid": doc["id"],
                    "score": float(doc_scores[i]),
                    "text": doc["text"]
                })
        return results

    def get_document(self, docid):
        for doc in self.corpus_data:
            if doc["id"] == docid:
                return {"docid": docid, "text": doc["text"]}
        return None

    def search_description(self, k=3):
        return f"Search the knowledge base using BM25. Returns top-{k} results with docid, score, and text."

    def get_document_description(self):
        return "Retrieve a full document by its ID."

    @property
    def search_type(self):
        return "BM25Okapi"

# --- GAM Tool Handler ---

class SearchToolHandler:
    def __init__(
        self,
        searcher,
        gam_dir: str,
        gam_model: str,
        gam_api_key: str,
        gam_api_base: str,
        gam_verbose: bool = True,
    ):
        self.searcher = searcher
        self.gam_dir = gam_dir
        self.gam_model = gam_model
        self.gam_api_key = gam_api_key
        self.gam_api_base = gam_api_base
        self.gam_verbose = gam_verbose
        
        self.search_history = []
        self._input_messages = None
        
        # Lazy-initialized GAM components
        self._gam_generator = None
        self._gam_workspace = None
        self._gam_tree = None
        self._gam_agent = None

    def bind_conversation(self, input_messages: list):
        self._input_messages = input_messages

    def _ensure_gam_initialized(self):
        if self._gam_agent is not None:
            return

        config = OpenAIGeneratorConfig(
            model_name=self.gam_model,
            api_key=self.gam_api_key,
            base_url=self.gam_api_base,
        )
        generator = OpenAIGenerator(config)

        gam_path = Path(self.gam_dir).resolve()
        gam_path.mkdir(parents=True, exist_ok=True)
        workspace = LocalWorkspace(root_path=str(gam_path))

        try:
            tree = GAMTree.from_disk(gam_path, workspace)
        except Exception:
            tree = GAMTree.create_empty(gam_path, name=gam_path.name)

        self._gam_generator = generator
        self._gam_workspace = workspace
        self._gam_tree = tree
        self._gam_agent = TextGAMAgent(
            generator, tree, workspace,
            use_chunking=False, verbose=self.gam_verbose,
        )

    def get_tool_definitions(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": self.searcher.search_description(),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memorize",
                    "description": "CRITICAL: Memorize search results into GAM to save context tokens. This replaces the long raw search results in your history with a short summary. Use this whenever you have gathered significant information or before your context window gets too full.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_indices": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Indices of searches to memorize (e.g. [0, 1])"
                            },
                            "question": {"type": "string", "description": "Guiding question for summarization"}
                        },
                        "required": ["search_indices"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "recall",
                    "description": "Recall information from GAM memory. Use this to retrieve details that you previously 'memorized' and are no longer in your immediate context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "Question to search in memory"}
                        },
                        "required": ["question"],
                    },
                },
            }
        ]

    def execute_tool(self, tool_name, args, tool_call_id=None):
        if tool_name == "search":
            query = args["query"]
            hits = self.searcher.search(query)
            result_json = json.dumps(hits, indent=2)
            
            search_idx = len(self.search_history)
            self.search_history.append({
                "index": search_idx,
                "query": query,
                "result": result_json,
                "tool_call_id": tool_call_id,
                "memorized": False
            })
            return f"[Search #{search_idx}] Query: \"{query}\"\n\n{result_json}"
            
        elif tool_name == "memorize":
            indices = args["search_indices"]
            self._ensure_gam_initialized()
            
            content_parts = []
            for idx in indices:
                if idx >= len(self.search_history):
                    continue
                entry = self.search_history[idx]
                content_parts.append(f"## Search #{idx}\nQuery: {entry['query']}\nResults:\n{entry['result']}")
            
            if not content_parts:
                return json.dumps({"status": "error", "message": "No valid search indices provided."})

            # Add to GAM
            self._gam_agent.add(content=content_parts)
            
            # Generate refined answer for each
            results = []
            for idx in indices:
                if idx >= len(self.search_history):
                    continue
                entry = self.search_history[idx]
                chat_agent = TextChatAgent(self._gam_generator, self._gam_tree, workspace=self._gam_workspace)
                chat_res = chat_agent.chat(args.get("question") or entry["query"])
                gam_answer = chat_res.answer
                
                gam_tagged = f"[GAM Memory Result] (refined from Search #{idx})\n\n{gam_answer}"
                
                # Replace in message history
                if self._input_messages:
                    for msg in self._input_messages:
                        msg_role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
                        msg_tool_call_id = msg.get("tool_call_id") if isinstance(msg, dict) else getattr(msg, "tool_call_id", None)
                        
                        if msg_role == "tool" and msg_tool_call_id == entry["tool_call_id"]:
                            if isinstance(msg, dict):
                                msg["content"] = gam_tagged
                            else:
                                msg.content = gam_tagged
                
                entry["memorized"] = True
                results.append({"index": idx, "status": "success"})
                
            return json.dumps({"status": "success", "results": results, "message": "Information successfully offloaded to GAM memory."})

        elif tool_name == "recall":
            self._ensure_gam_initialized()
            chat_agent = TextChatAgent(self._gam_generator, self._gam_tree, workspace=self._gam_workspace)
            res = chat_agent.chat(args["question"])
            return json.dumps({"answer": res.answer, "sources": res.sources})

# --- Main Logic ---

def request_example():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--gam-dir", default="./gam_example_storage")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: OPENAI_API_KEY is required.")
        return

    client = openai.OpenAI(api_key=args.api_key, base_url=args.api_base)
    searcher = BM25Searcher()
    
    handler = SearchToolHandler(
        searcher=searcher,
        gam_dir=args.gam_dir,
        gam_model=args.model,
        gam_api_key=args.api_key,
        gam_api_base=args.api_base
    )

    system_prompt = (
        "You are an advanced research agent specializing in long-horizon memory systems.\n"
        "Your goal is to provide a detailed, multi-faceted answer to the user's query.\n\n"
        "INSTRUCTIONS:\n"
        "1. Start by searching for different aspects of the query. Use multiple `search` calls.\n"
        "2. IMPORTANT: To manage your context window, you MUST use the `memorize` tool to compress and offload information into GAM after every 2-3 searches.\n"
        "3. After memorizing, the long raw results will be replaced by short summaries in your history.\n"
        "4. Use `recall` to retrieve specific details from GAM if you need them for your final synthesis.\n"
        "5. Finally, provide a comprehensive answer based on all gathered and recalled information."
    )
    
    # A more complex, multi-hop query
    user_query = (
        "Explain the complete workflow of GAM in long-horizon tasks, "
        "specifically focusing on how the 'memorize' tool handles context window bottlenecks "
        "and how the hierarchical taxonomy-based organization helps the agent recall information later."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    handler.bind_conversation(messages)

    print(f"\n🚀 Starting Complex Long Horizontal Example (BM25)")
    print(f"User Query: {user_query}\n")

    for i in range(10):  # Max 10 rounds for complex reasoning
        print(f"--- Round {i+1} ---")
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            tools=handler.get_tool_definitions(),
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        if msg.content:
            print(f"Assistant: {msg.content}")
            
        if not msg.tool_calls:
            # If the agent hasn't used memorize yet but is trying to finish, nudge it
            if not any(s["memorized"] for s in handler.search_history) and len(handler.search_history) > 0:
                print("[Nudge] Agent trying to finish without memorizing. Encouraging memory usage...")
                messages.append({"role": "user", "content": "Please memorize your search findings into GAM before providing the final answer."})
                continue
            break
            
        for tc in msg.tool_calls:
            print(f"Tool Call: {tc.function.name}({tc.function.arguments})")
            tool_res = handler.execute_tool(tc.function.name, json.loads(tc.function.arguments), tc.id)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": tool_res
            })
            print(f"Tool Response: {tool_res[:150]}...")

    print("\n✅ Complex Example finished.")

if __name__ == "__main__":
    request_example()
