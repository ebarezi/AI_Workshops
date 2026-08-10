# Workshop 2: RAG + LangGraph + Purdue TA

This folder contains materials for the second Purdue workshop on retrieval-augmented generation and LangGraph agent orchestration using a Purdue TA tool. Everything is delivered as notebooks — no separate scripts or environment files. Each notebook installs its own dependencies from a `%pip install` cell (Section 0), the same pattern as Workshop 1.

## Contents

- `Workshop2_Part1_RAG.ipynb` — Part 1: RAG from scratch (embed, index, retrieve, generate).
- `Workshop2_Part2a_Server.ipynb` — Part 2, server half: defines the MCP tools and starts the server in the foreground. Run this one first and leave its kernel running.
- `Workshop2_Part2b_Agent_ReAct.ipynb` — Part 2, agent half: connects a LangGraph agent to the server from `Part2a` over HTTP and orchestrates its tools as a generic **ReAct** loop (one `agent` node bound to every tool, looping with a `ToolNode`) to run the Boilermaker TA. Needs `Part2a`'s kernel running first.
- `Workshop2_Part2b_Agent_hardgraph.ipynb` — (optional) side-by-side comparison: the same Boilermaker TA task, but built as a **fixed two-node workflow** (`gather_context` → `write`) instead of a generic loop. Order is enforced structurally by the graph shape rather than by the system prompt — useful for showing students the tradeoff between a flexible loop and a structurally-guaranteed order. Also needs `Part2a`'s kernel running first.
- `boilermaker_ta_data/` — the course knowledge base (`knowledge_base.json`) and academic calendar (`purdue_calendar.json`) used by all four notebooks.

## Setup

Each notebook installs what it needs from its own Section 0 `%pip install` cell — run it once, then restart the kernel:

- **Part 1** needs `torch`, `transformers`, `accelerate`, `sentence-transformers`, `faiss-cpu`.
- **Part 2a** (server) needs `sentence-transformers`, `faiss-cpu`, `fastmcp`, `python-dotenv`.
- **Part 2b ReAct** and **Part 2b hardgraph** (agent, either version) need `torch`, `transformers`, `accelerate`, `langchain-huggingface` (for the default local LLM), plus `langchain`, `langchain-core`, `langgraph`, `langchain-mcp-adapters`, `langchain-ollama`, `python-dotenv`. If you've already run the install cell in one of the two agent notebooks in the same environment, you can skip it in the other.

### (Optional) Create a Conda environment named `2ndWorkshop`

```bash
conda create -n 2ndWorkshop python=3.10 pip -y
conda activate 2ndWorkshop
pip install ipykernel
python -m ipykernel install --user --name 2ndWorkshop --display-name "Python (2ndWorkshop)"
```

Select the `Python (2ndWorkshop)` kernel in each notebook, then run its Section 0 install cell once and restart the kernel.

### Purdue Gilbreth cluster

Before launching any of the notebooks, create and select a conda environment kernel (`2ndWorkshop`) from a terminal on the cluster:

```bash
module load conda
conda-env-mod create -n ENV_NAME_HERE -j
module use $HOME/privatemodules
module load conda-env/ENV_NAME_HERE
```

Replace `ENV_NAME_HERE` with your environment name (`2ndWorkshop`), then select the matching kernel in Jupyter before running the cells below.

## Free local / remote models

This workshop does not require paid API keys — both the RAG notebook and the agent notebook default to a small local Hugging Face model.

- **Part 1 (RAG)** and both **Part 2b agent notebooks** (ReAct and hardgraph) default to `Qwen/Qwen2.5-0.5B-Instruct` (same model as Workshop 1) — small enough to run on a laptop CPU in a few seconds. Swap it by editing `LOCAL_MODEL` in the notebook.
- Both **Part 2b agent notebooks** additionally support hosted/local-server backends — Ollama, Purdue GenAI Studio, Anthropic, OpenAI, or any OpenAI-compatible endpoint (vLLM, LM Studio) — by uncommenting one preset in Section C2 and setting `LLM_MODEL` (+ `LLM_BASE_URL` where needed). `Workshop2_Part2b_Agent_ReAct.ipynb`'s C2 has the full five-preset table with install/API-key steps for each; `Workshop2_Part2b_Agent_hardgraph.ipynb`'s C2 mirrors the same five presets. The local default trades some tool-calling reliability for zero setup, so switch to Ollama or a hosted API if the agent demo needs to be rock-solid live.

No OpenAI or Anthropic API key is required for the default flow.

## Running the workshop flow

Three required notebooks, run in order, plus one optional comparison:

1. **`Workshop2_Part1_RAG.ipynb`** — select the `Python (2ndWorkshop)` kernel and run top to bottom. Builds the RAG pipeline (embed, index, retrieve, generate) and answers sample queries.
2. **`Workshop2_Part2a_Server.ipynb`** — same kernel (or a fresh one), run top to bottom. Defines the MCP tools and starts the server; the last cell blocks and keeps it running. Leave this kernel running.
3. **`Workshop2_Part2b_Agent_ReAct.ipynb`** — a *separate* kernel, run top to bottom once Part 2a's server is up. Connects a LangGraph agent to those tools and runs the Boilermaker TA end to end, using a generic ReAct loop (one `agent` node bound to every tool, looping with a `ToolNode`).
4. **(Optional) `Workshop2_Part2b_Agent_hardgraph.ipynb`** — same prerequisite (Part 2a's server running), can run in its own kernel alongside or instead of the ReAct notebook. Rebuilds the same Boilermaker TA as a fixed-order graph (`gather_context` → `write`), so you can compare whether a good system prompt alone reproduces the ordering guarantee the hardgraph version gets for free from its graph structure. Both write to the same `workshop_outputs/announcements.txt` — clear it between runs for a clean comparison.

The agent uses three Purdue TA tools, defined in Part 2a:

- `search_knowledge_base(query)` — searches the local course knowledge base.
- `get_academic_calendar(query)` — reads the course calendar and schedule.
- `create_notification(subject, body)` — writes a notification announcement to `announcements.txt`.

Part 2a also exposes `retrieve_docs(query, k=3)`, which returns the top-k retrieved passages from the knowledge base (embeddings + FAISS) — the same retrieval approach as Part 1, now wrapped as a tool the agent can call.

## Notes

- The workshop flow is intentionally simple: RAG first, then the Purdue TA LangGraph agent, with the agent split across two notebooks (server / client) so MCP's client-server separation is genuinely visible rather than hidden behind a background thread.
- For a full workshop, discuss how retrieval grounds answers and how tool calls extend agent capabilities.
- Part 2a and the Part 2b agent notebooks don't share runtime state — the agent notebooks talk to Part 2a purely over HTTP, so any of these kernels can be restarted independently as long as Part 2a is running when an agent notebook connects.
- `Workshop2_Part2b_Agent_hardgraph.ipynb` is optional and exists specifically to make the fixed-graph-vs-prompting tradeoff concrete: run the same question through both agent notebooks (clearing `announcements.txt` between runs) and see whether the ReAct version's model ever calls `create_notification` before doing its lookups, or writes a placeholder date — failures the hardgraph version's structure rules out entirely.
