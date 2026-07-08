# Workshop 2: RAG + LangGraph + Purdue TA

This folder contains materials for the second Purdue workshop on retrieval-augmented generation and LangGraph agent orchestration using a Purdue TA tool.

## Contents

- `step1_rag_example.py` — step 1: run a single RAG example over a small course corpus.
- `step2_boiler_ta_server.py` — step 2: MCP server exposing the Purdue TA tools.
- `step2_boiler_ta_agent.py` — step 2: LangGraph agent that uses the Purdue TA tool set.
- `outline.md` — workshop structure and teaching flow.
- `requirements.txt` — workshop dependencies.

## Setup

Create a fresh environment and install the workshop dependencies:

```bash
python3 -m pip install -r requirements.txt
```

### (Optional) Create a Conda environment named `2ndWorkshop`

To create a reproducible Conda environment and keep the workshop sandboxed, run:

```bash
conda create -n 2ndWorkshop python=3.10 pip -y
conda activate 2ndWorkshop
pip install -r requirements.txt
```

Or, if you have an `environment.yml` you prefer to use:

```bash
conda env create -f environment.yml -n 2ndWorkshop
conda activate 2ndWorkshop
```

### Free local / remote models

This workshop does not require paid API keys. The examples default to free local or Hub models and use local retrieval.

- The RAG example uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- The default LLM can be a free Hugging Face model such as `google/flan-t5-small`.
- You can switch to another free Hub model by setting environment variables.

Example:

```bash
export USE_HF_HUB=1
export HF_MODEL=google/flan-t5-small
```

No OpenAI API key is required for the default flow.

## Model backends (quick guide)

You can swap LLM backends for demos or to use campus resources. Keep the code unchanged; set environment variables before running.

- Hugging Face Hub (recommended small models for the workshop):
	- `export USE_HF_HUB=1`
	- `export HF_MODEL=google/flan-t5-small`

- Local Hugging Face model (CPU/GPU):
	- `export LOCAL_MODEL=tiiuae/falcon-7b-instruct`
	- The demo will load the model locally via `transformers`.

- Ollama (local inference server):
	- Set `LLM_MODEL="ollama:llama3.1"` and run Ollama locally.

- OpenAI:
	- Set `LLM_MODEL="openai:gpt-4o"` and provide `OPENAI_API_KEY` in your environment.

- Purdue GenAI (campus HTTP endpoint):
	- Set `PURDUE_GENAI_KEY` to your API key and (optionally) override URL:
		- `export PURDUE_GENAI_KEY=...`
		- `export PURDUE_GENAI_URL=https://genai.rcac.purdue.edu/api/chat/completions`
	- Example usage (Python `requests`) https://www.rcac.purdue.edu/knowledge/genaistudio?all=true:

```python
import os, requests
url = os.environ.get("PURDUE_GENAI_URL", "https://genai.rcac.purdue.edu/api/chat/completions")
headers = {"Authorization": f"Bearer {os.environ['PURDUE_GENAI_KEY']}", "Content-Type": "application/json"}
body = {"model": "llama3.1:latest", "messages": [{"role": "user", "content": "What are the office hours?"}], "stream": False}
resp = requests.post(url, headers=headers, json=body)
print(resp.status_code, resp.text)
```

Notes:
- Use `LLM_MODEL` and `LLM_BASE_URL` (if needed) when using `init_chat_model()` in advanced demos (see Geoff examples in `Geoff/2026-purdue-ai-showcase`).
- For classroom reproducibility prefer small Hub models or local CPU-friendly models to avoid GPU/PKI issues.

## Running the workshop flow

### Step 1: Run the RAG example

```bash
python step1_rag_example.py
```

This script builds a vector store from the Purdue TA corpus in `boilermaker_ta_data/knowledge_base.json`, performs retrieval over those documents, and answers sample questions using retrieved context. Use this step to demonstrate grounding before moving to the agentic demo.

### Step 2: Run the Purdue TA LangGraph agent

Start the MCP tool server in one terminal:

```bash
python step2_boiler_ta_server.py
```

Then run the agent in a second terminal:

```bash
python step2_boiler_ta_agent.py
```

The agent uses three Purdue TA tools:

- `search_knowledge_base(query)` — searches the local course knowledge base.
- `get_academic_calendar(query)` — reads the course calendar and schedule.
- `create_notification(subject, body)` — writes a notification announcement.

Additionally, the MCP server exposes `retrieve_docs(query, k=3)` which returns the top-k retrieved passages from the knowledge base (embeddings + FAISS). The LangGraph agent can call this tool for retrieval-augmented actions.

## Notes

- The main workshop flow is intentionally simple: one RAG example first, then the Purdue TA LangGraph agent.
- For a full workshop, discuss how retrieval grounds answers and how tool calls extend agent capabilities.

**Note on MCP vs LangGraph:** The hands-on demo in this folder uses `step2_boiler_ta_server.py` (an MCP tool server) together with `step2_boiler_ta_agent.py` (a LangGraph agent) to show how MCP exposes discoverable tools and how LangGraph orchestrates tool calls in an agent loop. A separate explanatory example was removed to avoid duplication — use the server and agent files for the integrated demo.
