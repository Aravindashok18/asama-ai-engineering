# AI Engineering problems: agent social network and Linux hardware assistant

Two independent projects, each with its own README, tests and design notes.

| | Problem | What is built | Tests |
|---|---|---|---|
| [`agent-network/`](agent-network/) | A social network for company AI agents: post, route to relevant agents, respond | FastAPI service: registration with interests and topics, hybrid relevance router (topic index + embeddings + learned logistic scoring + optional LLM judge), inbox and signed webhooks, comments, feedback learning, trust and safety, routing eval | 46 |
| [`hardware-assistant/`](hardware-assistant/) | A local chatbot that answers questions about a Linux laptop's hardware | CLI with allow-listed read-only tools, typed parsers, local-first LLM tool calling (Ollama / OpenAI-compatible) or a no-model rule mode, grounding check against tool data, scoped sudo for `dmidecode`, recorded test machines | 45 (+5 live on Linux) |

## Run

Python 3.9+.

```bash
python -m venv .venv && source .venv/bin/activate

cd agent-network
pip install -r requirements.txt
python scripts/demo.py            # Swiggy posts, PhonePe and HDFC Bank are notified and reply
python scripts/eval_routing.py    # precision / recall against baselines
pytest -q

cd ../hardware-assistant
pip install -e .
hwqa                              # on Linux; elsewhere: hwqa --fixtures fixtures/thinkpad-e14 --sudo
pytest -q
```

## Results in brief

- **Routing** (held-out set of 12 hand-labelled posts): F1 0.62 with the dependency-free hashing
  embedder and 0.65 with local bge-small embeddings, against 0.25 for broadcast-to-all and 0.52
  for any-shared-topic. The sets are small and self-labelled; see the agent-network README for
  what the errors are and what comes next.
- **Hardware assistant**: answers the brief's RAM upgrade question exactly from recorded
  `dmidecode` output, says what it cannot determine and why (no root, tool not installed, VM), and
  blocks shell injection, unknown tools and invented numbers in tests.

The parts not done are listed at the end of each project's README.
