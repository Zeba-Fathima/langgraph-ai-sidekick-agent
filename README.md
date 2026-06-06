# 🤖 LangGraph AI Sidekick Agent

<p align="center">
  <img src="https://img.shields.io/badge/LangGraph-Agentic%20AI-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Qwen-2.5%207B-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Playwright-Browser%20Automation-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/LangSmith-Observability-purple?style=for-the-badge">
</p>

<p align="center">
  <b>An Autonomous AI Co-Worker Powered by LangGraph, Ollama, Playwright, LangSmith, and Tool-Using Agents</b>
</p>

---

# 🌟 Overview

LangGraph AI Sidekick Agent is a fully autonomous AI assistant designed to function as a personal digital co-worker.

Unlike traditional chatbots that simply answer questions, this agent can:

✅ Research information online

✅ Browse websites autonomously

✅ Execute Python code

✅ Create and manage files

✅ Query Wikipedia

✅ Evaluate its own responses

✅ Continue working until tasks are completed

✅ Save outputs for future use

The system follows a Worker → Tools → Evaluator workflow, allowing it to reason, act, verify, and improve its answers before returning them to the user.

---

# 🚀 What Makes This Different?

Most chatbots:

```text
User → LLM → Response
```

This Sidekick:

```text
User
 ↓
Worker Agent
 ↓
Tool Usage
 ↓
Information Gathering
 ↓
Self Evaluation
 ↓
Task Verification
 ↓
Final Response
```

The agent is capable of multi-step reasoning and tool execution rather than generating a single response.

---


img width="735" height="856" alt="image" src="https://github.com/user-attachments/assets/bec9884b-9094-4832-a2a7-e1538307cca2" />

---

# 🧠 Core Components

## 1️⃣ Worker Agent

The Worker Agent is the brain of the system.

Responsibilities:

* Understand user requests
* Determine which tools are needed
* Execute tool calls
* Gather information
* Create files
* Generate final responses



---

## 2️⃣ Tool Layer

The Sidekick has access to multiple tools.

### 🔎 Web Search Tool

Uses Serper API to perform internet searches.

---

### 🌐 Browser Tool (Playwright + Chromium)

Allows the agent to:

* Open websites
* Read webpages
* Navigate pages
* Extract information

Powered by:

```text
Playwright
+
Chromium
```

---

### 📚 Wikipedia Tool

Provides encyclopedia-style factual information.

---

### 🐍 Python REPL Tool

Used for:

* Calculations
* Data analysis
* Data transformations
* Report generation

Not used for:

❌ File writing

---

### 📂 File Management Tool

Creates and manages files inside:

```text
sandbox/
```

---

### 📲 Push Notification Tool

Can send notifications directly to the user.

Powered by:

```text
Pushover API
```

---

## 3️⃣ Evaluator Agent

The Evaluator acts as a quality checker.

Responsibilities:

* Verify completeness
* Check success criteria
* Detect missing information
* Decide whether more work is needed


---

# 🦙 Why Ollama?

Ollama allows local execution of LLMs.

Instead of:

```text
User
 ↓
OpenAI
```

We use:

```text
User
 ↓
Ollama
 ↓
Qwen 2.5
```

Benefits:

✅ Free

✅ Local execution

✅ No API limits

✅ No cloud dependency

---

# 🔬 Why LangSmith?

LangSmith provides complete visibility into the agent.

You can inspect:

* Worker reasoning
* Tool calls
* Tool outputs
* Evaluator decisions
* Graph routing

Example:

```text
Worker
 ↓
Search Tool
 ↓
Browser Tool
 ↓
Evaluator
 ↓
Final Answer
```

This makes debugging and improving agents significantly easier.

---


# 🔐 Environment Variables

Create:

```text
.env
```

Add:

```env
SERPER_API_KEY=your_serper_api_key

PUSHOVER_TOKEN=your_pushover_token
PUSHOVER_USER=your_pushover_user_key

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=langgraph-ai-sidekick-agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

---

# ▶️ Running The Application

Start Ollama:

```bash
ollama serve
```

Run the application:

```bash
uv run app.py
```

Open:

```text
http://127.0.0.1:7860
```

---



# 📄 Resume Description

Built an autonomous AI Sidekick Agent using LangGraph, LangChain, Ollama, Qwen, Playwright, Gradio, and LangSmith. Implemented a worker-evaluator architecture capable of web research, browser automation, Python execution, file management, and iterative task completion using tool-calling agents.

---

# 👩‍💻 Author

### Zeba Fathima

AI Engineer | Machine Learning Engineer | Generative AI Enthusiast


