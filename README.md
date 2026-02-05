# Agentic EDA System using LangGraph + Groq

## Overview

This project implements a **Data Science Agent** — an autonomous, state-driven system that can reason about data, plan analytical steps, execute real computations, and synthesize insights, much like a human data scientist.

Instead of a monolithic prompt-based EDA, the system decomposes data science work into **intent understanding, analytical planning, execution, and insight generation**, orchestrated via **LangGraph**.
This project implements an **Agentic Exploratory Data Analysis (EDA)** system using **LangGraph**, **LangChain**, and **Groq LLMs**, wrapped with a **Gradio UI** for interactive usage. The system behaves like an autonomous data analyst: it interprets user intent, plans analysis steps, performs EDA programmatically, and returns structured insights.

The notebook demonstrates how **stateful agent graphs** can be used to orchestrate reasoning, tool execution, and response generation in a clean, production-aligned way.

---

## Key Objectives

* Build a **Data Science Agent**, not a static EDA script
* Enable the agent to *think in steps* like a professional data scientist
* Separate **reasoning, execution, and interpretation**
* Ensure results are grounded in **actual computation**, not hallucination
* Provide an interactive interface for non-technical users
* Build a multi-step **EDA agent** instead of a single-prompt LLM
* Use **LangGraph** to explicitly model reasoning and execution flow
* Separate **planning**, **execution**, and **response synthesis**
* Enable interactive EDA via **Gradio**

---

## Tech Stack

| Component       | Purpose                                                      |
| --------------- | ------------------------------------------------------------ |
| LangGraph       | Agent orchestration via state graph                          |
| LangChain       | Prompting, schemas, tool wiring                              |
| Groq (ChatGroq) | High-speed LLM inference                                     |
| Pydantic        | Structured intent & state validation                         |
| Gradio          | User-facing interactive UI                                   |
| ezyml           | Lightweight ML training, evaluation, and metric optimization |
| Python          | Core execution                                               |

---

## High-Level Architecture

### Agent Execution Graph

The following diagram represents the **end-to-end agentic data science pipeline**, implemented using LangGraph. Each node corresponds to a distinct cognitive or computational responsibility, mirroring real-world data science workflows.

<!-- PLACEHOLDER: Agentic Data Science Execution Graph -->

> 📌 **Agent Execution Graph Diagram**
<img width="131" height="730" alt="image" src="https://github.com/user-attachments/assets/e6382419-13a7-4439-b83e-916847286d78" />

**Flow Explanation:**

* `__start__` → Entry point
* `planner` → Interprets user intent and plans steps
* `eda` → Performs exploratory data analysis
* `feature_eng` → Applies feature engineering logic
* `ml` → Trains and evaluates models
* `reflect` → Reviews model performance and decides iteration
* `response` → Produces human-readable insights
* `__end__` → Terminates execution

The dashed feedback loop between **ML ↔ Reflect** enables iterative improvement, a hallmark of senior-level and AutoML systems.

The system is built around a **state-driven agent graph**.

### Agent State

The agent maintains a shared mutable state (`AgentState`) that flows across nodes:

* `query`: User EDA request
* `intent`: Parsed analytical intent
* `plan`: Step-by-step analysis plan
* `analysis`: Computed EDA results
* `final_response`: Natural language insights

This makes the agent **transparent, debuggable, and extensible**.

---

## Agent Graph Flow (Conceptual Diagram)

```
User Query
   │
   ▼
[Intent Detection]
   │
   ▼
[Planner Agent]
   │
   ▼
[EDA Executor]
   │
   ▼
[Insight Generator]
   │
   ▼
 Final Response
```

Each block is a **LangGraph node**, and edges define deterministic execution order.

---

## Core Components Explained

### 1. Intent Detection

* Uses a Pydantic schema (`IntentSchema`)
* Extracts *what kind of EDA* the user wants (summary, correlation, distribution, etc.)
* Prevents vague or unsafe analysis execution

### 2. Planner Agent

* Uses a system prompt (`PLANNER_PROMPT`)
* Converts intent into **explicit EDA steps**
* Example:

  * Load dataset
  * Inspect columns
  * Generate statistical summary
  * Identify missing values

This mirrors how a human data scientist thinks.

---

### 3. EDA Execution Agent

* Executes real Python EDA logic
* Computes:

  * Shape of dataset
  * Column types
  * Descriptive statistics
  * Missing value counts

The LLM **does not hallucinate results** — it reasons about real outputs.

---

### 4. Insight Generator

* Transforms raw statistics into **human-readable insights**
* Highlights:

  * Potential data quality issues
  * Feature distributions
  * Analytical next steps

---

## LangGraph Implementation Details

```text
StateGraph(AgentState)
  ├── planner_node
  ├── eda_node
  └── response_node
```

* Nodes are registered with `graph.add_node()`
* Execution order defined with `graph.add_edge()`
* Compiled into a runnable app via `graph.compile()`

This ensures:

* Deterministic flow
* No hidden chain-of-thought
* Easy debugging & logging

---

## Gradio Interface

The agent is exposed through a **production-style Gradio UI**, allowing users to interact with the Data Science Agent using natural language and a dataset path.

<!-- PLACEHOLDER: Gradio UI Screenshot -->

> 📌 **Agentic Data Science Assistant – Gradio UI**
<img width="1704" height="512" alt="image" src="https://github.com/user-attachments/assets/29d0988e-b0af-446c-a5c5-b4559fd1a72f" />


### UI Capabilities

* Natural language task specification (e.g., *"Train XGBoost and maximize F1 score"*)
* Dataset path input (CSV)
* One-click execution
* Structured, scrollable agent responses

This interface abstracts away code-level complexity and makes the agent usable by:

* Analysts
* ML engineers
* Non-technical stakeholders

The UI reinforces the goal of the project: **industrial-grade autonomous data science**, not notebook-bound experimentation.

The notebook exposes the agent using **Gradio**, enabling:

* Natural language EDA queries
* Instant execution feedback
* Clean UI for non-technical users

This makes the system suitable for:

* Data analysts
* Product managers
* ML engineers

---

## Example Usage

### Using the Agent Directly

**Input**:

> "Perform an exploratory data analysis on the dataset and summarize key insights"

**Output**:

* Dataset dimensions
* Feature-level statistics
* Missing value warnings
* Suggested next analysis steps

---

## Using the `ezyml` Package

This project integrates seamlessly with **`ezyml`**, a lightweight ML utility package that abstracts model training, evaluation, and metric optimization into simple, declarative calls.

### Why `ezyml`?

* Removes boilerplate ML code
* Standardizes training & evaluation
* Prevents LLMs from hallucinating model logic
* Makes the agent *tool-driven*, not *text-driven*

Within the agent pipeline, `ezyml` is typically invoked inside the **ML node**.

### Installation

```bash
pip install ezyml
```

### Example: Training a Classification Model

```python
from ezyml import train_classifier

results = train_classifier(
    data=df,
    target="label",
    model="xgboost",
    metric="f1",
    test_size=0.2,
    random_state=42
)
```

### Returned Artifacts

The `train_classifier` call returns a structured object containing:

* Trained model
* Accuracy, F1, ROC-AUC scores
* Confusion matrix
* Feature importance (if supported)

These outputs are passed back into the agent state for:

* Reflection
* Iteration decisions
* Final explanation generation

### Role in the Agentic Loop

In this system:

* **Planner** decides *what model & metric to use*
* **ML node** executes training via `ezyml`
* **Reflect node** evaluates results and decides whether to retry
* **Response node** explains performance in plain English

This makes `ezyml` a **critical execution backbone** of the AutoML pathway.

---

**Input**:

> "Perform an exploratory data analysis on the dataset and summarize key insights"

**Output**:

* Dataset dimensions
* Feature-level statistics
* Missing value warnings
* Suggested next analysis steps

---

## Agent-to-Industry Role Mapping

This system can be directly mapped to real-world **data science industry roles**, based on capability maturity and autonomy. The same architecture scales naturally from a junior analyst to an AutoML-grade agent.

---

### Level 1: Junior Data Analyst

**Equivalent Responsibilities:**

* Understand business or stakeholder questions
* Perform basic EDA
* Generate summary statistics
* Identify missing values and obvious data issues

**Agent Capabilities at This Level:**

* Intent parsing from natural language queries
* Dataset inspection (shape, columns, types)
* Descriptive statistics generation
* Basic data quality checks

**How the Notebook Matches:**

* Intent Detection node mimics requirement understanding
* EDA Executor performs standard analyst-level checks
* Insight Generator explains results in plain language

➡️ At this level, the agent behaves like a **competent junior analyst** executing well-defined tasks.

---

### Level 2: Mid-Level Data Scientist

**Equivalent Responsibilities:**

* Decide *what analysis to run next*
* Identify relationships between variables
* Assess data readiness for modeling
* Propose next analytical steps

**Agent Capabilities at This Level:**

* Planner agent decomposes requests into ordered steps
* Structured reasoning over analytical flow
* Identification of potential risks (missing data, skewness)
* Suggestion of further analysis

**How the Notebook Matches:**

* Planner node mirrors a data scientist’s analytical reasoning
* StateGraph enforces logical sequencing
* Insights include recommendations, not just numbers

➡️ Here, the agent functions like a **mid-level data scientist**, capable of reasoning beyond rote EDA.

---

### Level 3: Senior Data Scientist / Analytics Lead

**Equivalent Responsibilities:**

* Design end-to-end analysis strategy
* Ensure analytical rigor and reproducibility
* Review and validate junior analysts’ work
* Communicate insights to stakeholders

**Agent Capabilities at This Level:**

* Deterministic, auditable execution via LangGraph
* Explicit state tracking (intent → plan → results)
* Explainable outputs suitable for decision-makers
* Reduced hallucination risk through real computation

**How the Notebook Matches:**

* State-driven architecture enforces best practices
* Clear separation of reasoning vs execution
* Outputs are structured, reviewable, and reproducible

➡️ At this stage, the agent acts as a **senior DS reviewer and orchestrator**.

---

### Level 4: AutoML / Autonomous Data Science Agent

**Equivalent Responsibilities:**

* Fully automated data understanding
* Adaptive analysis based on dataset properties
* Minimal human intervention
* Continuous improvement via feedback loops

**Agent Capabilities (Next Evolution):**

* Automatic visualization generation
* Feature engineering proposals
* Model selection and evaluation
* Iterative refinement using agent memory

**How This Project Evolves There:**

* Add modeling and evaluation nodes
* Introduce persistent memory (vector DB / cache)
* Enable agent self-critique and retry loops

➡️ With these extensions, the system becomes an **AutoML-grade autonomous data science agent**.

---

### Summary Table

| Industry Role  | Human Equivalent    | Agent Behavior              |
| -------------- | ------------------- | --------------------------- |
| Junior Analyst | Executes EDA        | Intent → Stats → Summary    |
| Mid-Level DS   | Plans analysis      | Reasoned EDA flow           |
| Senior DS      | Reviews & validates | Deterministic orchestration |
| AutoML Agent   | Fully autonomous    | Self-directed data science  |

---

## Results & Observations

* Agentic EDA is **more reliable** than single-prompt analysis
* LangGraph enforces **discipline in reasoning flow**
* State-based design enables:

  * Memory
  * Debugging
  * Incremental upgrades

Performance-wise, **Groq LLMs** provide near real-time responses, making this practical for interactive workflows.

---

## Strengths of This Approach

✅ No prompt spaghetti
✅ Deterministic execution
✅ Real computation (not hallucinated stats)
✅ Production-aligned architecture

---

## Limitations

* Notebook-based (needs packaging for prod)
* No persistent memory store yet
* Visualization graphs not auto-generated (text-based EDA)

---

## Future Improvements

* Add automated **matplotlib / seaborn plots**
* Introduce dataset upload support
* Cache EDA results
* Extend agent with feature engineering & modeling

---



> Versions are intentionally not pinned to allow flexibility during experimentation. For production, pin versions after validation.

---

## License

This project is released under the **MIT License**, making it free to use, modify, distribute, and include in commercial products.

Create a file named `LICENSE` with the following content:

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Conclusion

This notebook is a concrete implementation of a **Data Science Agent** — an autonomous system capable of performing exploratory analysis with reasoning, structure, and accountability.

By combining **LangGraph’s explicit control flow**, **Groq’s low-latency LLMs**, and **real Python execution**, the agent behaves like a junior-to-mid level data scientist: it understands requests, plans methodically, executes safely, and communicates insights clearly.

This is not just automation.

This is **Data Science as an Agentic System**.

This notebook demonstrates a **production-grade pattern** for building intelligent data agents. By combining LangGraph's explicit control flow with Groq’s low-latency LLMs, the system achieves **trustworthy, explainable, and scalable EDA automation**.

This is not just EDA — it is **EDA as an agentic system**.
