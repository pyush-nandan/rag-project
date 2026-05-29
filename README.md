<p align="center">
  <img src="https://img.shields.io/badge/LLM-LLaMA_3.1-blueviolet?style=for-the-badge" alt="LLaMA 3.1"/>
  <img src="https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-blue?style=for-the-badge" alt="MiniLM"/>
  <img src="https://img.shields.io/badge/Reranker-Cohere_v4-orange?style=for-the-badge" alt="Cohere"/>
  <img src="https://img.shields.io/badge/VectorDB-ChromaDB-green?style=for-the-badge" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Eval-RAGAS-red?style=for-the-badge" alt="RAGAS"/>
</p>

# ⚖️ LegalEase

**A production-grade RAG system for Indian legal case law — powered by hybrid retrieval, neural reranking, and automated quality evaluation.**

LegalEase helps legal professionals, law students, and researchers query Indian court judgments and statutory provisions using natural language. It retrieves relevant legal passages, reranks them for precision, and generates grounded answers with full source citations.

---

## ✨ Features at a Glance

| Feature | Description |
|---|---|
| 🔍 **Hybrid Search** | Combines semantic (vector) + keyword (BM25) retrieval for best recall |
| 🏆 **Neural Reranking** | Cohere Rerank v4 re-scores results to surface the most relevant chunks |
| 📄 **Live Document Upload** | Upload PDFs via the sidebar — auto-ingested into the knowledge base |
| 🏛️ **Statute vs. Case Tagging** | Documents auto-classified as `Statute` or `Case` with visual badges |
| 🎛️ **Prompt Versioning** | 5 built-in prompt personas + custom prompt mode |
| 📊 **RAGAS Evaluation** | Automated faithfulness, relevancy, precision & recall scoring |
| 🔄 **CI/CD Quality Gate** | GitHub Actions pipeline that blocks merges if faithfulness drops |
| 💬 **Chat Interface** | Streamlit-powered conversational UI with source transparency |

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph USER["👤 User"]
        Q["Legal Question"]
        PDF["PDF Upload"]
    end

    subgraph UI["🖥️ Streamlit App"]
        CHAT["Chat Interface"]
        SIDEBAR["Sidebar Controls"]
        SOURCES["Source Viewer"]
    end

    subgraph INGESTION["📥 Ingestion Pipeline"]
        LOADER["PyPDF Loader"]
        CLEAN["Text Cleaner"]
        SPLIT["Legal-Aware Splitter"]
        CLASSIFY["Doc Classifier"]
    end

    subgraph RETRIEVAL["🔍 Retrieval Pipeline"]
        VEC["Vector Retriever (ChromaDB + MiniLM)"]
        BM25["BM25 Retriever (Keyword Search)"]
        HYBRID["Ensemble Retriever (50/50 Fusion)"]
        RERANK["Cohere Rerank v4 (Top 4 Results)"]
    end

    subgraph GENERATION["🤖 Generation"]
        PROMPT["Prompt Template (YAML Config)"]
        LLM["LLaMA 3.1 8B (via Groq)"]
        ANSWER["Grounded Answer"]
    end

    Q --> CHAT
    PDF --> SIDEBAR
    SIDEBAR --> LOADER
    LOADER --> CLEAN --> SPLIT --> CLASSIFY
    CLASSIFY -->|Vectors| VEC
    CLASSIFY -->|BM25 Index| BM25

    CHAT --> VEC
    CHAT --> BM25
    VEC --> HYBRID
    BM25 --> HYBRID
    HYBRID --> RERANK
    RERANK --> PROMPT
    PROMPT --> LLM
    LLM --> ANSWER
    ANSWER --> SOURCES
    SOURCES --> CHAT

    style USER fill:#1a1a2e,stroke:#C9A227,color:#E8E6E3
    style UI fill:#16213e,stroke:#4A90D9,color:#E8E6E3
    style INGESTION fill:#0f3460,stroke:#6DD49E,color:#E8E6E3
    style RETRIEVAL fill:#1a1a2e,stroke:#E07C5A,color:#E8E6E3
    style GENERATION fill:#16213e,stroke:#9B8EC4,color:#E8E6E3
```

---

## 🔄 Retrieval Pipeline — Deep Dive

The retrieval strategy uses a **three-stage pipeline** to maximize both recall and precision:

```mermaid
flowchart LR
    QUERY["🔎 User Query"] --> S1

    subgraph S1["Stage 1 — Dual Retrieval"]
        direction TB
        A["Semantic Search (ChromaDB · top-10)"]
        B["Keyword Search (BM25 · top-10)"]
    end

    S1 --> S2

    subgraph S2["Stage 2 — Fusion"]
        C["Reciprocal Rank Fusion (weights: 0.5 / 0.5, c = 60)"]
    end

    S2 --> S3

    subgraph S3["Stage 3 — Reranking"]
        D["Cohere Rerank v4 Pro (Top 4 chunks)"]
    end

    S3 --> OUT["📝 Context for LLM"]

    style S1 fill:#0f3460,stroke:#4A90D9,color:#E8E6E3
    style S2 fill:#1a1a2e,stroke:#E8C547,color:#E8E6E3
    style S3 fill:#16213e,stroke:#E07C5A,color:#E8E6E3
```

| Stage | What it Does | Why it Matters |
|---|---|---|
| **Dual Retrieval** | Runs both vector similarity and BM25 keyword search in parallel | Captures both semantic meaning *and* exact legal terminology |
| **Ensemble Fusion** | Merges results using Reciprocal Rank Fusion (RRF) | Balances recall across both retrieval paradigms |
| **Neural Reranking** | Cohere's cross-encoder re-scores all candidates | Pushes the most contextually relevant chunks to the top |

---

## 📥 Document Ingestion

Documents are processed differently based on their type:

```mermaid
flowchart TD
    PDF["📄 PDF File"] --> LOAD["PyPDF Loader"]
    LOAD --> CLEAN["Text Cleaning (Rejoin hyphens, collapse whitespace)"]
    CLEAN --> CHECK{"Filename starts with 'Section_'?"}
    
    CHECK -->|Yes| STATUTE["🏛️ Statute (Single chunk per document)"]
    CHECK -->|No| CASE["📋 Case Judgment (Chunked at 600 chars)"]
    
    STATUTE --> STORE["ChromaDB + BM25 Index"]
    CASE --> SPLITTER["Legal-Aware Splitter (Splits on HELD, FACTS, Issues, Judgement, etc.)"]
    SPLITTER --> STORE

    style CHECK fill:#1a1a2e,stroke:#E8C547,color:#E8E6E3
    style STATUTE fill:#0f3460,stroke:#4A90D9,color:#E8E6E3
    style CASE fill:#0f3460,stroke:#E07C5A,color:#E8E6E3
```

- **Statutes** (e.g., `Section_138_in_The_Negotiable_Instruments_Act_1881.PDF`) → stored as a single chunk to preserve statutory context
- **Case Judgments** → split using a legal-aware chunking strategy that respects judgment structure (`HELD`, `FACTS`, `Issues`, `Ratio`, etc.)

---

## 🎛️ Prompt Versioning System

LegalEase ships with **5 prompt versions**, each tailored for a different legal research task. Prompts are managed via `prompts.yaml` and selectable from the sidebar.

| Version | Persona | Best For |
|---|---|---|
| `v1` ⭐ | Friendly Explanatory Assistant | General legal Q&A — accessible language + Key Takeaway |
| `v2` | Contract Risk Analyst | Contract review — obligations, liabilities, risk factors |
| `v3` | Case Law Researcher | Case analysis — Facts, Issues, Holding, Reasoning |
| `v4` | Regulatory Compliance Officer | Statute interpretation — element breakdowns, exceptions |
| `v5` | Legal Drafting Assistant | Memo drafting — inline citations, formal legal tone |

> **Custom Prompt Mode:** Toggle on "Custom prompt mode" in the sidebar to write your own system prompt. Include `{context}` where retrieved documents should be injected.

---

## 📊 Evaluation Pipeline

LegalEase includes a rigorous evaluation framework using [RAGAS](https://docs.ragas.io/) with a curated test set of **20 questions** covering NI Act sections and landmark case law.

```mermaid
flowchart LR
    TS["📋 testset.csv (20 Q&A pairs with ground truth + context)"] --> RAG["🤖 RAG Chain (Generate responses)"]
    RAG --> CACHE["💾 Response Cache (rag_responses_cache.json)"]
    CACHE --> EVAL["📊 RAGAS Evaluate"]
    
    EVAL --> M1["Faithfulness"]
    EVAL --> M2["Answer Relevancy"]
    EVAL --> M3["Context Precision"]
    EVAL --> M4["Context Recall"]
    
    M1 --> GATE{"🚦 Quality Gate (Faithfulness >= 0.75?)"}
    GATE -->|Pass ✅| OK["Merge Allowed"]
    GATE -->|Fail ❌| BLOCK["Merge Blocked"]

    style GATE fill:#1a1a2e,stroke:#E8C547,color:#E8E6E3
    style OK fill:#0f3460,stroke:#6DD49E,color:#E8E6E3
    style BLOCK fill:#0f3460,stroke:#D94F4F,color:#E8E6E3
```

| Metric | What it Measures |
|---|---|
| **Faithfulness** | Are claims in the answer supported by the retrieved context? |
| **Answer Relevancy** | Does the answer address the user's question? |
| **Context Precision** | Are the retrieved chunks relevant to the question? |
| **Context Recall** | Does the retrieved context cover the ground-truth answer? |

### CI/CD Integration

A GitHub Actions workflow runs the evaluation pipeline on every pull request to `main`:

```yaml
# .github/workflows/evaluate.yml
on:
  pull_request:
    branches: [main]
  workflow_dispatch:
```

If **faithfulness** drops below **0.75**, the pipeline exits with code `1` — blocking the merge.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | LLaMA 3.1 8B Instant (via [Groq](https://groq.com)) | Fast inference for answer generation |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) | Document and query embedding |
| **Vector Store** | ChromaDB | Persistent vector storage and similarity search |
| **Keyword Search** | BM25 (`rank-bm25`) | Term-frequency based retrieval |
| **Reranker** | Cohere Rerank v4 Pro | Cross-encoder reranking |
| **Framework** | LangChain | Orchestration of chains and retrievers |
| **Frontend** | Streamlit | Chat UI with sidebar controls |
| **Evaluation** | RAGAS | Automated RAG quality metrics |
| **CI/CD** | GitHub Actions | Evaluation pipeline on PRs |
| **Document Parsing** | PyPDF | PDF loading and page extraction |

---

## 🚀 Quick Start

### Access the Deployed App:
 
- https://legelease.streamlit.app/

### Prerequisites

- Python 3.11+
- [Groq API Key](https://console.groq.com) (free tier available)
- [Cohere API Key](https://dashboard.cohere.com) (free tier available)

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/legalease.git
cd legalease

python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

### 3. Ingest Documents

Place your PDF files in the `data/` directory, then run:

```bash
python ingest_data.py
```

> The project ships with Indian NI Act sections (134–147) and several landmark Supreme/High Court judgments pre-loaded in `data/`.

### 4. Launch the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` with the custom dark theme applied automatically.

### 5. Run Evaluation (Optional)

```bash
python evaluate.py
```

Results are saved to `evaluation_results.csv`.

---

## 📁 Project Structure

```
legalease/
├── app.py                  # Streamlit chat UI
├── main.py                 # CLI interface for RAG chain
├── ingest_data.py          # Document loading, cleaning, chunking & indexing
├── retriever.py            # Hybrid retriever + Cohere reranker
├── evaluate.py             # RAGAS evaluation pipeline
├── prompts.yaml            # Prompt version configs (v1–v5)
├── testset.csv             # 20 curated evaluation Q&A pairs
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .streamlit/
│   └── config.toml         # Custom dark theme (navy + gold)
├── .github/
│   └── workflows/
│       └── evaluate.yml    # CI/CD evaluation pipeline
├── data/                   # PDF documents (statutes + case law)
├── chroma_db/              # ChromaDB persistent storage
└── bm25_retriever.pkl      # Serialized BM25 index
```

---

## 📜 Knowledge Base

The project ships with a curated corpus focused on **cheque dishonour law** under the Negotiable Instruments Act, 1881:

**Statutes** (14 sections)
- Sections 134–147 of the NI Act, including the critical Section 138 (cheque dishonour offence)

**Case Law** (12 landmark judgments)
- *Rangappa vs Sri Mohan* (2010) — Section 139 presumption scope
- *Dashrath Rupsingh Rathod vs State of Maharashtra* (2014) — Territorial jurisdiction
- *Icon Buildcon vs Aggarwal Developers* (2014) — Stop payment bona fides
- *Yogendra Pratap Singh vs Savitri Pandey* (2014)
- *M/S Laxmi Dyechem vs State of Gujarat* (2012)
- *Sony George Kurian vs State of Kerala* (2015)
- *Avneet Luthra vs Sunita Vijay* (2025) — Joint account holder liability
- And more...

---

## 📝 License

This project is for educational and research purposes.

---

<p align="center">
  <b>Built with ❤️ by Pyush Nandan</b>
</p>
