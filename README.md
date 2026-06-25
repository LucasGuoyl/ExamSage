# 🎓 ExamSage

> Upload your slides and past papers → AI ranks the most likely exam topics → it auto-generates practice questions with answers, exportable to PDF in one click.

A **training-free** exam-prediction system. It reuses off-the-shelf large language models and open-source embeddings, running a *retrieval → multi-signal fusion → generation* pipeline that predicts the **knowledge points most likely to be tested** from your course materials, and generates matching practice questions for each.

Core design goals: **works with few (or zero) past papers**, and **generalizes across courses** — switching subjects only takes a one-line course description.

---

## ✨ Features

- 📥 **Many input formats** — slides as PDF / PPTX / Markdown; past papers as JSON / PDF / Markdown.
- 🔥 **Exam heatmap** — semantically aligns past questions to slide passages to quantify each topic's historical exam frequency.
- 🧠 **Few-shot friendly** — a built-in LLM pedagogical prior takes over when past papers are scarce (even zero); weights shift smoothly with data volume.
- 🌐 **Cross-course** — scoring doesn't rely on subject-specific data; one line of course context adapts it to power systems, chemistry, CS, economics, anything.
- 🚫 **Smart filtering** — rules + LLM jointly detect and drop title pages, agendas, recaps, background, and admin text that aren't real knowledge points.
- ✍️ **Question generation** — few-shot mimics the style of past papers, then an LLM-as-judge ranks the candidates.
- 📄 **One-click export** — Markdown / JSON / **PDF** (renders English, CJK, and math symbols; print-ready).
- 🖥️ **Web UI** — a Streamlit app: drag-and-drop upload, click to run, preview online, download.
- 🌍 **Output language control** — questions/answers default to the source material's language; force English or Chinese with one switch.

---

## 🛠️ Tech Stack

| Area | Choice | Purpose |
|------|--------|---------|
| Language | **Python 3.10+** | Whole stack |
| Web UI | **Streamlit** | Upload / configure / display / download |
| Embeddings | **sentence-transformers + BGE** | Encode slides & questions into vectors |
| Vector search | **FAISS** (CPU) | Cosine-similarity retrieval (question ↔ slide alignment) |
| LLM | **OpenAI SDK** | Works with DeepSeek / OpenAI / Qwen / any OpenAI-compatible API |
| Data models | **Pydantic v2** | Typed models & serialization |
| Doc parsing | **PyMuPDF** (PDF) · **python-pptx** (PPTX) | Extract slide text |
| PDF export | **ReportLab** | Render a print-ready PDF from the structured report |
| Robustness | **tenacity** | Exponential-backoff retries on API calls |
| CLI output | **rich** | Progress and colored logs |

### Models used

- **Embedding**: `BAAI/bge-large-zh-v1.5` (bilingual, runs locally, free, ~1.3 GB auto-download on first use). English-only courses can use `bge-large-en-v1.5`; OpenAI `text-embedding-3` is also supported.
- **LLM**: default `deepseek-chat` (good, cheap). Swap in any OpenAI-compatible model (GPT-4o, Qwen, …). **Used in four places: pedagogy scoring, knowledge-point summarization, question generation, and answer reranking.**

> 💡 The system **trains and fine-tunes nothing** — it works out of the box.

---

## 🧩 Architecture & Algorithms

```
        Slides (PDF/PPTX/MD) + Past papers (JSON/PDF/MD) + Tutorials/Syllabus (optional)
                                │
                                ▼
   ┌──────────────────── Stage 1 · Ingest ──────────────────────┐
   │  PyMuPDF / python-pptx extract text  →  sliding-window Chunks │
   └──────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 ┌──────────────┐      ┌─────────────────┐      ┌──────────────────┐
 │ Stage 2 · C  │      │ Stage 2.5 · Score│      │ Stage 2.6 · Filter│
 │   Align      │      │  LLM pedagogy    │      │  drop non-content │
 │ Q→slide ret. │      │  + rule signals  │      │ titles/agenda/etc │
 │ FAISS heatmap│      │ (no history dep) │      │  rules + LLM gate │
 └──────────────┘      └─────────────────┘      └──────────────────┘
        └───────────────────────┼───────────────────────┘
                                ▼
   ┌──────────────────── Stage 3 · D · Fuse ────────────────────┐
   │  weighted sum of: exam_freq · pedagogy · structural ·        │
   │  emphasis · teaching-time · tutorial-overlap · syllabus      │
   │  ★ Adaptive weights: few papers → LLM prior leads (0.45);    │
   │    many papers → historical frequency leads (0.35)           │
   │  greedy embedding clustering → Knowledge Points              │
   └──────────────────────────────────────────────────────────────┘
                                │
   ┌──────────────── Stage 3.5 · Summarise ─────────────────────┐
   │  LLM writes per topic: clean title + concept + 3 exam angles │
   └──────────────────────────────────────────────────────────────┘
                                │
   ┌──────────────────── Stage 4 · E · Generate ────────────────┐
   │  few-shot questions in past-paper style → LLM-as-judge +     │
   │  embedding-novelty rerank                                     │
   └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
             📊 Topic ranking + 📝 Practice questions + 💡 Answers
                  exported as  report.md / predictions.json / report.pdf
```

### Key algorithms

1. **Semantic alignment (Stage C)** — chunks and questions are BGE-encoded, L2-normalized, and indexed in a FAISS `IndexFlatIP` (inner product = cosine). Each past question retrieves its top-K similar chunks; contributions (similarity × recency weight) accumulate and min-max normalize into an **exam heatmap**.
2. **Pedagogical prior (Scorer)** — chunks are batched to the LLM, which rates each one's exam probability from first principles (definition / derivation / application / background…), **without any course-specific history**. This is the source of the few-shot and cross-course ability. A zero-cost rule signal (definitions, theorems, action verbs, emphasis-keyword density) complements it.
3. **Non-content filter** — regex rules catch title slides, agendas, recaps, references, acknowledgements, and admin text; an LLM-score threshold backs it up. A safety fallback reverts to no-filter if too much is dropped.
4. **Adaptive fusion (Stage D)** — seven signals are weighted-summed; weights **linearly interpolate** between "sparse" and "normal" sets based on the number of past papers — less data leans on the LLM prior, more data leans on historical frequency.
5. **Knowledge-point clustering** — greedy embedding-similarity clustering merges near-duplicate chunks into topics.
6. **Styled generation & rerank (Stage E)** — few-shot past papers anchor the style; an LLM-as-judge (style / quality / novelty) combined with embedding novelty scores and keeps the best per topic.
7. **Offline evaluation** — a built-in hold-out evaluator (split by year) reports **Top-K Coverage** and **MRR** against a random baseline, for tuning the fusion weights.

---

## 🚀 Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

> First run downloads the BGE embedding model (~1.3 GB); it's cached afterward.

### 2. Configure your API key

Works with any OpenAI-compatible service (we recommend [DeepSeek](https://platform.deepseek.com) — cheap and strong).

- **Web UI**: no config file — paste the key in the sidebar.
- **CLI**: copy the template and fill in the key (`config.yaml` is git-ignored):
  ```bash
  cp config.example.yaml config.yaml   # then edit config.yaml and set api_key
  ```

### 3. Launch the Web UI (recommended)

```bash
streamlit run app.py
```

Opens **http://localhost:8501**. In the UI: enter your API key and course name → drag-and-drop slides (and past papers) → click **🚀 Run Analysis** → view the ranking/questions → download the PDF.

### 4. CLI mode

```bash
python examples/run_university.py --course <course_dir> --config config.yaml
```

Writes `output/report.md`, `output/predictions.json`, and `output/report.pdf`.

---

## 📁 Preparing Data

Organize one course like this (only `slides/` is required):

```
my_course/
├── slides/          # required   .pdf / .pptx / .md
├── past_papers/     # optional (more is better)   .json / .pdf / .md
├── tutorials/       # optional
└── syllabus.md      # optional
```

Recommended past-papers JSON format (downloadable as a template in the Web UI):

```json
[
  {
    "id": "2023_q1",
    "year": 2023,
    "text": "Full question text…",
    "type": "computation",
    "answer": "Reference answer key points (optional)"
  }
]
```

---

## 🗂️ Project Structure

```
ExamSage/
├── app.py                  # Streamlit Web UI entry point
├── config.example.yaml     # config template (copy to config.yaml)
├── requirements.txt
├── exam_predictor/         # core engine package
│   ├── pipeline.py         # end-to-end orchestration + Markdown report
│   ├── ingest.py           # PDF / PPTX / MD parsing
│   ├── chunker.py          # text chunking
│   ├── embedder.py         # local BGE / API embeddings (switchable)
│   ├── vector_store.py     # FAISS wrapper
│   ├── aligner.py          # Stage C: alignment + exam heatmap
│   ├── scorer.py           # pedagogical prior + non-content filter
│   ├── fusion.py           # Stage D: adaptive multi-signal fusion
│   ├── generator.py        # knowledge-point summary + Stage E generation
│   ├── reranker.py         # LLM-as-judge rerank
│   ├── exporter.py         # PDF export (ReportLab)
│   ├── evaluator.py        # hold-out evaluation (Coverage / MRR)
│   └── schema.py           # Pydantic data models
├── examples/
│   ├── run_university.py   # single-course prediction
│   └── run_gaokao.py       # hold-out evaluation
└── tests/
    └── test_basic.py       # smoke tests
```

---

## 🔒 Security

- Key-bearing files (`config.yaml`, `.env`, …) are git-ignored and **never committed**.
- Run `git status` before committing to confirm no key file is tracked.
- **Never** put a real API key in a committed file or in source.

---

## 📜 License

Released under the [MIT License](LICENSE).

---

*Predictions and generated content are for study reference only — always defer to your course materials and instructor.*
