# 🎯 Career Discovery Assistant (CDA)
 From **Team GenAI-Uncharted** [*exploring new territory in AI-powered career discovery*]..\
powered by Intel- **Qwen3-8B** LLM Model

Built with Qwen3-8B + Qdrant Semantic Search + Streamlit

---

## What is CDA?

The Career Discovery Assistant is an AI-powered job matching application that helps busy professionals
discover the most relevant job opportunities based on their profile — without manually filtering
through hundreds of listings.

| Feature | Details |
|---|---|
| **Semantic Job Search** | Qdrant + fastembed vector search matches jobs by meaning, not just keywords |
| **AI-Powered Ranking** | Qwen3-8B ranks top matches and explains *why* each role fits the candidate |
| **Rich Profile Input** | Skills, preferred roles, industries, salary, work preference |
| **Interactive UI** | Streamlit chat-style interface with expandable job cards |
| **30 Curated Jobs** | Diverse postings across Fintech, AI/ML, Cloud, E-commerce, Healthcare, and more |

---

## Architecture

```
User Profile (sidebar)
        │
        ▼
  Natural Language Query
        │
        ▼
  Qdrant Semantic Search ──► Top 10 semantically similar jobs
        │
        ▼
  Qwen3-8B LLM ──────────► Ranked top 5 with personalised match explanations
        │
        ▼
  Streamlit Dashboard ────► Interactive job cards + AI analysis
```

---

## Getting Started

### Option A — Docker Compose (recommended)

```bash
# 1. Start Qdrant + CDA app
cd cda/
docker-compose up --build -d
      # if permission denied run with sudo privillage
      sudo docker compose up --build -d

# 2. Populate the Qdrant jobs collection
pip install qdrant-client fastembed
python tools/create_job_embeddings.py

      # if n/w issue or port issue at 6334 than run vector db at 6333 
      # sudo docker run -p 6333:6333 -d qdrant/qdrant
      # now run 
      python tools/create_job_embeddings.py

# 3. Open the app
open http://localhost:8502
```

### Option B — Local development

```bash
# 1. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 2. Install dependencies
cd cda/app
pip install -r requirements.txt

# 3. Populate jobs
cd ../tools
pip install -r requirements.txt
python create_job_embeddings.py

# 4. Run the app
cd ../app
streamlit run main.py
```

---

## Project Structure

```
cda/
├── app/
│   ├── main.py              # Streamlit app (profile → search → rank → display)
│   ├── requirements.txt
│   └── Dockerfile
├── tools/
│   ├── create_job_embeddings.py   # Populate Qdrant with 30 job postings
│   └── requirements.txt
├── docker-compose.yaml
└── README.md
```

---

## Business SLAs

| Parameter | Value |
|---|---|
| Model throughput | 40–80 tokens/sec |
| Concurrent users | 50–100 |
| Input token size | 500–1,200 tokens |
| Output token size | 150–400 tokens |
| TTFT (min/max) | 0.5 sec – 2 sec |

---

## Key KPIs

- **Time-to-Apply Reduction:** Hours → Minutes (↓ 90%)
- **Job Relevance Accuracy:** > 85% (AI-ranked matches)
- **Application Throughput:** 5× more applications per week
- **Manual Effort:** Drastically reduced via automation

---

## Model

**Primary:** `Qwen/Qwen3-8B` — chosen for strong instruction-following, reasoning,
agent capabilities, and multilingual support, making it ideal for understanding
career profiles, ranking roles, and generating personalised explanations.

---

*Team GenAI-Uncharted — exploring new territory in AI-powered career discovery.*
