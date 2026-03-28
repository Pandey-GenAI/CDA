"""
Career Discovery Assistant (CDA)
─────────────────────────────────
A Streamlit app that uses Qdrant semantic search + Qwen3-8B to match
professionals with the most relevant job opportunities based on their profile.

Architecture:
  1. User fills in a career profile in the sidebar.
  2. Profile is converted to a natural-language search query.
  3. Qdrant (fastembed) performs semantic search against the 'jobs' collection.
  4. Qwen3-8B ranks the top matches and explains why each fits the candidate.
  5. Results are displayed as interactive cards with AI-generated insights.
"""

import os
import pickle
import numpy as np
import requests
import streamlit as st
from sklearn.preprocessing import normalize
from qdrant_client import QdrantClient

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
QDRANT_URL = os.environ.get("QDRANT_CLIENT", "http://localhost:6333")
QWEN_API_URL = os.environ.get(
    "QWEN_API_URL",
    "http://wiphack30qx5aw.cloudloka.com:8000/v1/chat/completions",
)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen/Qwen3-8B")
COLLECTION_NAME = "jobs"
TOP_K_SEARCH = 10   # candidates fetched from Qdrant
TOP_K_DISPLAY = 5   # top results explained by LLM

# Path to the sklearn TF-IDF + SVD model saved by create_job_embeddings.py
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding_model.pkl")

client_qdrant = QdrantClient(QDRANT_URL)


@st.cache_resource
def load_embedding_model():
    """Load the fitted TF-IDF + SVD model from disk (created by tools/create_job_embeddings.py)."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def encode_query(text: str) -> list:
    """Convert a text query to a normalised dense vector using the fitted TF-IDF vectorizer."""
    bundle = load_embedding_model()
    if bundle is None:
        return None
    tfidf_vec = bundle["vectorizer"].transform([text])
    dense_arr = tfidf_vec.toarray()
    # If query has no known vocabulary terms the vector will be all-zeros;
    # normalise returns zeros safely (no divide-by-zero).
    norm = np.linalg.norm(dense_arr)
    if norm > 0:
        dense_arr = dense_arr / norm
    return dense_arr[0].tolist()

# ─────────────────────────────────────────────────────────────────────────────
# LLM helper
# ─────────────────────────────────────────────────────────────────────────────
def qwen_chat(prompt: str, max_tokens: int = 800) -> str:
    """Call the Qwen3-8B OpenAI-compatible endpoint."""
    candidate_models = [
        QWEN_MODEL,
        "qwen/Qwen3-8B",
        "Qwen/Qwen3-8B",
        "qwen3-8b",
        "Qwen3-8B",
    ]
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }
    last_error = None
    for model_name in candidate_models:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": False,
        }
        try:
            r = requests.post(QWEN_API_URL, json=payload, headers=headers, timeout=120)
            if r.status_code == 404 and "model" in r.text:
                last_error = f"model not found ({model_name})"
                continue
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Qwen API error: {e}") from e

    raise RuntimeError(f"Qwen model not found on any candidate. Last error: {last_error}")


# ─────────────────────────────────────────────────────────────────────────────
# Profile → search query
# ─────────────────────────────────────────────────────────────────────────────
def build_profile_query(profile: dict) -> str:
    """Convert the user's profile dict to a rich natural-language search query."""
    parts = []
    if profile.get("current_role"):
        parts.append(f"Role: {profile['current_role']}")
    if profile.get("years_exp"):
        parts.append(f"{profile['years_exp']} years of experience")
    if profile.get("skills"):
        parts.append(f"Skills: {profile['skills']}")
    if profile.get("preferred_roles"):
        parts.append(f"Target roles: {', '.join(profile['preferred_roles'])}")
    if profile.get("industries"):
        parts.append(f"Preferred industries: {', '.join(profile['industries'])}")
    if profile.get("remote_pref") and profile["remote_pref"] != "Any":
        parts.append(f"Work preference: {profile['remote_pref']}")
    if profile.get("job_type") and profile["job_type"] != "Both":
        parts.append(f"Job type: {profile['job_type']}")
    return ". ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant semantic search
# ─────────────────────────────────────────────────────────────────────────────
def search_jobs(query_text: str, top_k: int = TOP_K_SEARCH) -> list:
    """Encode the query with the sklearn model and search Qdrant."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Embedding model not found. "
            "Please run `python tools/create_job_embeddings.py` first."
        )
        return []
    try:
        query_vector = encode_query(query_text)
        results = client_qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
        return results
    except Exception as e:
        st.error(f"Qdrant search error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# LLM ranking + explanation
# ─────────────────────────────────────────────────────────────────────────────
def rank_and_explain(profile: dict, job_results: list) -> str:
    """Ask Qwen3 to rank the top jobs and explain why each fits the candidate."""
    jobs_text = ""
    for i, result in enumerate(job_results[:TOP_K_DISPLAY], 1):
        meta = result.payload if hasattr(result, "payload") else {}
        sal_min = meta.get("salary_min", 0)
        sal_max = meta.get("salary_max", 0)
        jobs_text += (
            f"\nJob {i}: {meta.get('title', 'N/A')} at {meta.get('company', 'N/A')}\n"
            f"  Location: {meta.get('location', 'N/A')} | Work: {meta.get('remote', 'N/A')}\n"
            f"  Level: {meta.get('level', 'N/A')} | Exp required: {meta.get('years_exp', 'N/A')}\n"
            f"  Skills: {meta.get('skills', 'N/A')}\n"
            f"  Industry: {meta.get('industry', 'N/A')} | Type: {meta.get('job_type', 'N/A')}\n"
            f"  Salary: ${sal_min:,} – ${sal_max:,} / year\n"
            f"  Description: {meta.get('description', 'N/A')}\n"
        )

    salary_line = (
        f"Minimum salary expectation: ${profile.get('min_salary', 0):,}/year"
        if profile.get("min_salary") else ""
    )

    prompt = f"""You are a Career Discovery Assistant helping a professional find the ideal job.

Candidate Profile:
- Name: {profile.get('name', 'the candidate')}
- Current Role: {profile.get('current_role', 'Not specified')}
- Years of Experience: {profile.get('years_exp', 'Not specified')}
- Skills: {profile.get('skills', 'Not specified')}
- Target Roles: {', '.join(profile.get('preferred_roles', [])) or 'Not specified'}
- Work Preference: {profile.get('remote_pref', 'Any')}
- Industries of Interest: {', '.join(profile.get('industries', [])) or 'Not specified'}
- Job Type: {profile.get('job_type', 'Full-time')}
{salary_line}

Below are {min(len(job_results), TOP_K_DISPLAY)} semantically matched job listings. For each one, provide a structured assessment using exactly this format:

---
**🏢 [Job Number]. [Exact Job Title] — [Company Name]**
📊 **Match Score:** [X]/10
✅ **Why it fits:** [2 sentences specifically linking the candidate's skills and experience to this role's requirements]
⚠️ **One thing to note:** [1 sentence about a potential gap, stretch requirement, or "Excellent overall match" if none]
---

Be honest, specific, and concise. Reference actual skills from the candidate's profile.

Jobs to evaluate:
{jobs_text}"""

    return qwen_chat(prompt, max_tokens=900)


# ─────────────────────────────────────────────────────────────────────────────
# Job card renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_job_card(index: int, result) -> None:
    """Render a compact, expandable job card for a Qdrant result."""
    meta = result.payload if hasattr(result, "payload") else {}
    title = meta.get("title", "Unknown Role")
    company = meta.get("company", "Unknown Company")
    location = meta.get("location", "N/A")
    remote = meta.get("remote", "N/A")
    level = meta.get("level", "N/A")
    years_exp = meta.get("years_exp", "N/A")
    skills = meta.get("skills", "")
    description = meta.get("description", "N/A")
    industry = meta.get("industry", "N/A")
    job_type = meta.get("job_type", "N/A")
    sal_min = meta.get("salary_min", 0)
    sal_max = meta.get("salary_max", 0)
    score = getattr(result, "score", None)

    score_label = f"  ·  Score: {score:.2f}" if score is not None else ""
    header = f"{index}. **{title}** — {company} | {location}{score_label}"

    with st.expander(header, expanded=(index <= 3)):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Level", level)
        col2.metric("Experience", years_exp)
        col3.metric("Work Policy", remote)
        col4.metric("Salary", f"${sal_min // 1000}k–${sal_max // 1000}k")

        st.markdown(f"**Industry:** {industry}   |   **Type:** {job_type}")

        if skills:
            skill_tags = "  ".join([f"`{s.strip()}`" for s in skills.split(",")])
            st.markdown(f"**Skills:** {skill_tags}")

        st.markdown(f"**About the role:** {description}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Streamlit app
# ─────────────────────────────────────────────────────────────────────────────
def show() -> None:
    st.set_page_config(
        page_title="Career Discovery Assistant",
        page_icon="🎯",
        layout="wide",
    )

    # ── Custom CSS for a polished look ──────────────────────────────────────
    st.markdown(
        """
        <style>
        .main-title { font-size: 2.4rem; font-weight: 700; color: #1a1a2e; }
        .subtitle   { font-size: 1.1rem; color: #666; margin-bottom: 1.5rem; }
        .kpi-box    { background: #f0f4ff; border-radius: 10px; padding: 1rem;
                      text-align: center; border: 1px solid #d0dbf5; }
        .section-header { font-size: 1.3rem; font-weight: 600; margin-top: 1.5rem; }
        div[data-testid="stExpander"] { border: 1px solid #e0e7ff; border-radius: 8px;
                                        margin-bottom: 0.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar – Profile Form ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎯 Career Discovery Assistant")
        st.markdown("*Team Gen-AI Uncharted*")
        st.markdown("---")
        st.markdown("### 👤 Your Profile")

        name = st.text_input("Full Name", placeholder="e.g. Azy Sharma")
        current_role = st.text_input(
            "Current Role", placeholder="e.g. Senior Software Engineer"
        )
        years_exp = st.slider("Years of Experience", 0, 30, 5)
        skills = st.text_area(
            "Skills (comma-separated)",
            placeholder="e.g. Python, AWS, Kubernetes, React, System Design",
            height=100,
        )

        st.markdown("---")
        preferred_roles = st.multiselect(
            "🎯 Target Roles",
            options=[
                "Software Engineer",
                "Senior Software Engineer",
                "Staff Engineer",
                "Principal Engineer",
                "Engineering Lead",
                "Engineering Manager",
                "Technical Architect",
                "Solutions Architect",
                "Enterprise Architect",
                "Data Engineer",
                "Data Scientist",
                "ML Engineer",
                "ML Research Engineer",
                "DevOps / SRE",
                "Security Engineer",
                "Frontend Engineer",
                "Full Stack Engineer",
                "iOS Engineer",
                "Android Engineer",
                "Platform Engineer",
                "Product Manager",
            ],
            default=["Senior Software Engineer"],
        )

        remote_pref = st.selectbox(
            "🌍 Work Preference", ["Any", "Remote", "Hybrid", "On-site"]
        )

        industries = st.multiselect(
            "🏭 Industries of Interest",
            options=[
                "AI / ML",
                "Fintech",
                "Cloud / Infrastructure",
                "E-commerce",
                "Consumer Tech",
                "Enterprise Software",
                "Healthcare Tech",
                "Cybersecurity",
                "Data Analytics",
                "Automotive Tech",
            ],
        )

        st.markdown("---")
        min_salary = st.number_input(
            "💰 Min. Salary (USD/year)", min_value=0, value=150000, step=10000
        )
        job_type = st.selectbox("📋 Job Type", ["Full-time", "Contract", "Both"])

        st.markdown("---")
        search_clicked = st.button(
            "🔍 Discover Matching Jobs",
            use_container_width=True,
            type="primary",
        )

    # ── Main Content ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="main-title">🎯 Career Discovery Assistant</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">AI-powered job matching · Qwen3-8B + Qdrant Semantic Search · Team GenAI Uncharted</div>',
        unsafe_allow_html=True,
    )

    if not search_clicked:
        # ── Welcome / Onboarding screen ──────────────────────────────────────
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div class="kpi-box">📋 <b>Step 1</b><br>Fill in your profile in the sidebar</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                '<div class="kpi-box">🔍 <b>Step 2</b><br>Click <i>Discover Matching Jobs</i></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                '<div class="kpi-box">✨ <b>Step 3</b><br>Get AI-ranked matches with personalised explanations</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### Why Career Discovery Assistant?")

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        kpi_col1.metric("Time-to-Apply Reduction", "Hours → Minutes", "↓ 90%")
        kpi_col2.metric("Job Relevance Accuracy", "> 85%", "AI-ranked")
        kpi_col3.metric("Applications / Week", "5× more", "↑ Throughput")
        kpi_col4.metric("Manual Effort", "↓ Drastically", "Automated")

        st.markdown("---")
        st.info(
            "**How it works:**  Your profile is converted to a semantic query and matched "
            "against a curated job database using Qdrant vector search. "
            "Qwen3-8B then ranks the top matches and explains exactly why each role fits you — "
            "referencing your actual skills and career goals."
        )
        return

    # ── Discovery Flow ───────────────────────────────────────────────────────
    profile = {
        "name": name,
        "current_role": current_role,
        "years_exp": years_exp,
        "skills": skills,
        "preferred_roles": preferred_roles,
        "remote_pref": remote_pref,
        "industries": industries,
        "min_salary": min_salary,
        "job_type": job_type,
    }

    query = build_profile_query(profile)

    # Validate basic input
    if not skills and not current_role and not preferred_roles:
        st.warning("Please fill in at least your Current Role, Skills, or Target Roles before searching.")
        return

    st.markdown("---")

    # ── Step 1: Semantic search ──────────────────────────────────────────────
    with st.spinner("🔍 Performing semantic job search in Qdrant..."):
        job_results = search_jobs(query, top_k=TOP_K_SEARCH)

    if not job_results:
        st.error(
            "No jobs found. Make sure the Qdrant 'jobs' collection is populated "
            "by running `tools/create_job_embeddings.py` first."
        )
        return

    # ── Step 2: LLM ranking ─────────────────────────────────────────────────
    with st.spinner(f"🤖 Qwen3-8B is ranking and explaining your top {TOP_K_DISPLAY} matches..."):
        try:
            llm_analysis = rank_and_explain(profile, job_results)
        except RuntimeError as e:
            st.error(f"LLM error: {e}")
            llm_analysis = None

    # ── Results header ───────────────────────────────────────────────────────
    greeting = f"### Results for {name}" if name else "### Your Matches"
    st.markdown(greeting)
    st.caption(
        f"Search query: *\"{query[:120]}{'…' if len(query) > 120 else ''}\"*"
    )
    st.markdown(f"Found **{len(job_results)}** semantic matches. "
                f"Top **{TOP_K_DISPLAY}** ranked and explained by Qwen3-8B below.")

    # ── AI Analysis ──────────────────────────────────────────────────────────
    if llm_analysis:
        st.markdown("---")
        st.markdown("## 🤖 AI-Powered Match Analysis")
        st.markdown(llm_analysis)

    # ── Full result cards ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"## 📋 All {len(job_results)} Semantic Matches")
    st.caption("Ranked by semantic similarity to your profile. Top 3 expanded by default.")

    for i, result in enumerate(job_results, 1):
        render_job_card(i, result)

    # ── Profile summary ──────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📄 Your Search Profile (used as query)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {name or '—'}")
            st.markdown(f"**Current Role:** {current_role or '—'}")
            st.markdown(f"**Years of Experience:** {years_exp}")
            st.markdown(f"**Skills:** {skills or '—'}")
        with col2:
            st.markdown(f"**Target Roles:** {', '.join(preferred_roles) or '—'}")
            st.markdown(f"**Work Preference:** {remote_pref}")
            st.markdown(f"**Industries:** {', '.join(industries) or '—'}")
            st.markdown(f"**Min. Salary:** ${min_salary:,}")
            st.markdown(f"**Job Type:** {job_type}")
        st.code(query, language=None)


if __name__ == "__main__":
    show()
