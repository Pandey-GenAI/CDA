"""
CDA – Career Discovery Assistant
Populates the Qdrant 'jobs' collection with 30 realistic job postings.
Uses sklearn TF-IDF + SVD (Latent Semantic Analysis) for embeddings —
no internet connection or model downloads required.

Usage:
    pip install qdrant-client scikit-learn numpy
    python create_job_embeddings.py
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "jobs"

# The fitted vectorizer is saved here so the app can reuse it for query encoding.
# Path is relative to this script's location → saves into cda/app/embedding_model.pkl
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "app", "embedding_model.pkl")

client = QdrantClient("http://localhost:6333")

COLLECTION_NAME = "jobs"

# ────────────────────────────────────────────────────────────
# 30 diverse, realistic job postings
# ────────────────────────────────────────────────────────────
jobs = [
    {
        "id": 0,
        "title": "Senior Software Engineer – Backend",
        "company": "Stripe",
        "location": "San Francisco, CA",
        "remote": "Hybrid",
        "level": "Senior",
        "skills": "Python, Go, Distributed Systems, REST APIs, PostgreSQL, Redis",
        "description": (
            "Build reliable, high-throughput backend services that power global payments for millions of businesses. "
            "You will design APIs, own microservices end-to-end, and collaborate with product and infrastructure teams "
            "to ship financial products at scale."
        ),
        "salary_min": 180000,
        "salary_max": 250000,
        "industry": "Fintech",
        "job_type": "Full-time",
        "years_exp": "5+",
    },
    {
        "id": 1,
        "title": "Staff Software Engineer – Infrastructure",
        "company": "Google",
        "location": "Mountain View, CA",
        "remote": "Hybrid",
        "level": "Staff",
        "skills": "C++, Go, Kubernetes, Borg, Large-Scale Systems, Performance Engineering",
        "description": (
            "Drive the architecture of Google-scale infrastructure supporting billions of users. "
            "You will lead cross-team technical initiatives, mentor senior engineers, and define engineering "
            "standards across the org."
        ),
        "salary_min": 220000,
        "salary_max": 320000,
        "industry": "Cloud / Infrastructure",
        "job_type": "Full-time",
        "years_exp": "8+",
    },
    {
        "id": 2,
        "title": "Engineering Manager – Product Engineering",
        "company": "Meta",
        "location": "Menlo Park, CA",
        "remote": "Hybrid",
        "level": "Manager",
        "skills": "People Management, Agile, React, Python, Team Building, Roadmap Planning",
        "description": (
            "Lead a team of 8–12 engineers building consumer-facing social features used by billions of people. "
            "You will be responsible for hiring, career development, delivery, and collaboration with PMs and designers."
        ),
        "salary_min": 200000,
        "salary_max": 290000,
        "industry": "Consumer Tech",
        "job_type": "Full-time",
        "years_exp": "7+",
    },
    {
        "id": 3,
        "title": "Principal Software Engineer – Search",
        "company": "Amazon",
        "location": "Seattle, WA",
        "remote": "Hybrid",
        "level": "Principal",
        "skills": "Java, Distributed Systems, Search Algorithms, Machine Learning, AWS, System Design",
        "description": (
            "Architect the next generation of Amazon's product search and recommendation systems. "
            "Define the technical vision, mentor staff engineers, and partner with science teams to integrate ML models into search."
        ),
        "salary_min": 230000,
        "salary_max": 350000,
        "industry": "E-commerce",
        "job_type": "Full-time",
        "years_exp": "10+",
    },
    {
        "id": 4,
        "title": "Senior Frontend Engineer",
        "company": "Figma",
        "location": "San Francisco, CA",
        "remote": "Hybrid",
        "level": "Senior",
        "skills": "TypeScript, React, WebGL, Canvas API, Performance Optimization, CSS",
        "description": (
            "Craft pixel-perfect, high-performance UI experiences for Figma's collaborative design tool used by millions of designers. "
            "Work on rendering, real-time collaboration features, and new creative tools."
        ),
        "salary_min": 170000,
        "salary_max": 240000,
        "industry": "Enterprise Software",
        "job_type": "Full-time",
        "years_exp": "5+",
    },
    {
        "id": 5,
        "title": "Solutions Architect – Azure",
        "company": "Microsoft",
        "location": "Redmond, WA",
        "remote": "Hybrid",
        "level": "Senior",
        "skills": "Azure, Cloud Architecture, Terraform, Kubernetes, Enterprise Sales, Solution Design",
        "description": (
            "Partner with Fortune 500 customers to design cloud migration and modernization strategies on Azure. "
            "You will lead technical workshops, create reference architectures, and work closely with sales and product teams."
        ),
        "salary_min": 160000,
        "salary_max": 230000,
        "industry": "Cloud / Infrastructure",
        "job_type": "Full-time",
        "years_exp": "6+",
    },
    {
        "id": 6,
        "title": "Senior ML Engineer",
        "company": "OpenAI",
        "location": "San Francisco, CA",
        "remote": "Hybrid",
        "level": "Senior",
        "skills": "Python, PyTorch, Transformers, CUDA, Distributed Training, LLM Fine-tuning",
        "description": (
            "Train and optimize large language models pushing the frontier of AI capabilities. "
            "You will work on pre-training pipelines, RLHF, evaluation frameworks, and productionizing models at scale."
        ),
        "salary_min": 250000,
        "salary_max": 400000,
        "industry": "AI / ML",
        "job_type": "Full-time",
        "years_exp": "5+",
    },
    {
        "id": 7,
        "title": "Senior Data Engineer",
        "company": "Databricks",
        "location": "San Francisco, CA",
        "remote": "Remote",
        "level": "Senior",
        "skills": "Apache Spark, Delta Lake, Python, SQL, Kafka, Airflow, Data Modeling",
        "description": (
            "Build the data infrastructure that powers analytics and ML for Databricks' own data platform. "
            "Design scalable ETL pipelines, optimize Spark workloads, and champion data quality standards."
        ),
        "salary_min": 160000,
        "salary_max": 220000,
        "industry": "Data Analytics",
        "job_type": "Full-time",
        "years_exp": "5+",
    },
    {
        "id": 8,
        "title": "Site Reliability Engineer",
        "company": "Netflix",
        "location": "Los Gatos, CA",
        "remote": "Hybrid",
        "level": "Senior",
        "skills": "AWS, Kubernetes, Terraform, Python, Go, Observability, Incident Management",
        "description": (
            "Ensure Netflix streams reliably for 250M+ subscribers worldwide. "
            "Automate operations, build self-healing systems, define SLOs, and run chaos engineering experiments."
        ),
        "salary_min": 190000,
        "salary_max": 280000,
        "industry": "Consumer Tech",
        "job_type": "Full-time",
        "years_exp": "6+",
    },
    {
        "id": 9,
        "title": "Technical Architect – Digital Banking",
        "company": "JP Morgan Chase",
        "location": "New York, NY",
        "remote": "Hybrid",
        "level": "Architect",
        "skills": "Java, Spring Boot, Microservices, Cloud Native, API Design, Enterprise Architecture",
        "description": (
            "Lead the architectural transformation of JP Morgan's digital banking platform serving 60M+ customers. "
            "Define target state architecture, review technical designs, and guide engineering teams through cloud migration."
        ),
        "salary_min": 200000,
        "salary_max": 300000,
        "industry": "Fintech",
        "job_type": "Full-time",
        "years_exp": "10+",
    },
    {
        "id": 10,
        "title": "Senior Full Stack Engineer",
        "company": "Notion",
        "location": "San Francisco, CA",
        "remote": "Remote",
        "level": "Senior",
        "skills": "TypeScript, React, Node.js, PostgreSQL, GraphQL, Real-time Collaboration",
        "description": (
            "Build the productivity platform used by millions of individuals and teams worldwide. "
            "Own features end-to-end from backend to frontend, optimize performance, and help define Notion's technical roadmap."
        ),
        "salary_min": 165000,
        "salary_max": 230000,
        "industry": "Enterprise Software",
        "job_type": "Full-time",
        "years_exp": "4+",
    },
    {
        "id": 11,
        "title": "ML Platform Engineer",
        "company": "Hugging Face",
        "location": "New York, NY",
        "remote": "Remote",
        "level": "Senior",
        "skills": "Python, PyTorch, Docker, Kubernetes, FastAPI, Model Serving, MLOps",
        "description": (
            "Build and scale the infrastructure that serves hundreds of thousands of open-source AI models. "
            "Work on inference optimization, model hosting, and developer tooling for the ML community."
        ),
        "salary_min": 155000,
        "salary_max": 220000,
        "industry": "AI / ML",
        "job_type": "Full-time",
        "years_exp": "4+",
    },
    {
        "id": 12,
        "title": "Senior Data Scientist – Personalization",
        "company": "Airbnb",
        "location": "San Francisco, CA",
        "remote": "Hybrid",
        "level": "Senior",
        "skills": "Python, SQL, Machine Learning, A/B Testing, Recommendation Systems, Statistics",
        "description": (
            "Drive the algorithms that match 100M+ guests to the perfect Airbnb listing. "
            "Design and ship ML models for search ranking and personalization, run large-scale A/B experiments, "
            "and partner with engineering to productionize models."
        ),
        "salary_min": 175000,
        "salary_max": 250000,
        "industry": "Consumer Tech",
        "job_type": "Full-time",
        "years_exp": "5+",
    },
    {
        "id": 13,
        "title": "Senior Security Engineer",
        "company": "CrowdStrike",
        "location": "Austin, TX",
        "remote": "Remote",
        "level": "Senior",
        "skills": "Cybersecurity, Threat Detection, Python, EDR, Cloud Security, SIEM, Incident Response",
        "description": (
            "Protect CrowdStrike's Falcon platform and build detection capabilities to stop adversaries. "
            "Conduct threat modeling, red-team exercises, and develop automated security tooling."
        ),
        "salary_min": 160000,
        "salary_max": 230000,
        "industry": "Cybersecurity",
        "job_type": "Full-time",
        "years_exp": "5+",
    },
    {
        "id": 14,
        "title": "Senior iOS Engineer",
        "company": "Apple",
        "location": "Cupertino, CA",
        "remote": "On-site",
        "level": "Senior",
        "skills": "Swift, Objective-C, UIKit, SwiftUI, Core Data, Xcode, Performance Optimization",
        "description": (
            "Create world-class iOS experiences used by over a billion Apple device users. "
            "Own critical system frameworks, collaborate with hardware teams, and set the standard for mobile software quality."
        ),
        "salary_min": 185000,
        "salary_max": 270000,
        "industry": "Consumer Tech",
        "job_type": "Full-time",
        "years_exp": "5+",
    },
    {
        "id": 15,
        "title": "Senior Android Engineer",
        "company": "Spotify",
        "location": "New York, NY",
        "remote": "Hybrid",
        "level": "Senior",
        "skills": "Kotlin, Android SDK, Jetpack Compose, MVVM, Media APIs, CI/CD",
        "description": (
            "Build the Spotify Android app used by 200M+ active users worldwide. "
            "Focus on audio streaming reliability, offline playback, and crafting engaging UI with Jetpack Compose."
        ),
        "salary_min": 160000,
        "salary_max": 230000,
        "industry": "Consumer Tech",
        "job_type": "Full-time",
        "years_exp": "4+",
    },
    {
        "id": 16,
        "title": "Platform Engineer – Edge Networking",
        "company": "Cloudflare",
        "location": "San Francisco, CA",
        "remote": "Remote",
        "level": "Senior",
        "skills": "Rust, Go, Linux Networking, eBPF, DNS, TLS, Distributed Systems",
        "description": (
            "Scale Cloudflare's global edge network that handles 50M+ HTTP requests per second. "
            "Implement low-latency networking solutions, optimize critical paths, and build developer-facing platform primitives."
        ),
        "salary_min": 175000,
        "salary_max": 250000,
        "industry": "Cloud / Infrastructure",
        "job_type": "Full-time",
        "years_exp": "5+",
    },
    {
        "id": 17,
        "title": "Senior Python Engineer – Data Platform",
        "company": "Palantir",
        "location": "New York, NY",
        "remote": "Hybrid",
        "level": "Senior",
        "skills": "Python, Spark, Data Pipelines, Ontology, TypeScript, REST APIs",
        "description": (
            "Build Palantir's Foundry platform that transforms complex data into actionable decisions for government "
            "and enterprise clients. Design data integration workflows, build APIs, and improve platform scalability."
        ),
        "salary_min": 170000,
        "salary_max": 240000,
        "industry": "Enterprise Software",
        "job_type": "Full-time",
        "years_exp": "4+",
    },
    {
        "id": 18,
        "title": "Engineering Manager – Autopilot AI",
        "company": "Tesla",
        "location": "Palo Alto, CA",
        "remote": "On-site",
        "level": "Manager",
        "skills": "People Management, Computer Vision, Deep Learning, C++, Python, Real-time Systems",
        "description": (
            "Lead a team of 10+ engineers building Tesla's self-driving AI stack. "
            "Drive roadmap execution for perception and planning systems, manage performance, and collaborate with "
            "research scientists on model development."
        ),
        "salary_min": 210000,
        "salary_max": 320000,
        "industry": "Automotive Tech",
        "job_type": "Full-time",
        "years_exp": "8+",
    },
    {
        "id": 19,
        "title": "Senior API Engineer",
        "company": "Twilio",
        "location": "San Francisco, CA",
        "remote": "Remote",
        "level": "Senior",
        "skills": "Java, Python, REST APIs, Webhooks, Message Queues, PostgreSQL, Developer Experience",
        "description": (
            "Design and build the APIs that power communication for 300,000+ businesses worldwide. "
            "Improve API reliability, developer onboarding experience, and build SDKs for new communication channels."
        ),
        "salary_min": 155000,
        "salary_max": 220000,
        "industry": "Enterprise Software",
        "job_type": "Full-time",
        "years_exp": "4+",
    },
    {
        "id": 20,
        "title": "Staff Data Engineer – Analytics Platform",
        "company": "Snowflake",
        "location": "San Mateo, CA",
        "remote": "Hybrid",
        "level": "Staff",
        "skills": "Snowflake, dbt, Python, SQL, Data Modeling, Airflow, Lakehouse Architecture",
        "description": (
            "Build the internal data platform that powers analytics for Snowflake's business and engineering teams. "
            "Define data architecture standards, mentor data engineers, and drive adoption of lakehouse patterns."
        ),
        "salary_min": 185000,
        "salary_max": 260000,
        "industry": "Data Analytics",
        "job_type": "Full-time",
        "years_exp": "7+",
    },
    {
        "id": 21,
        "title": "Enterprise Architect – AI Transformation",
        "company": "IBM",
        "location": "Armonk, NY",
        "remote": "Hybrid",
        "level": "Architect",
        "skills": "Enterprise Architecture, AI/ML, Cloud, TOGAF, Stakeholder Management, Solution Design",
        "description": (
            "Lead AI-driven digital transformation engagements for Fortune 100 clients. "
            "Define enterprise AI strategies, design reference architectures, and guide organizations through "
            "responsible AI adoption."
        ),
        "salary_min": 180000,
        "salary_max": 270000,
        "industry": "Enterprise Software",
        "job_type": "Full-time",
        "years_exp": "12+",
    },
    {
        "id": 22,
        "title": "Senior Software Engineer – Payments",
        "company": "PayPal",
        "location": "San Jose, CA",
        "remote": "Hybrid",
        "level": "Senior",
        "skills": "Java, Spring, Microservices, Payment Processing, Kafka, MySQL, PCI Compliance",
        "description": (
            "Build the core payment processing system handling $1T+ in annual transaction volume. "
            "Improve reliability, latency, and fraud detection capabilities across PayPal's global payments platform."
        ),
        "salary_min": 160000,
        "salary_max": 230000,
        "industry": "Fintech",
        "job_type": "Full-time",
        "years_exp": "5+",
    },
    {
        "id": 23,
        "title": "Senior Data Platform Engineer",
        "company": "Uber",
        "location": "San Francisco, CA",
        "remote": "Hybrid",
        "level": "Senior",
        "skills": "Flink, Kafka, Python, Scala, Hive, Presto, Real-time Streaming, Data Infra",
        "description": (
            "Build and scale Uber's real-time data infrastructure processing petabytes of trip, pricing, "
            "and marketplace data daily. Enable analytics, ML, and operational insights for 30,000+ engineers."
        ),
        "salary_min": 175000,
        "salary_max": 250000,
        "industry": "Consumer Tech",
        "job_type": "Full-time",
        "years_exp": "5+",
    },
    {
        "id": 24,
        "title": "Senior Cloud Solutions Engineer",
        "company": "Amazon Web Services",
        "location": "Seattle, WA",
        "remote": "Hybrid",
        "level": "Senior",
        "skills": "AWS, CDK, Terraform, Python, Solution Architecture, Technical Pre-sales",
        "description": (
            "Help AWS's largest enterprise customers architect secure, cost-effective cloud solutions. "
            "Run technical proof-of-concepts, deliver workshops, and serve as a technical advisor across multiple AWS services."
        ),
        "salary_min": 165000,
        "salary_max": 240000,
        "industry": "Cloud / Infrastructure",
        "job_type": "Full-time",
        "years_exp": "6+",
    },
    {
        "id": 25,
        "title": "Senior Go Engineer – Container Runtime",
        "company": "Docker",
        "location": "Palo Alto, CA",
        "remote": "Remote",
        "level": "Senior",
        "skills": "Go, Docker, containerd, Linux Namespaces, Networking, Open Source",
        "description": (
            "Work on Docker Desktop and the container ecosystem used by 20M+ developers worldwide. "
            "Contribute to open-source runtimes, improve developer workflows, and build platform integrations."
        ),
        "salary_min": 155000,
        "salary_max": 220000,
        "industry": "Cloud / Infrastructure",
        "job_type": "Full-time",
        "years_exp": "4+",
    },
    {
        "id": 26,
        "title": "Engineering Lead – Trading Platform",
        "company": "Robinhood",
        "location": "Menlo Park, CA",
        "remote": "Hybrid",
        "level": "Lead",
        "skills": "Python, Go, Low-latency Systems, Financial Markets, Trading APIs, PostgreSQL",
        "description": (
            "Lead the engineering effort for Robinhood's options and equities trading platform. "
            "Improve execution speed, build regulatory compliance features, and guide a team of 6 engineers."
        ),
        "salary_min": 190000,
        "salary_max": 270000,
        "industry": "Fintech",
        "job_type": "Full-time",
        "years_exp": "7+",
    },
    {
        "id": 27,
        "title": "Senior React Developer – Commerce",
        "company": "Shopify",
        "location": "Ottawa, Canada",
        "remote": "Remote",
        "level": "Senior",
        "skills": "React, TypeScript, GraphQL, Ruby on Rails, Polaris Design System, E-commerce",
        "description": (
            "Build the merchant-facing checkout and storefront experiences for 2M+ Shopify merchants globally. "
            "Optimize conversion flows, contribute to the Polaris design system, and ship A/B experiments."
        ),
        "salary_min": 140000,
        "salary_max": 200000,
        "industry": "E-commerce",
        "job_type": "Full-time",
        "years_exp": "4+",
    },
    {
        "id": 28,
        "title": "Healthcare AI Engineer",
        "company": "Oscar Health",
        "location": "New York, NY",
        "remote": "Hybrid",
        "level": "Senior",
        "skills": "Python, NLP, FHIR, Machine Learning, Healthcare Data, PostgreSQL, LLMs",
        "description": (
            "Apply AI to improve healthcare outcomes for Oscar's 1M+ members. "
            "Build NLP models for clinical note analysis, automate prior authorization workflows, "
            "and develop predictive models for member health risk."
        ),
        "salary_min": 160000,
        "salary_max": 230000,
        "industry": "Healthcare Tech",
        "job_type": "Full-time",
        "years_exp": "4+",
    },
    {
        "id": 29,
        "title": "Staff ML Research Engineer",
        "company": "DeepMind",
        "location": "London, UK",
        "remote": "Hybrid",
        "level": "Staff",
        "skills": "Python, JAX, PyTorch, Reinforcement Learning, Research Engineering, Mathematical Modeling",
        "description": (
            "Bridge cutting-edge research and production systems at one of the world's leading AI labs. "
            "Implement novel RL algorithms, build training infrastructure, and collaborate with researchers "
            "to move discoveries from paper to production."
        ),
        "salary_min": 170000,
        "salary_max": 280000,
        "industry": "AI / ML",
        "job_type": "Full-time",
        "years_exp": "6+",
    },
]

# ────────────────────────────────────────────────────────────
# Build natural-language documents for semantic embedding
# ────────────────────────────────────────────────────────────
documents = []

for job in jobs:
    doc = (
        f"{job['title']} at {job['company']} located in {job['location']}. "
        f"Work policy: {job['remote']}. Level: {job['level']}. "
        f"Requires {job['years_exp']} years of experience. "
        f"Key skills: {job['skills']}. "
        f"Industry: {job['industry']}. Job type: {job['job_type']}. "
        f"{job['description']} "
        f"Salary range: ${job['salary_min']:,} to ${job['salary_max']:,} USD per year."
    )
    documents.append(doc)

# ────────────────────────────────────────────────────────────
# Fit TF-IDF vectorizer — fully offline, fixed output dimension
# ────────────────────────────────────────────────────────────
print("Fitting TF-IDF vectorizer (no internet needed)...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),     # unigrams + bigrams for better phrase matching
    min_df=1,
    sublinear_tf=True,      # log-scaling of term frequency
    # No max_features — use full vocabulary so dimension is stable
)
tfidf_matrix = vectorizer.fit_transform(documents)   # shape: (30, vocab_size)
dense_matrix = tfidf_matrix.toarray()                # convert sparse → dense
dense_matrix = normalize(dense_matrix)               # L2-normalise for cosine similarity

# Derive vector size from the actual fitted vocabulary
VECTOR_SIZE = dense_matrix.shape[1]
print(f"Vectors shape: {dense_matrix.shape}  →  VECTOR_SIZE = {VECTOR_SIZE}")

# Save vectorizer AND the actual vector size so the app stays in sync
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"vectorizer": vectorizer, "vector_size": VECTOR_SIZE}, f)
print(f"Embedding model saved to: {MODEL_PATH}")

# ────────────────────────────────────────────────────────────
# Create Qdrant collection and upload vectors
# ────────────────────────────────────────────────────────────
print(f"\nCreating Qdrant collection '{COLLECTION_NAME}'...")

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=job["id"],
        vector=dense_matrix[i].tolist(),
        payload=job,
    )
    for i, job in enumerate(jobs)
]

client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Uploaded {len(points)} job postings.")

# ────────────────────────────────────────────────────────────
# Verify with a test search
# ────────────────────────────────────────────────────────────
print("\nVerifying with a test search...")
test_query = "senior backend engineer Python distributed systems fintech"
test_tfidf = vectorizer.transform([test_query])
test_vector = normalize(test_tfidf.toarray())[0].tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=test_vector,
    limit=3,
    with_payload=True,
).points

for r in results:
    print(f"  → {r.payload['title']} at {r.payload['company']} (score: {r.score:.3f})")

print("\n✅ Job collection ready for the Career Discovery Assistant.")
