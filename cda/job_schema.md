# Job JSON Schema

Each file in this folder should contain exactly one job object.

## Required fields

- `id` (integer, unique across all files)
- `title` (string)
- `company` (string)
- `location` (string)
- `remote` (string; e.g., `Remote`, `Hybrid`, `On-site`)
- `level` (string; e.g., `Senior`, `Staff`, `Manager`)
- `skills` (string; comma-separated skills)
- `description` (string)
- `salary_min` (integer)
- `salary_max` (integer)
- `industry` (string)
- `job_type` (string; e.g., `Full-time`)
- `years_exp` (string; e.g., `5+`)

## Naming convention

Use:

`NN_company_name.json`

Examples:

- `00_stripe.json`
- `09_jp_morgan_chase.json`

`NN` should typically match the `id` with zero padding.

## Example

```json
{
  "id": 30,
  "title": "Senior Backend Engineer",
  "company": "Example Corp",
  "location": "San Francisco, CA",
  "remote": "Hybrid",
  "level": "Senior",
  "skills": "Python, FastAPI, PostgreSQL, Redis",
  "description": "Build scalable backend services for core products.",
  "salary_min": 170000,
  "salary_max": 240000,
  "industry": "Enterprise Software",
  "job_type": "Full-time",
  "years_exp": "5+"
}
```

## Notes

- Keep `id` unique, otherwise ingestion will fail.
- Keep all required fields present, otherwise ingestion will fail.
- If you add, edit, or remove files, rerun:

```bash
cd cda/tools
python3 create_job_embeddings.py
```
