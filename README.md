# ForgeAssist AI — Phase 1 starter

Django + Django REST Framework starter for the Manufacturing Manual Assistant challenge.

## Important status

This is a scaffold, **not a finished hackathon submission**. Authentication, basic role permissions, data models, page-preserving PDF extraction, starter UI and permission tests are included. The event-provided embedding model, vector store and LLM must be connected in `core/services/rag.py`. The browser form is also a placeholder and needs CSRF-safe API wiring and source rendering.

## Requirements

- Python 3.11+ (use the Python version supported by the event runtime)
- PostgreSQL recommended; SQLite fallback exists for initial smoke testing only
- Event-provided embedding and LLM access

## Windows setup (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`. For PostgreSQL, create a database and user, then set `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, and `DB_PORT`. Do not commit `.env`.

```powershell
python manage.py check
python manage.py makemigrations core
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Open `http://127.0.0.1:8000/admin/`. Create groups named exactly `Administrator`, `Engineer`, and `Supervisor`, then assign users to the appropriate group. Superusers bypass role checks for emergency/admin setup; use normal accounts for demo tests.

## Team integration contract

- Backend owner: models, migrations, groups/permissions, machine/manual APIs.
- RAG owner: `retrieve_for_machine(machine_id, question, top_k)` and `generate_grounded_answer(question, passages)` in `core/services/rag.py`.
- UI owner: Bootstrap screens and CSRF-safe calls to `/api/machines/`, `/api/manuals/`, `/api/questions/ask/`, and `/api/supervisor/history/`.
- Agree on API payloads before splitting work. Integrate to `main` at least every 30–45 minutes; avoid independent duplicate model definitions.

## Critical implementation requirements

1. Vector retrieval must filter by `machine_id` **before** ranking/returning evidence.
2. Preserve the actual 1-based PDF page number in every chunk and source.
3. Build citations from database metadata, not text invented by the LLM.
4. Abstain when evidence is absent or below a validated threshold.
5. Check roles on the server for every protected endpoint; hiding a UI button is not authorization.
6. Validate file size, extension, PDF content and empty/extraction-failure cases before final submission.
7. Finish the vector-store integration test in `core/tests/test_machine_isolation.py`; the current test is a contract placeholder, not proof of isolation.

## Tests

```powershell
python manage.py test core.tests
```

Add tests for: Engineer cannot upload, Engineer cannot register machines, Machine A retrieval never returns Machine B chunks, unsupported questions abstain, citations match actual stored pages, and question history persists. Do not claim the scaffold passes isolation until the real vector adapter is tested.

## Suggested five-hour MVP

1. 0:00–0:30: repo, database, models, API contracts.
2. 0:30–1:30: login, roles, machine/manual upload and PDF extraction.
3. 1:30–3:00: embeddings, machine-filtered retrieval, grounded answer and citations.
4. 3:00–4:00: UI, history and supervisor read-only view.
5. 4:00–5:00: automated tests, demo rehearsal, README and frozen commit.

## Suggested repository integration

Use one shared repository with branches `backend`, `rag`, and `frontend` (or one shared branch with clearly owned files). Commit small working increments. Do not allow separate branches to redefine shared models independently. Merge and run migrations/tests together frequently.

## Phase 2 reuse

Keep user roles, database configuration, API patterns, file handling and Bootstrap layout modular. Phase 2 will add product registration, image inspection records, supplied model inference, batch jobs (Celery/Redis per brief), and supervisor review. Do not implement Phase 2 now.
