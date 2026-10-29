# Architecture and API contract

## Main request flow

Browser → Django/DRF authentication and role checks → machine-scoped retrieval → grounded LLM answer → persisted question and source rows → browser citation display.

## API starter routes

- `GET /api/machines/` — authenticated machine list
- `POST /api/machines/` — Administrator only
- `GET /api/manuals/` — authenticated manual list
- `POST /api/manuals/` — Administrator only; multipart PDF upload
- `POST /api/questions/ask/` — Engineer role; JSON `{ "machine": 1, "question": "..." }`
- `GET /api/questions/` — own history for non-supervisors; all history for Supervisor
- `GET /api/supervisor/history/` — Supervisor read-only history

## RAG adapter contract

`retrieve_for_machine(machine_id, question, top_k=4)` returns passages with `manual_id`, `manual_title`, `revision`, `page_number`, `text`, and optional score. It must apply machine filtering at retrieval time and validate each returned manual against the selected machine.

`generate_grounded_answer(question, passages)` returns `(answer_text, insufficient_evidence_bool)`. If no passages or evidence is weak, abstain. Store citations using the returned passage metadata after verifying the manual belongs to the selected machine.

## Role matrix

| Capability | Administrator | Engineer | Supervisor |
|---|---:|---:|---:|
| Register machine | Yes | No | No |
| Upload/manage manuals | Yes | No | No |
| Ask maintenance question | No by default | Yes | No |
| View own question history | Optional | Yes | Read all |
| View all history/unanswered queries | No by default | No | Yes, read-only |
