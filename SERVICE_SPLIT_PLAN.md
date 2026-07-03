# XTROLOGY - Service Split Plan

## Objective

Split the current monolithic API into three independent services while keeping business logic unchanged.

---

## Chat Service (Public)

Endpoints

- /ask_question
- /welcome_message
- /query
- /qna_gpt_mini
- /qna_gemini
- /qna_sl_search

Responsibilities

- Chat
- Retrieval
- GPT/Gemini
- Prompt generation

---

## Chart Service (Public)

Endpoints

- /upload_chart
- /status/{job_id}

Responsibilities

- Chart upload
- Chart processing
- Chart embeddings

---

## Admin Service (Private)

Endpoints

- /upload_kb
- /delete_kb
- /delete_chart
- /qna_sl
- /qna_sl_validation
- /qna_ml_submit
- /qna_generate
- /delete_qna_sl
- /retrieval_test

Responsibilities

- KB management
- SL Training
- Testing
- Maintenance

---

## Shared Modules

- db.py
- vector_db.py
- storage.py
- kb_builder.py
- notifier.py
- prompts.py
- settings.py
- qna_generator.py

---

## Phase 1 Goal

No business logic changes.

No helper function movement.

Only separate API ownership.