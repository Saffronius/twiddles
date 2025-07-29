# Channel Second Brain – MVP Specification (Expanded)

> **Purpose:** Stable reference file for Cursor/LLM agents. Keep updated when implementation changes. Everything here is authoritative for the MVP.

---
## 1. Architecture Overview
**Goal:** Multi‑level channel tree + scratchpad (“Spitball”) routing + living channel descriptions + optional RAG retrieval + minimal memory layering.

**Stack:**
- Backend: Python 3.11, FastAPI, Postgres 16, pgvector, Alembic.
- Frontend: React + TypeScript + Vite.
- LLMs: OpenAI chat model (e.g., `gpt-4o-mini`), embedding model `text-embedding-3-small`.
- Editor: Editor.js (scratchpad + channel description).

---
## 2. Data Model
All UUIDs are v4. Timestamps are `TIMESTAMPTZ`.

### 2.1 Tables
#### `channels`
| Column | Type | Notes |
|--------|------|------|
| id | UUID PK | |
| parent_id | UUID FK nullable | Self‑reference for tree |
| name | TEXT | Unique per parent (enforce in migration) |
| description | TEXT | Current living description (plain text summary) |
| description_json | JSONB | Raw Editor.js doc (source of description) |
| system_prompt | TEXT | Custom prompt text (may be empty) |
| inherit_prompt | BOOLEAN default TRUE | If true, prepend ancestor prompts |
| created_at | TIMESTAMPTZ default now() | |
| embedding_centroid | VECTOR(1536) nullable | Running mean of message embeddings |

Indexes: `(parent_id)`, GIN for `description_json` (optional), btree on `name` filtered by `parent_id`.

#### `messages`
| Column | Type | Notes |
|--------|------|------|
| id | UUID PK | |
| channel_id | UUID FK channels(id) | |
| role | TEXT CHECK in('user','assistant','system') | |
| content | TEXT | Stored as plain text (concatenated blocks for scratchpad) |
| created_at | TIMESTAMPTZ default now() | |
| embedding | VECTOR(1536) nullable | Async populated |
| memory_layer | TEXT CHECK in('recent','compressed') default 'recent' | Retrieval tier |
| scratchpad_origin_id | UUID nullable | Links routed scratchpad |

Index: `(channel_id, created_at)`, IVFFlat on `embedding`.

#### `scratchpad_entries`
| Column | Type | Notes |
|--------|------|------|
| id | UUID PK | |
| content_json | JSONB | Editor.js document |
| content_text | TEXT | Flattened plain text cache |
| state | TEXT CHECK in('staged','routed') default 'staged' | |
| embedding | VECTOR(1536) nullable | |
| created_at | TIMESTAMPTZ default now() | |

#### `routing_logs`
| Column | Type | Notes |
|--------|------|------|
| id | UUID PK | |
| entry_id | UUID FK scratchpad_entries(id) | |
| target_channel_id | UUID nullable | Null if user discarded |
| confidence | REAL nullable | Router score |
| routed_by | TEXT ('user'/'router') | |
| created_at | TIMESTAMPTZ default now() | |

#### `memory_summaries`
| Column | Type | Notes |
|--------|------|------|
| id | UUID PK | |
| channel_id | UUID FK | |
| source_message_ids | UUID[] | Messages compressed into this summary |
| content | TEXT | Summary text |
| created_at | TIMESTAMPTZ default now() | |

#### `context_slices`
| Column | Type | Notes |
|--------|------|------|
| id | UUID PK | |
| channel_id | UUID FK | |
| user_message_id | UUID FK messages(id) | The triggering user message |
| included_message_ids | UUID[] | Raw message ids sent to LLM |
| included_summary_ids | UUID[] | memory_summaries ids sent |
| mode | TEXT CHECK in('plain','rag') | |
| token_estimate | INT nullable | Optional approximate token count |
| created_at | TIMESTAMPTZ default now() | |

#### `knowledge_cards`
| Column | Type | Notes |
|--------|------|------|
| id | UUID PK | |
| channel_id | UUID FK | |
| title | TEXT | |
| body | TEXT | Distilled content |
| source_message_ids | UUID[] | Provenance |
| created_at | TIMESTAMPTZ default now() | |

---
## 3. Routing Algorithm
**Input:** Entire scratchpad text.
1. Compute embedding `E` of scratchpad.
2. For each channel compute score: `score = cosine(E, embedding_centroid)` (skip channels with null centroid → treat score=-∞).
3. If `max_score >= 0.45` → suggest that channel as `best_guess`.
4. Else propose new channel.
5. User can **Accept** (route) or **Create New** (prompt name + optional seed description). On route:
   - Create one `messages` row (`role='user'`, `content` = concatenated block texts, `scratchpad_origin_id` set).
   - Update channel centroid: `new_centroid = old + (E - old)/N` where N = new message count.
   - Mark scratchpad `state='routed'`.
   - Generate description bullet (see §4).

**Undo:** Within 5 minutes remove created message, revert scratchpad to `staged`, recompute centroid (fallback: average all message embeddings), remove bullet with matching RID.

---
## 4. Description Update & Undo
**Bullet format:** `- [RID:<scratchpad_id>] <summary sentence>` appended to description Editor.js doc as a new paragraph/list item.

**Prompt (summary):**
```
Summarize the NEW content in <=25 words, factual, no intro words.
New Content:
"""
{scratchpad_text}
"""
Output only the sentence.
```
**Undo:** Find block containing `[RID:<id>]` and delete it.

---
## 5. Summarization (Memory Layer Compression)
Endpoint manually triggered: `POST /channels/{id}/summarize_old?retain=50`.
1. Select messages for channel ordered oldest→newest excluding newest 50 recent.
2. Group sequential messages into chunks ~2000–4000 tokens.
3. Prompt per chunk:
```
Summarize these chat messages preserving key facts, decisions, open questions. <=300 tokens.
MESSAGES:
{concatenated}
```
4. Store each result as `memory_summaries` with `source_message_ids` list. Mark those source messages `memory_layer='compressed'`.

_No auto schedule in MVP._

---
## 6. Chat Context Builder
Modes:
### Plain
- Collect last 10 `recent` messages (skip `compressed`). If token overflow (> ~8k including user input), drop oldest until fits; if still large, replace oldest half with channel description text.

### RAG
- Start with Plain context.
- Embed user input; retrieve top 3 `memory_summaries` by cosine similarity.
- Include their `content` appended after a heading `"[Summary]"`.

### ContextSlice Recording
Before LLM call persist `context_slices` row with ids. After assistant reply store as new `messages` row, embed async.

---
## 7. Knowledge Card Creation
Endpoint: `POST /knowledge_cards` body:
```
{ "channel_id":<uuid>, "title":"...", "body":"...", "source_message_ids":[<uuid>,...] }
```
Behavior: Insert record; optional embedding later (not required for MVP). No auto extraction.

---
## 8. REST Endpoints (Summary)
### Channels
- `POST /channels` `{name,parent_id?,system_prompt?}` → channel JSON
- `GET /channels/tree` → nested tree
- `PATCH /channels/{id}` `{name?,system_prompt?,inherit_prompt?}`
- `POST /channels/{id}/summarize_old?retain=50` → creates memory summaries

### Scratchpad / Routing
- `POST /scratchpad` `{content_json}` → staged entry
- `GET /scratchpad/staged` → list staged
- `POST /scratchpad/{id}/route` → compute suggestion `{best_channel?,confidence}`
- `POST /routing/{entry_id}/accept` `{channel_id}` → routes
- `POST /routing/{entry_id}/create_channel` `{name,seed_description?}` → creates + routes
- `POST /routing/{entry_id}/undo`

### Messages / Chat
- `GET /channels/{id}/messages?after=<timestamp>`
- `POST /channels/{id}/messages` `{content,rag?:bool}` → creates user message, triggers assistant reply

### Knowledge Cards
- `POST /knowledge_cards`
- `GET /knowledge_cards?channel_id=<id>`

### Utility
- `GET /health`

### Example: Route Accept Response
```json
{
  "message_id": "uuid-of-created-message",
  "channel_id": "uuid",
  "description_updated": true
}
```

---
## 9. Environment Variables
```
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/second_brain
OPENAI_API_KEY=...
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
ROUTER_THRESHOLD=0.45
TOKEN_BUDGET=8000
```

---
## 10. Startup Instructions
1. `docker compose up -d postgres` or local Postgres.
2. `alembic upgrade head` (creates pgvector extension inside migration).
3. Run backend: `uvicorn app.main:app --reload`.
4. Frontend: `pnpm install && pnpm dev` (Vite default port).
5. Visit frontend, create first channel manually.

---
## 11. Embedding & Cost Control
- Batch embedding operations (async queue) to reduce API round trips.
- Skip embedding for assistant messages >4k tokens (optional) or truncate.
- Reuse existing embeddings for undo (do not recompute).

---
## 12. OpenAI Error Handling
- Network/429: exponential backoff (max 3 retries).
- If embedding fails, leave vector null; router skips until available.

---
## 13. Testing Targets
- Router: below threshold proposes new; above threshold selects existing.
- Undo: removes message + bullet.
- Summarize: memory_summaries created, messages flipped to compressed.
- Chat RAG: context_slices row created with summary IDs.

---
## 14. Future TODO (Out of Scope MVP)
- Auth & multi‑user ACL.
- Graph view / visualization.
- Automatic KnowledgeCard extraction.
- Entity tracking & flashcards.
- Scheduled summarization (cron) & retention policies.
- Semantic search across channels.

---
## 15. Glossary
- **Recent Messages:** messages.memory_layer='recent'.
- **Compressed Summary:** memory_summaries row representing multiple old messages.
- **RAG:** Retrieval Augmented Generation (recent + summaries).
- **ContextSlice:** Audit record of exactly what was sent to the LLM.

---
**End of Spec**

