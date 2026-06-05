# photovault-categorizer

Local, ephemeral ML job that automatically tags and categorizes photos stored in
[PhotoVault](https://github.com/Jarkendar/photovault-server). Uses a frozen CLIP embedder
plus a lightweight zero-shot classifier — adding a new label means re-scoring
cached embeddings, not retraining a network.

Runs as `docker run --rm` — nothing stays alive between runs.

---

## Implementation status

| Iteration | Scope | Status |
|---|---|---|
| **1** | Scaffold, nightly `categorize.py`, unit tests | ✅ done |
| **2** | `bulk_embed.py`, `add_label.py`, self-heal, systemd timer | ✅ done |

---

## Machine split

| Machine | Role | When |
|---|---|---|
| Pi 5 (8 GB RAM) | Nightly oneshot — embed delta → score → write → exit | Automated (systemd timer) |
| PC (RTX 2070 Super) | Optional one-time bulk embed of a large backlog | On-demand only |

Both machines connect to the same PostgreSQL instance via **Tailscale**.
A lock file (`/tmp/photovault-categorize.lock`) prevents concurrent runs.

`categorize.py` and `add_label.py` run fully on Pi — they only encode text prompts
(cheap) and score cached image vectors (fast matmul). Image encoding on Pi is ~0.5 s/photo
which is fine for the nightly delta of a few dozen photos.
The RTX only matters if you have a large existing photo library to embed all at once.

---

## How it works

### Signals

| Phase | Signal | Technique |
|---|---|---|
| 1 | Visual appearance | CLIP embeddings + zero-shot classification |
| 2 | People / identity | Face detection + ArcFace embeddings + clustering (planned) |
| 3 | Events / trips | Temporal + geographic heuristic clustering (planned) |

### Assignment model

Tags and categories in PhotoVault carry a `source` field on each junction row:

```
source ∈ { manual, auto, denied }

Precedence: manual > denied > auto
```

- This job only writes `source = auto` rows.
- Pairs with an existing `manual` or `denied` row are never touched.
- `denied` tombstones (user explicitly unlinked) act as hard negatives — permanently skipped.
- Re-scoring updates `score` on existing `auto` rows only.

Each tag and category also has two control flags:

| Flag | Default | Meaning |
|---|---|---|
| `auto_enabled` | `false` | May the bot assign this label automatically? |
| `rolled_out` | `true` | `false` = new label awaiting full library backfill |

The nightly job only works on labels where `auto_enabled = true` **and** `rolled_out = true`.

---

## Phase 1 — CLIP visual pipeline

### Embedder

- **Model:** MobileCLIP-S2 via `open_clip_torch`, pretrained weights `datacompdr`, **fp32**
- **Device:** `cuda` if available, otherwise `cpu` — same code runs on RTX (fast tests) and Pi (nightly automat)
- **Preprocessing:** inference transform bundled with `open_clip` (center-crop 224 + normalise, no augmentation)
- **Model id** (stored in DB + vector store filename): `mobileclip-s2-datacompdr`

ONNX export is deferred as a future optimisation. If Pi turns out to be too slow for the
nightly delta, exporting to ONNX fp32 is a drop-in swap — the vector space is identical.

### Vector store

- **Format:** NumPy `.npz` file — `ids: object[N]`, `vectors: float32[N, D]`
- **Location:** `{VECTOR_STORE_DIR}/mobileclip-s2-datacompdr.npz`
- One file per model id. Bumping the model = full re-embed pass.
- Scoring is a dense matmul over all photo vectors — no ANN index needed at this scale
  (~2 KB/photo → 100 MB for 50 k photos).

### PL→EN prompt mapping

Polish tag/category names are mapped to English prompt strings via a static `prompts.yaml` file:

```yaml
"#morze":
  - sea
  - ocean
  - seashore
"#rower":
  - bicycle
  - bike
  - cycling
```

The CLIP text encoder receives the English prompts wrapped in the template
`"a photo of {term}"`. Multiple terms per label are encoded separately and
averaged into a single normalised prototype vector.

Tags/categories without a mapping entry are **skipped** with a `WARNING` log line —
that is the signal to add the entry. Build the file incrementally: add a new entry
whenever you create a new tag/category with `auto_enabled = true`.

### Scripts

#### `categorize.py` — nightly Pi run ✅

```bash
python -m photovault_categorizer.cli.categorize
```

1. Acquires exclusive lock (`/tmp/photovault-categorize.lock`); exits 0 if already held.
2. **Delta queue:** `SELECT id, medium_path FROM photos WHERE processing_status = 'pending_categorization'`.
3. Loads MobileCLIP-S2 + `.npz` store + text prototypes (once per run).
4. For each photo: embed `medium.jpg` → score against all auto-enabled prototypes →
   write `source = auto` junction rows → flip `processing_status` to `'ready'`,
   all in one transaction.
5. Prunes orphan vectors (deleted photos) from the store; saves.
6. Logs: `photos processed: N, tags inserted: M, categories inserted: K, denied skipped: D`.

The nightly run also performs a **self-heal pass** before processing the delta queue: any
photo in `processing_status='ready'` whose vector is absent from the store (e.g. the store
was reset or the photo was deleted and re-uploaded) is re-embedded and re-scored in the same
run. Setting `processing_status='ready'` again is idempotent; `assign_auto` still respects
`manual`/`denied` tombstones.

#### `bulk_embed.py` — one-time full-library backfill ✅

```bash
python -m photovault_categorizer.cli.bulk_embed
```

Embeds all photos missing from the vector store or embedded with an older model.
Use when you have a large existing backlog. Runs on Pi too, just slower (~0.5 s/photo CPU).
The run is resumable — checkpointed every 256 photos, so interrupting and restarting is safe.

Does **not** score or write junction rows — run `categorize.py` separately to assign labels.

#### `add_label.py` — new-label backfill ✅

```bash
python -m photovault_categorizer.cli.add_label <tag-or-category-id>
```

Backfills a single newly created label against **cached** vectors — no re-embedding.

Requirements before running:
1. The tag/category must exist in the DB with `auto_enabled = true`.
2. The label name must have an entry in `prompts.yaml`.
3. For categories: `CATEGORY_MIN_SCORE` must be > 0 (otherwise every photo matches).

The script brackets the backfill pass with `rolled_out = false` → score → `rolled_out = true`.
`assign_auto` still respects `manual`/`denied` tombstones so no user edits are overwritten.

### Scoring knobs (to tune)

| Knob | Default | Meaning |
|---|---|---|
| `TAG_THRESHOLD` | `0.25` | Minimum cosine similarity for a tag to be assigned |
| `CATEGORY_TOP_K` | `1` | Maximum number of categories to assign per photo |
| `CATEGORY_MIN_SCORE` | `0.0` | Minimum score a category must reach to be assigned |

Note: `CATEGORY_MIN_SCORE=0.0` is safe for the nightly `top-k` ranking (it selects the
best-scoring category regardless) but **must be set to a positive value** (e.g. `0.20`)
before using `add_label.py` for categories — otherwise every photo would be assigned.

Document final values here once tuned against real photos.

---

## Phase 2 — People (planned)

Dedicated face detection + recognition pipeline with a **separate** vector store
(face embeddings live in a different space than CLIP scene embeddings).

- Face model: InsightFace / ArcFace ONNX (lightweight enough for Pi inference).
- Separate `faces.hnsw` store keyed by `face-<uuid>` → `photo-<uuid>` + bounding box.
- Identity clustering: DBSCAN / agglomerative → proposed identity → maps to a `tag-*` or `category-*`.
- Once labelled: `auto_enabled = true` on the person tag/category; nightly runs assign new matching photos.

---

## Phase 3 — Events (planned)

Pure heuristic clustering over `captured_at` + `lat`/`lng` — no model, no network call.

Vacation heuristic (to tune):
- Away from home: lat/lng centroid outside a configurable home radius.
- Multi-day: cluster spans ≥ 2 calendar days.
- Minimum photos: ≥ 10 photos in the cluster.

Outputs proposed categories (`"Trip · {place_name} · {year-month}"`) with `rolled_out = false`;
`add_label.py` then backfills the assignments.

---

## Configuration

All configuration via environment variables. Copy `.env.example` to `.env` and fill in the values.

| Variable | Default | Description |
|---|---|---|
| `DB_URL` | `jdbc:postgresql://localhost:5432/photovault` | PostgreSQL URL — JDBC or plain psycopg3 format |
| `DB_USER` | `photovault` | Database user |
| `DB_PASSWORD` | _(empty)_ | Database password |
| `PHOTO_STORAGE_ROOT` | `./data/photos` | Root path where `medium.jpg` files are stored |
| `VECTOR_STORE_DIR` | `./data/vectors` | Directory for `.npz` vector store files |
| `PROMPTS_PATH` | `prompts.yaml` | Path to the PL→EN prompt mapping file |
| `TAG_THRESHOLD` | `0.25` | Cosine similarity floor for tag assignment |
| `CATEGORY_TOP_K` | `1` | Max categories assigned per photo |
| `CATEGORY_MIN_SCORE` | `0.0` | Minimum score for any category assignment |

---

## Running

```bash
# Nightly Pi run (Docker — automated via systemd timer, see deploy/)
docker run --rm \
  --env-file .env \
  -v /path/to/photos:/photos:ro \
  -v /path/to/vectors:/vectors \
  photovault-categorizer

# Local run for testing (venv, uses CUDA if available)
python -m photovault_categorizer.cli.categorize

# One-time bulk embed of an existing library backlog
docker run --rm --env-file .env \
  -v /path/to/photos:/photos:ro -v /path/to/vectors:/vectors \
  photovault-categorizer python -m photovault_categorizer.cli.bulk_embed

# Backfill a newly created label against cached vectors
docker run --rm --env-file .env \
  -v /path/to/photos:/photos:ro -v /path/to/vectors:/vectors \
  photovault-categorizer python -m photovault_categorizer.cli.add_label tag-<uuid>
```

For automated deployment on the Pi (systemd timer firing daily at 03:00), see
[`deploy/README.md`](deploy/README.md).

### Local development setup

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools
.venv/bin/pip install torch          # add --index-url .../cu121 for CUDA
.venv/bin/pip install -e ".[dev]"

# Unit tests (no DB required)
.venv/bin/pytest tests/

# Integration tests only (requires running Postgres matching DB_URL; auto-skipped otherwise)
.venv/bin/pytest tests/test_write_precedence.py tests/test_db_iter2.py
```

---

## Relation to photovault-server

This repo is included as a git submodule in
[photovault-server](https://github.com/Jarkendar/photovault-server) under `categorizer/`.
The server and the categorizer share the same PostgreSQL database but no code — the server
is a Ktor JVM process, the categorizer is an ephemeral Python container.

Schema changes required by new categorizer phases are tracked in
[`CATEGORIZER.md`](https://github.com/Jarkendar/photovault-server/blob/main/CATEGORIZER.md)
in the server repo.
