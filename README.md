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

## Phase 2 — People

Face detection and identity recognition pipeline with a **separate** face vector store
(ArcFace embeddings live in a different space than CLIP scene embeddings).

- **Model:** InsightFace **buffalo_l** (SCRFD-10G detector + ArcFace R50 → 512-d L2-norm vector, ONNX, local inference)
- **Face store:** `{FACE_STORE_DIR}/faces-buffalo_l.npz` keyed by `face-<uuid>`, separate from CLIP store
- **Face metadata:** `faces` table in PostgreSQL (bbox in `medium.jpg` pixels, `det_score`, `cluster_id`, model)
- **Labelling:** admin API at `/v1/admin/face-clusters` — consumed by a standalone web/desktop tool, NOT the Android app; requires JWT `role=admin`

### Scripts

#### `detect_faces.py` — bulk face detection backfill ✅

```bash
python -m photovault_categorizer.cli.detect_faces
```

Detects faces in all photos that have not yet been processed (or were processed with an older model).
Run on the PC/RTX for the initial backfill; runs on Pi too but is slower.
Chunked and resumable — safe to interrupt and restart.

`categorize.py` also runs face detection as part of the nightly pass for new delta photos.

#### `cluster_faces.py` — group unassigned faces into clusters (on-demand PC)

```bash
python -m photovault_categorizer.cli.cluster_faces
```

Runs DBSCAN (cosine metric, `eps=0.4`, `min_samples=2`) on face vectors that have no cluster yet.
Labelled clusters are frozen — only `cluster_id IS NULL` faces are touched.
Creates `face_clusters` rows in the DB and sets `faces.cluster_id`.
Noise points (DBSCAN label −1) remain unassigned for the next run.

Run after `detect_faces.py` has populated the face store and before using the admin API to label clusters.

### Face pipeline knobs (tune and document values in `config.py`)

| Knob | Default | Meaning |
|---|---|---|
| `FACE_DET_THRESH` | `0.5` | Detector confidence floor — lower = more faces detected (including profiles), higher = fewer false positives |
| `FACE_MIN_PX` | `40` | Minimum face bounding-box side in pixels — filters out distant/blurry faces |
| `FACE_MATCH_THRESHOLD` | `0.5` | Cosine similarity floor for nightly identity matching — tune on photos of the same person at different ages and lighting |

DBSCAN `eps=0.4` / `min_samples=2` are set in `cli/cluster_faces.py`. Adjust if clusters are too coarse (increase `eps`) or too fragmented (decrease `eps`).

### CLI smoke test (no app required)

Complete end-to-end Phase 2 verification via SQL and CLI — no Android app or admin web tool needed.

```bash
# 1. Bulk face detection across the entire library (PC/RTX or Pi)
python -m photovault_categorizer.cli.detect_faces
# Verify: SELECT count(*) FROM faces;
# Verify: SELECT faces_detected_at, face_detection_model FROM photos LIMIT 5;

# 2. Group faces into clusters
python -m photovault_categorizer.cli.cluster_faces
# Verify: SELECT id, face_count, representative_face_id FROM face_clusters;
# Verify: SELECT cluster_id, count(*) FROM faces GROUP BY cluster_id ORDER BY count DESC;

# 3. Inspect a cluster — pick the one that looks like a specific person
psql $DB_URL -c "
  SELECT f.id, f.photo_id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, f.det_score
  FROM faces f
  WHERE f.cluster_id = 'fcluster-<uuid>'
  ORDER BY f.det_score DESC LIMIT 10;"

# 4a. Label via admin API (once deployed)
curl -s -X POST http://localhost:8080/v1/admin/face-clusters/fcluster-<uuid>/label \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"tagId": "tag-<uuid>"}'

# 4b. Label directly via SQL (bridge before the admin API is deployed)
psql $DB_URL <<'SQL'
UPDATE tags SET auto_enabled = true WHERE id = 'tag-<uuid>';
INSERT INTO photo_tags (photo_id, tag_id, score, source, embedding_run)
SELECT DISTINCT f.photo_id, 'tag-<uuid>', f.det_score, 'auto', 'manual-label-cli'
FROM faces f WHERE f.cluster_id = 'fcluster-<uuid>'
ON CONFLICT (photo_id, tag_id) DO UPDATE
    SET score = EXCLUDED.score, embedding_run = EXCLUDED.embedding_run
WHERE photo_tags.source = 'auto';
SQL

# 5. Upload a new photo of the same person, wait for pending_categorization, then:
python -m photovault_categorizer.cli.categorize
# Check log: "faces detected: N, faces matched: M, new unlabeled faces: K"
# Check DB:  SELECT source, score FROM photo_tags WHERE photo_id = '<new-photo-id>';
```

---

## Phase 3 — Events

Pure heuristic trip detection — no model, no network call. Groups photos by time, filters by
distance from home and duration, then creates `auto_enabled` categories and writes
`source = auto` assignments directly (no CLIP prompts involved).

### Scripts

#### `detect_events.py` — trip detection (on-demand) ✅

```bash
python -m photovault_categorizer.cli.detect_events
```

Detects vacation sessions in the library and creates one category per trip.

Algorithm:
1. Fetch all photos with `captured_at` set (GPS optional but needed for the heuristic).
2. Sort by `captured_at`, split into sessions on gaps > `EVENT_GAP_HOURS`.
3. For each session check all three heuristic conditions:
   - GPS centroid is more than `HOME_RADIUS_KM` from home (`HOME_LAT`/`HOME_LNG` required)
   - Session spans at least `EVENT_MIN_DAYS` distinct calendar days
   - Session contains at least `EVENT_MIN_PHOTOS` photos
4. Passing sessions get a category named `Trip · {place} · {year-month}` (most common
   `place_name` in the session, falls back to year-month if no GPS place names).
5. Category is created with `auto_enabled = true`, `rolled_out = true` (idempotent by name).
6. All photos in the session get a `source = auto` `photo_categories` row (respects
   `manual`/`denied` precedence — same as every other categoriser write).

Run on-demand from the PC. Re-running is safe — existing categories are found by name and not duplicated.

### Event pipeline knobs (tune and document values in `config.py`)

| Knob | Default | Meaning |
|---|---|---|
| `HOME_LAT` | _(required)_ | Home latitude — script refuses to run without it |
| `HOME_LNG` | _(required)_ | Home longitude |
| `HOME_RADIUS_KM` | `25.0` | Sessions whose GPS centroid is closer than this are treated as "at home" |
| `EVENT_GAP_HOURS` | `6.0` | Gap between consecutive photos that starts a new session |
| `EVENT_MIN_DAYS` | `2` | Minimum distinct calendar days a session must span |
| `EVENT_MIN_PHOTOS` | `10` | Minimum photos in a session |

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
| `FACE_STORE_DIR` | `./data/faces` | Directory for the face vector `.npz` file |
| `INSIGHTFACE_HOME` | `~/.insightface` | Path where InsightFace looks for model packs (must contain pre-downloaded `buffalo_l`) |
| `FACE_DET_THRESH` | `0.5` | Detector confidence floor (Phase 2) |
| `FACE_MIN_PX` | `40` | Minimum face size in pixels (Phase 2) |
| `FACE_MATCH_THRESHOLD` | `0.5` | Identity matching cosine threshold (Phase 2) |
| `HOME_LAT` | _(required for Phase 3)_ | Home latitude in decimal degrees |
| `HOME_LNG` | _(required for Phase 3)_ | Home longitude in decimal degrees |
| `HOME_RADIUS_KM` | `25.0` | Home radius — sessions closer than this are not proposed as trips (Phase 3) |
| `EVENT_GAP_HOURS` | `6.0` | Gap in hours that starts a new temporal session (Phase 3) |
| `EVENT_MIN_DAYS` | `2` | Minimum calendar days a session must span (Phase 3) |
| `EVENT_MIN_PHOTOS` | `10` | Minimum photos in a session (Phase 3) |

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
