# photovault-categorizer

Local, ephemeral ML job that automatically tags and categorizes photos stored in
[PhotoVault](https://github.com/Jarkendar/photovault-server). Uses a frozen CLIP embedder
plus a lightweight zero-shot / k-NN classifier so adding a new label means re-scoring
cached embeddings — not retraining a network.

Runs as `docker run --rm` — nothing stays alive between runs.

---

## Machine split

| Machine | Role | When |
|---|---|---|
| PC (RTX 2070 Super, 8 GB VRAM) | Bulk CLIP embedding; new-label backfill | On-demand (interactive) |
| Pi 5 (8 GB RAM) | Nightly oneshot — embed delta → score → write → exit | Scheduled (systemd timer) |

Both machines connect to the same PostgreSQL instance via **Tailscale**.
A lock file (`/tmp/photovault-categorize.lock`) prevents concurrent runs.

---

## How it works

### Signals

| Phase | Signal | Technique |
|---|---|---|
| 1 | Visual appearance | CLIP embeddings + zero-shot classification / k-NN |
| 2 | People / identity | Face detection + ArcFace embeddings + clustering |
| 3 | Events / trips | Temporal + geographic heuristic clustering |

### Assignment model

Tags and categories in PhotoVault carry a `source` field on each junction row:

```
source ∈ { manual, auto, denied }

Precedence: manual > denied > auto
```

- This job only writes `source = auto` rows.
- Pairs with an existing `manual` or `denied` row are never touched.
- `denied` tombstones (user explicitly unlinked) act as hard negatives — the job skips them permanently.
- Re-scoring updates `score` on existing `auto` rows only.

Each tag and category also has two control flags:

| Flag | Default | Meaning |
|---|---|---|
| `auto_enabled` | `false` | May the bot assign this label automatically? |
| `rolled_out` | `true` | `false` = new label awaiting full library backfill |

The nightly job only works on labels where `auto_enabled = true`.

---

## Phase 1 — CLIP visual pipeline

### Embedder

- **Model:** MobileCLIP-S2 exported to ONNX (~20 MB)
- **Preprocessing:** `center_crop(224)`, `normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])`, `float32`
- Same model on both PC and Pi — identical graph, preprocessing, and floating-point precision — required for a shared vector space.

### Vector store

- **Format:** `hnswlib` index file, keyed by `photo-<uuid>` string IDs.
- One index file per model version (filename carries model id). Bumping the model = full re-embed pass.
- Stored on a shared volume (or `rsync`-ed PC↔Pi over Tailscale before the nightly run).
- Nightly job self-heals: re-embeds any photo missing from the store; prunes vectors for photo ids that no longer exist in Postgres.

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

The CLIP text encoder receives the English prompts. Tags/categories without a mapping entry are skipped by the classifier — the mapping is built incrementally, add a new entry when you create a new tag/category with `auto_enabled = true`.

Prompt template: `"a photo of {label}"`.

### Scripts

#### `bulk_embed.py` (RTX PC — on-demand)

```
python bulk_embed.py
```

1. Connects to Postgres over Tailscale.
2. Fetches all photos where `embedded_at IS NULL` or `embedded_at < model_updated_at`.
3. Downloads `medium.jpg` from the storage root.
4. Computes CLIP embeddings in batches (batch size tuned to 8 GB VRAM).
5. Writes embeddings to the hnswlib index.
6. Updates `photos.embedded_at` and `photos.embedding_model`.

#### `add_label.py` (RTX PC — on-demand, after creating a new tag/category)

```
python add_label.py <category-or-tag-id>
```

1. Sets `rolled_out = false` on the target label.
2. Runs a full library scoring pass (zero-shot or k-NN) for that single label.
3. Inserts `source = auto` rows where no row exists; respects `denied` tombstones.
4. Sets `rolled_out = true`.

#### `categorize.py` (Pi — nightly)

1. Acquires lock file; exits 0 if already locked.
2. **Delta queue:** `SELECT id FROM photos WHERE processing_status = 'pending_categorization'`
3. Downloads + embeds delta photos; writes to store; updates `photos.embedded_at`.
4. Scores each photo against all tags/categories where `auto_enabled = true` and `rolled_out = true`.
5. Inserts `source = auto` junction rows (respects precedence — skips `manual`/`denied` pairs).
6. Flips `photos.processing_status` from `pending_categorization` → `ready` in the same transaction.
7. Logs summary: `photos processed: N, tags inserted: M, categories inserted: K, denied skipped: D`.
8. Releases lock file; exits 0.

### Scoring strategy (to tune)

- **Tags** — multi-label: insert `auto` row if `score >= 0.25` (threshold-based).
- **Categories** — top-1/2: pick the highest-scoring `auto_enabled` category/categories.

These thresholds are open knobs — document final values here once tuned.

---

## Phase 2 — People (planned)

Dedicated face detection + recognition pipeline with a **separate** vector store (face embeddings live in a different space than CLIP scene embeddings).

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

All configuration via environment variables (same `.env` / Docker secrets as the PhotoVault server):

| Variable | Description |
|---|---|
| `DB_URL` | PostgreSQL psycopg2 DSN |
| `DB_USER` | Database user |
| `DB_PASSWORD` | Database password |
| `PHOTO_STORAGE_ROOT` | Root path where `medium.jpg` files are stored |
| `VECTOR_STORE_PATH` | Path to the hnswlib index file |
| `HOME_LAT` | Home latitude (Phase 3) |
| `HOME_LNG` | Home longitude (Phase 3) |
| `HOME_RADIUS_KM` | Home radius in km (Phase 3, default: 50) |

---

## Running

```bash
# Nightly Pi run
docker run --rm \
  --env-file .env \
  -v /path/to/photos:/photos:ro \
  -v /path/to/vectors:/vectors \
  photovault-categorizer

# On-demand bulk embed (PC)
docker run --rm \
  --env-file .env \
  -v /path/to/photos:/photos:ro \
  -v /path/to/vectors:/vectors \
  photovault-categorizer python bulk_embed.py

# Backfill a new label (PC)
docker run --rm \
  --env-file .env \
  -v /path/to/photos:/photos:ro \
  -v /path/to/vectors:/vectors \
  photovault-categorizer python add_label.py cat-<uuid>
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
