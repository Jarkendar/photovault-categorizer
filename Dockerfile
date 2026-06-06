# photovault-categorizer — ephemeral CLIP + face categorization job
#
# Target: linux/arm64 (Pi 5) with CPU-only PyTorch and ONNX Runtime.
# For local testing on an x86_64 machine with CUDA, install dependencies in a
# local venv instead:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
#   pip install onnxruntime-gpu   # instead of onnxruntime
#
# Build:
#   docker buildx build --platform linux/arm64 -t photovault-categorizer .
#
# Run (nightly):
#   docker run --rm --env-file .env \
#     -v /path/to/photos:/photos:ro \
#     -v /path/to/vectors:/vectors \
#     -v /path/to/insightface-models:/root/.insightface \
#     photovault-categorizer
#
# The InsightFace buffalo_l pack (~300 MB) is pre-downloaded into the image
# at build time (see step below) so the runtime container has no network access.
# INSIGHTFACE_HOME inside the image is /root/.insightface.

FROM python:3.12-slim

WORKDIR /app

# System deps:
#   libpq5              — psycopg binary wheel
#   libjpeg62-turbo     — Pillow JPEG support
#   libpng16-16         — Pillow PNG support
#   libgl1 libglib2.0-0 — OpenCV (opencv-python-headless) runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libjpeg62-turbo \
        libpng16-16 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch + torchvision from the CPU wheel index.
# Both must come from the same index — mixing PyPI torchvision with a +cpu torch
# causes a RuntimeError at import ("operator torchvision::nms does not exist").
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies, skipping torch/torchvision to avoid
# a second download from PyPI that would overwrite the CPU wheels above.
COPY requirements.txt .
RUN grep -vE '^torch(vision)?' requirements.txt > requirements_no_torch.txt \
    && pip install --no-cache-dir -r requirements_no_torch.txt \
    && rm requirements_no_torch.txt

# Install the package itself
COPY . .
RUN pip install --no-cache-dir -e . --no-deps

# Pre-download the InsightFace buffalo_l model pack at build time.
# This prevents any network access at runtime — critical for Pi 5 offline runs.
# The pack is stored under INSIGHTFACE_HOME (/root/.insightface/models/buffalo_l/).
ENV INSIGHTFACE_HOME=/root/.insightface
RUN python - <<'EOF'
import insightface
from insightface.app import FaceAnalysis
# Trigger model download; providers list is irrelevant here — just need the files.
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_thresh=0.5, det_size=(512, 512))
print("buffalo_l pack downloaded successfully")
EOF

# Default volumes (override with -v at runtime)
VOLUME ["/photos", "/vectors"]

# Environment variable defaults — override with --env-file or -e flags
ENV PHOTO_STORAGE_ROOT=/photos \
    VECTOR_STORE_DIR=/vectors \
    FACE_STORE_DIR=/vectors \
    PROMPTS_PATH=/app/prompts.yaml

ENTRYPOINT ["python", "-m", "photovault_categorizer.cli.categorize"]
