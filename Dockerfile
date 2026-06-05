# photovault-categorizer — ephemeral CLIP categorization job
#
# Target: linux/arm64 (Pi 5) with CPU-only PyTorch.
# For local testing on an x86_64 machine with CUDA, install dependencies in a
# local venv instead (pip install torch --index-url https://download.pytorch.org/whl/cu121).
#
# Build:
#   docker buildx build --platform linux/arm64 -t photovault-categorizer .
#
# Run (nightly):
#   docker run --rm --env-file .env \
#     -v /path/to/photos:/photos:ro \
#     -v /path/to/vectors:/vectors \
#     photovault-categorizer

FROM python:3.12-slim

WORKDIR /app

# System deps needed by Pillow + psycopg binary wheel
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libjpeg62-turbo \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids pulling a CUDA variant from PyPI default index)
RUN pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > requirements_no_torch.txt \
    && pip install --no-cache-dir -r requirements_no_torch.txt \
    && rm requirements_no_torch.txt

# Install the package itself
COPY . .
RUN pip install --no-cache-dir -e . --no-deps

# Default volumes (override with -v at runtime)
VOLUME ["/photos", "/vectors"]

# Environment variable defaults — override with --env-file or -e flags
ENV PHOTO_STORAGE_ROOT=/photos \
    VECTOR_STORE_DIR=/vectors \
    PROMPTS_PATH=/app/prompts.yaml

ENTRYPOINT ["python", "-m", "photovault_categorizer.cli.categorize"]
