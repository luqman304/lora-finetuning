# syntax=docker/dockerfile:1.7
# CPU inference image: base model + LoRA adapter, served through the `lora-ft predict` CLI.
FROM python:3.12-slim-bookworm
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/models
WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[hub]"
RUN useradd --create-home --uid 10001 app && mkdir -p /models && chown app:app /models
USER app

# Mount a local adapter at /adapter, or pass a Hub repo id with --adapter.
# The base model is downloaded into /models on first run; mount a volume there to cache it.
ENTRYPOINT ["lora-ft", "predict", "--device", "cpu"]
CMD ["--help"]
