FROM python:3.12-slim

WORKDIR /app

# System libs required by OpenCV and YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy the full workspace (needed for local workspace packages)
COPY pyproject.toml uv.lock ./
COPY agents-core/ agents-core/
COPY plugins/ plugins/
COPY examples/09_interview_pressure_analyzer/ examples/09_interview_pressure_analyzer/

# Suppress hardlink warning on overlayfs (common in Docker)
ENV UV_LINK_MODE=copy

# Install all workspace dependencies
RUN uv sync --frozen --no-dev

# Persistent session storage
RUN mkdir -p /app/sessions

# Dashboard port (Railway overrides via PORT env var)
EXPOSE 8080

# If CALL_ID is set → run the live agent; otherwise → run the dashboard preview
CMD ["sh", "-c", "\
  if [ -n \"$CALL_ID\" ]; then \
    uv run python examples/09_interview_pressure_analyzer/interview_analyzer.py; \
  else \
    uv run python examples/09_interview_pressure_analyzer/preview_dashboard.py; \
  fi"]
