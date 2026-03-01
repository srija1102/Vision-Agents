FROM python:3.12-slim

WORKDIR /app

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

# Dashboard port
EXPOSE 8080

CMD ["uv", "run", "python", "examples/09_interview_pressure_analyzer/interview_analyzer.py", \
     "--call-id", "${CALL_ID:-hackathon-demo}"]
