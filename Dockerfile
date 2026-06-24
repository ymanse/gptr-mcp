FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl also used for NodeSource setup)
RUN apt-get update && apt-get install -y \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 via NodeSource (bundles npm) — required by the Claude CLI.
# The Debian-repo `npm` package pulls a huge, fetch-flaky dependency tree and
# only ships Node 18; NodeSource is the canonical, lighter way to get a modern
# Node into a slim image.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Claude CLI — the claude_agent provider (Claude subscription via the Agent SDK)
# shells out to it for LLM calls. Baked into the image (the SDK wheel does not
# bundle a guaranteed CLI binary).
RUN npm install -g @anthropic-ai/claude-code

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables for Docker
ENV MCP_TRANSPORT=streamable-http
ENV MCP_PATH=/mcp
ENV DOCKER_CONTAINER=true
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=7s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "server.py"] 
