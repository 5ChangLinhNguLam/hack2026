# syntax=docker/dockerfile:1.7

# For a release build, override this with an immutable digest, for example:
#   --build-arg PYTHON_BASE=python:3.10-slim-bookworm@sha256:<approved-digest>
ARG PYTHON_BASE=python:3.10-slim-bookworm
FROM ${PYTHON_BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    SAFELOOP_RUNTIME_ROOT=/opt/safeloop

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/safeloop
# This direct COPY is the plan gate: a full build stops before dependency
# installation until the real KVS/WebRTC worker entry point exists.
COPY safeloop/live_worker.py ./safeloop/live_worker.py
COPY deploy/aws-live/requirements.lock /tmp/requirements.lock
RUN python -m pip install --requirement /tmp/requirements.lock

# .dockerignore is the security boundary: only runtime code and fingerprinted
# model artifacts are available to these COPY instructions.
COPY pyproject.toml ./
COPY C1 ./C1
COPY drive_state ./drive_state
COPY safeloop ./safeloop
COPY models ./models
RUN python -m pip install --no-deps . \
    && python -m compileall -q C1 drive_state safeloop \
    && test ! -e data \
    && test ! -e predictions \
    && test ! -e reports \
    && test ! -e tools

RUN groupadd --gid 65532 safeloop \
    && useradd --uid 65532 --gid 65532 --no-create-home --shell /usr/sbin/nologin safeloop \
    && chown -R 65532:65532 /opt/safeloop
USER 65532:65532

HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
    CMD ["python", "-m", "safeloop.live_worker", "--healthcheck"]

ENTRYPOINT ["python", "-m", "safeloop.live_worker"]
