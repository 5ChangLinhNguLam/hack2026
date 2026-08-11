# AWS live worker container (plan only)

This directory is a local, reviewable container plan. It does not create an
ECR repository, push an image, or call AWS.

The repository-root `.dockerignore` is an allow-list. The build context omits
practice data, truth/labels/depth, predictions, reports, replay senders,
training notebooks and credentials. The image contains only the live runtime,
the C1 checkpoint and the fingerprinted DMS v13 bundle.

Local review build (after `safeloop.live_worker` and the persistent KVS/WebRTC
adapter are complete):

```bash
docker build --file deploy/aws-live/Dockerfile --tag safeloop-aws-live:local .
docker image inspect safeloop-aws-live:local --format '{{.Size}}'
docker run --rm safeloop-aws-live:local --healthcheck
```

Until then the Dockerfile deliberately fails its full build at the direct
`COPY safeloop/live_worker.py` gate, before dependency installation.
`docker buildx build --check` remains usable for a no-image static review;
there is intentionally no image size to report at this checkpoint.

Release gates require an approved base-image digest and an ECR reference of
the form `repository@sha256:<digest>`. A mutable tag is never accepted by the
ECS task definition. The container must be scanned again after build, and its
filesystem manifest must prove that no `data/`, `predictions/`, `reports/`,
private key or source bundle is present.

The image intentionally has no AWS SDK credentials. On ECS it receives only
the task role through the container credential endpoint. Its root filesystem
is read-only; `/tmp` is the only writable task mount.
