#!/usr/bin/env python3
"""Truth-free local live-model soak for prepared CarSky recorded streams.

The source boundary is deliberately narrow: this harness accepts only
``TruthFreeRecordedBundle`` instances validated by
``carsky_recorded_stream_sender.py``.  JPEG/PNG payloads are decoded locally,
then joined to road/cabin/ego envelopes by their explicit session,
generation, source-sequence, media-ID, and RTP identities.  There is no raw
dataset fallback, synthetic frame/ego path, precomputed prediction reader,
AWS client, or H.264 claim in this module.

The inference controller runs on one dedicated worker.  Its ingress handoff
has one waiting slot (plus the frame currently executing); saturation aborts
the current source session rather than growing a queue or silently rebasing
identity.  Decision-v2 serialization runs in the asynchronous
``LatestOnlyKuksaMirror`` publisher thread against a local mock sink.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safeloop.carsky_live import (  # noqa: E402
    CabinFrameEnvelope,
    EgoTelemetryEnvelope,
    LatestRuntimeSnapshotSlot,
    LiveInferenceController,
    LiveInferenceSession,
    RoadFrameEnvelope,
    RtpFrameIdentity,
    SourceTickKey,
)
from safeloop.live_decision import LiveDecisionV2Builder  # noqa: E402
from safeloop.live_publish import LatestOnlyKuksaMirror  # noqa: E402
from tools.carsky_recorded_stream_sender import (  # noqa: E402
    FRAME_SCHEMA,
    INFERENCE_MODE,
    PERIOD_NS,
    RTP_CLOCK_RATE_HZ,
    RTP_MODULUS,
    SOURCE_HZ,
    VIDEO_SOURCE,
    RecordedStreamSender,
    SenderClock,
    SystemClock,
    TransportFrame,
    TransportSession,
    TruthFreeRecordedBundle,
)


REPORT_SCHEMA = "safeloop.carsky.live-soak-report.v1"
MEDIA_CODEC = "IMAGE_FILE_DECODE"
H264_INCLUDED = False
AWS_CONNECTIVITY = False
INGRESS_CAPACITY = 1
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".carsky-build" / "soak"
DEFAULT_BUNDLES = (
    REPO_ROOT / ".carsky-build" / "demo" / "T01-Sample",
    REPO_ROOT / ".carsky-build" / "demo" / "T02-Sample",
)
DEFAULT_C1_CHECKPOINT = REPO_ROOT / "C1" / "student_ttc.pth"
DEFAULT_C2_BUNDLE = REPO_ROOT / "models" / "driver_state_phase_2_v13"
MAX_LATENCY_RESERVOIR = 4096
MAX_ERROR_ITEMS = 256


class SoakError(RuntimeError):
    """A local soak contract or runtime operation failed."""


class InferenceBackpressure(SoakError):
    """The capacity-one inference handoff could not accept another tick."""


class SessionInferenceError(SoakError):
    """The dedicated inference worker faulted the active source session."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_non_negative(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and non-negative")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _positive_duration(value: object, *, field: str) -> float:
    number = _finite_non_negative(value, field=field)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _json_value(value: Any) -> Any:
    """Detach immutable/runtime values into strict JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _strict_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            _json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SoakError(f"soak report is not strict JSON: {exc}") from exc
    if not isinstance(decoded, dict) or decoded.get("schema") != REPORT_SCHEMA:
        raise SoakError("soak report root/schema is invalid")
    return encoded + b"\n"


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile of no values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


class _LatencyAccumulator:
    """Exact count/mean/max with a bounded deterministic quantile reservoir."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.samples: list[float] = []

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        if len(self.samples) < MAX_LATENCY_RESERVOIR:
            self.samples.append(value)
            return
        # Deterministic Algorithm-R style replacement. Quantiles become an
        # approximation after the fixed reservoir fills; count/mean/max stay exact.
        candidate = ((self.count * 1_103_515_245 + 12_345) & 0x7FFFFFFF) % self.count
        if candidate < MAX_LATENCY_RESERVOIR:
            self.samples[candidate] = value

    def summary(self) -> dict[str, float | int | bool | str | None]:
        if self.count == 0:
            return {
                "count": 0,
                "sample_count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "p99": None,
                "max": None,
                "quantiles_approximate": False,
                "method": "bounded_deterministic_reservoir",
            }
        ordered = sorted(self.samples)
        return {
            "count": self.count,
            "sample_count": len(ordered),
            "mean": round(self.total / self.count, 6),
            "p50": round(_percentile(ordered, 0.50), 6),
            "p95": round(_percentile(ordered, 0.95), 6),
            "p99": round(_percentile(ordered, 0.99), 6),
            "max": round(self.maximum, 6),
            "quantiles_approximate": self.count > len(ordered),
            "method": "bounded_deterministic_reservoir",
        }


class SoakMetrics:
    """Small thread-safe metrics collector shared by all three workers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._maxima: dict[str, float] = {}
        self._latencies: dict[str, _LatencyAccumulator] = {}
        self._errors: list[dict[str, Any]] = []
        self._errors_omitted = 0

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[name] += int(amount)

    def maximum(self, name: str, value: float) -> None:
        with self._lock:
            self._maxima[name] = max(float(value), self._maxima.get(name, 0.0))

    def observe(self, name: str, milliseconds: float) -> None:
        value = float(milliseconds)
        if not math.isfinite(value) or value < 0.0:
            return
        with self._lock:
            self._latencies.setdefault(name, _LatencyAccumulator()).observe(value)

    def error(
        self,
        operation: str,
        error: BaseException,
        *,
        session_id: str | None = None,
        source_sequence: int | None = None,
    ) -> None:
        entry = {
            "operation": operation,
            "type": type(error).__name__,
            "message": str(error)[:500],
            "session_id": session_id,
            "source_sequence": source_sequence,
        }
        with self._lock:
            if len(self._errors) < MAX_ERROR_ITEMS:
                self._errors.append(entry)
            else:
                self._errors_omitted += 1

    def count(self, name: str) -> int:
        with self._lock:
            return int(self._counts[name])

    def snapshot(self) -> tuple[dict[str, int], dict[str, float], dict[str, Any], list[dict[str, Any]]]:
        with self._lock:
            counts = {key: int(value) for key, value in self._counts.items()}
            counts["metric_error_items_omitted"] = self._errors_omitted
            maxima = dict(self._maxima)
            latencies = {
                key: accumulator.summary()
                for key, accumulator in self._latencies.items()
            }
            errors = [dict(item) for item in self._errors]
            if self._errors_omitted:
                errors.append(
                    {
                        "operation": "metrics",
                        "type": "ItemsOmitted",
                        "message": f"{self._errors_omitted} additional errors omitted",
                        "session_id": None,
                        "source_sequence": None,
                    }
                )
        return counts, maxima, latencies, errors


def _proc_memory_bytes() -> dict[str, int | None]:
    values: dict[str, int | None] = {"rss": None, "rss_high_water": None}
    try:
        fields: dict[str, int] = {}
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                name, raw, unit = line.split()[:3]
                if unit != "kB":
                    continue
                fields[name.rstrip(":")] = int(raw) * 1024
        values["rss"] = fields.get("VmRSS")
        values["rss_high_water"] = fields.get("VmHWM")
    except (OSError, UnicodeError, ValueError):
        pass
    return values


def _cuda_memory(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "device": device,
        "device_name": None,
        "total_bytes": None,
        "allocated_bytes": None,
        "reserved_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return result
        resolved = torch.device("cuda" if device == "auto" else device)
        if resolved.type != "cuda":
            return result
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "available": True,
                "device": str(resolved),
                "device_name": properties.name,
                "total_bytes": int(properties.total_memory),
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            }
        )
    except Exception as exc:
        result["sample_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return result


def _reset_cuda_peak(device: str) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            resolved = torch.device("cuda" if device == "auto" else device)
            if resolved.type == "cuda":
                torch.cuda.reset_peak_memory_stats(resolved)
    except Exception:
        pass


def _cuda_synchronize(device: str) -> None:
    try:
        import torch

        resolved = torch.device("cuda" if device == "auto" else device)
        if resolved.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(resolved)
    except Exception:
        # The concrete runtimes validate CUDA availability during model load.
        # Timing instrumentation must not replace their useful exception.
        pass


def _is_oom(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return "out of memory" in text or "cuda oom" in text


class ResourceSampler:
    """Capture process CPU/RSS and allocator-level CUDA memory peaks."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.started_usage = resource.getrusage(resource.RUSAGE_SELF)
        self.started_memory = _proc_memory_bytes()
        self.started_cuda = _cuda_memory(device)
        self.max_rss = self.started_memory["rss"] or 0
        self.max_cuda_allocated = self.started_cuda.get("allocated_bytes") or 0
        self.max_cuda_reserved = self.started_cuda.get("reserved_bytes") or 0
        self._started_ns = time.perf_counter_ns()
        self.samples: list[dict[str, Any]] = [
            {
                "phase": "process_start",
                "elapsed_seconds": 0.0,
                "rss_bytes": self.started_memory["rss"],
                "cuda_allocated_bytes": self.started_cuda.get("allocated_bytes"),
                "cuda_reserved_bytes": self.started_cuda.get("reserved_bytes"),
            }
        ]

    def sample(self, phase: str = "sample") -> None:
        memory = _proc_memory_bytes()
        cuda = _cuda_memory(self.device)
        self.max_rss = max(self.max_rss, memory["rss"] or 0)
        self.max_cuda_allocated = max(
            self.max_cuda_allocated, cuda.get("allocated_bytes") or 0
        )
        self.max_cuda_reserved = max(
            self.max_cuda_reserved, cuda.get("reserved_bytes") or 0
        )
        self.samples.append(
            {
                "phase": phase,
                "elapsed_seconds": round(
                    (time.perf_counter_ns() - self._started_ns) / 1_000_000_000.0,
                    6,
                ),
                "rss_bytes": memory["rss"],
                "cuda_allocated_bytes": cuda.get("allocated_bytes"),
                "cuda_reserved_bytes": cuda.get("reserved_bytes"),
            }
        )

    def report(self, wall_seconds: float) -> dict[str, Any]:
        self.sample("report_end")
        ended_usage = resource.getrusage(resource.RUSAGE_SELF)
        ended_memory = _proc_memory_bytes()
        ended_cuda = _cuda_memory(self.device)
        user_cpu = ended_usage.ru_utime - self.started_usage.ru_utime
        system_cpu = ended_usage.ru_stime - self.started_usage.ru_stime
        total_cpu = user_cpu + system_cpu
        return {
            "cpu": {
                "user_seconds": round(user_cpu, 6),
                "system_seconds": round(system_cpu, 6),
                "total_seconds": round(total_cpu, 6),
                "percent_of_one_core": round(
                    100.0 * total_cpu / wall_seconds, 3
                )
                if wall_seconds > 0.0
                else None,
            },
            "memory": {
                "rss_start_bytes": self.started_memory["rss"],
                "rss_end_bytes": ended_memory["rss"],
                "rss_max_sampled_bytes": self.max_rss or None,
                "rss_high_water_bytes": ended_memory["rss_high_water"],
            },
            "vram": {
                "start": self.started_cuda,
                "end": ended_cuda,
                "max_sampled_allocated_bytes": self.max_cuda_allocated or None,
                "max_sampled_reserved_bytes": self.max_cuda_reserved or None,
            },
            "samples": list(self.samples),
        }


class SessionConfigurable(Protocol):
    def configure_session(
        self, metadata: Mapping[str, Any], *, expected_frames: int
    ) -> None: ...


@dataclass
class RuntimeStack:
    """Resident challenge processors and immutable decision model identity."""

    c1: Any
    c2: Any
    c3: Any
    drive_quality: Any
    contextual_risk: Any
    models: Mapping[str, Mapping[str, str]]
    closeables: tuple[Any, ...] = ()

    def close(self) -> None:
        errors: list[BaseException] = []
        seen: set[int] = set()
        for processor in self.closeables:
            if id(processor) in seen:
                continue
            seen.add(id(processor))
            close = getattr(processor, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise SoakError(
                "runtime close failed: "
                + "; ".join(f"{type(item).__name__}: {item}" for item in errors)
            )


class _SessionC1Runtime:
    """Change sanitized scalar metadata without reloading the C1 model."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def configure_session(
        self, metadata: Mapping[str, Any], *, expected_frames: int
    ) -> None:
        del expected_frames
        from C1.features import FEATURES_USED, ScalarFeatureStream

        self.runtime.scalar_stream = ScalarFeatureStream(
            metadata=metadata,
            sample_hz=self.runtime.sample_hz,
            feature_order=FEATURES_USED,
        )

    def reset(self) -> None:
        self.runtime.reset()

    def process_bgr(self, *args: Any, **kwargs: Any) -> Any:
        return self.runtime.process_bgr(*args, **kwargs)

    def close(self) -> None:
        self.runtime.close()


class _SessionC3Runtime:
    """Recreate only the lightweight C3 accumulator at a session boundary."""

    def __init__(self) -> None:
        self.runtime: Any | None = None

    def configure_session(
        self, metadata: Mapping[str, Any], *, expected_frames: int
    ) -> None:
        from safeloop.c3 import Challenge3Accumulator

        if "speed_limit_kmh" not in metadata:
            raise SoakError("sanitized bundle metadata lacks speed_limit_kmh")
        self.runtime = Challenge3Accumulator(
            speed_limit_kmh=float(metadata["speed_limit_kmh"]),
            expected_frames=expected_frames,
        )

    def reset(self) -> None:
        if self.runtime is None:
            raise SoakError("C3 session metadata was not configured")
        self.runtime.reset()

    def update(self, *args: Any, **kwargs: Any) -> Any:
        if self.runtime is None:
            raise SoakError("C3 session metadata was not configured")
        return self.runtime.update(*args, **kwargs)


class _TimedC1:
    def __init__(self, inner: Any, metrics: SoakMetrics) -> None:
        self.inner = inner
        self.metrics = metrics

    def configure_session(
        self, metadata: Mapping[str, Any], *, expected_frames: int
    ) -> None:
        configure = getattr(self.inner, "configure_session", None)
        if callable(configure):
            configure(metadata, expected_frames=expected_frames)

    def reset(self) -> None:
        self.inner.reset()

    def process_bgr(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            result = self.inner.process_bgr(*args, **kwargs)
        finally:
            self.metrics.observe(
                "c1_component_ms", (time.perf_counter_ns() - started) / 1_000_000.0
            )
            self.metrics.increment("c1_calls")
        if bool(getattr(result, "model_updated", False)):
            self.metrics.increment("c1_model_updates")
            native = getattr(result, "latency_ms", None)
            if isinstance(native, (int, float)):
                self.metrics.observe("c1_model_native_update_ms", float(native))
        return result


class _TimedC2:
    def __init__(self, inner: Any, metrics: SoakMetrics) -> None:
        self.inner = inner
        self.metrics = metrics

    def reset(self) -> None:
        self.inner.reset()

    def process_bgr(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            result = self.inner.process_bgr(*args, **kwargs)
        finally:
            self.metrics.observe(
                "c2_component_ms", (time.perf_counter_ns() - started) / 1_000_000.0
            )
            self.metrics.increment("c2_calls")
        native = getattr(result, "latency_ms", None)
        if isinstance(native, (int, float)):
            self.metrics.observe("c2_model_native_ms", float(native))
        return result


class _TimedC3:
    def __init__(self, inner: Any, metrics: SoakMetrics) -> None:
        self.inner = inner
        self.metrics = metrics

    def configure_session(
        self, metadata: Mapping[str, Any], *, expected_frames: int
    ) -> None:
        configure = getattr(self.inner, "configure_session", None)
        if callable(configure):
            configure(metadata, expected_frames=expected_frames)

    def reset(self) -> None:
        self.inner.reset()

    def update(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return self.inner.update(*args, **kwargs)
        finally:
            self.metrics.observe(
                "c3_component_ms", (time.perf_counter_ns() - started) / 1_000_000.0
            )
            self.metrics.increment("c3_calls")


class _TimedDriveQuality:
    def __init__(self, inner: Any, metrics: SoakMetrics) -> None:
        self.inner = inner
        self.metrics = metrics

    def reset(self) -> None:
        self.inner.reset()

    def update(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return self.inner.update(*args, **kwargs)
        finally:
            self.metrics.observe(
                "drive_quality_component_ms",
                (time.perf_counter_ns() - started) / 1_000_000.0,
            )
            self.metrics.increment("drive_quality_calls")


class _TimedRisk:
    def __init__(self, inner: Any, metrics: SoakMetrics) -> None:
        self.inner = inner
        self.metrics = metrics

    def reset(self) -> None:
        reset = getattr(self.inner, "reset", None)
        if callable(reset):
            reset()

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return self.inner.evaluate(*args, **kwargs)
        finally:
            self.metrics.observe(
                "contextual_risk_component_ms",
                (time.perf_counter_ns() - started) / 1_000_000.0,
            )
            self.metrics.increment("contextual_risk_calls")


def _model_map(c1_checkpoint: Path, c2_bundle: Path) -> dict[str, dict[str, str]]:
    from safeloop import c3 as c3_module
    from safeloop import contextual_risk as risk_module
    from safeloop import drive_quality as quality_module

    c2_manifest = c2_bundle / "bundle_manifest.json"
    c2_payload = _validate_c2_bundle_manifest(c2_bundle)
    c2_version = str(c2_payload.get("bundle", c2_bundle.name))
    return {
        "c1": {
            "version": c1_checkpoint.stem,
            "digest_sha256": _sha256_file(c1_checkpoint),
        },
        "c2": {
            "version": c2_version,
            "digest_sha256": _sha256_file(c2_manifest),
        },
        "c3": {
            "version": c3_module.FORMULA_VERSION,
            "digest_sha256": _sha256_file(Path(c3_module.__file__).resolve()),
        },
        "drive_quality": {
            "version": quality_module.FORMULA_VERSION,
            "digest_sha256": _sha256_file(Path(quality_module.__file__).resolve()),
        },
        "contextual_risk": {
            "version": risk_module.POLICY_VERSION,
            "digest_sha256": _sha256_file(Path(risk_module.__file__).resolve()),
        },
    }


def _validate_c2_bundle_manifest(c2_bundle: Path) -> dict[str, Any]:
    """Fail closed on every DMS artifact before any model is constructed."""

    root = c2_bundle.resolve(strict=True)
    manifest_path = root / "bundle_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SoakError("C2 bundle_manifest.json must be a regular non-symlink file")
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SoakError(f"cannot parse C2 bundle manifest: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("sha256"), dict):
        raise SoakError("C2 bundle manifest must contain a sha256 object")
    hashes = payload["sha256"]
    if not hashes:
        raise SoakError("C2 bundle manifest sha256 inventory is empty")
    required_artifacts = {
        "face_landmarker.task",
        "version-RFB-320.onnx",
        "visual/best.pt",
        "ocular/best.pt",
        "gate_ocular/best.pt",
        "temporal/best.pt",
    }
    if not required_artifacts.issubset(hashes):
        raise SoakError(
            "C2 manifest does not bind every DMSBundle model artifact"
        )
    for raw_name, raw_digest in hashes.items():
        if not isinstance(raw_name, str) or not raw_name or "\\" in raw_name:
            raise SoakError("C2 manifest contains an unsafe artifact path")
        relative = Path(raw_name)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise SoakError(f"C2 manifest contains an unsafe artifact path: {raw_name!r}")
        if (
            not isinstance(raw_digest, str)
            or len(raw_digest) != 64
            or any(character not in "0123456789abcdef" for character in raw_digest)
        ):
            raise SoakError(f"C2 manifest digest is invalid for {raw_name}")
        artifact = root.joinpath(*relative.parts)
        if artifact.is_symlink() or not artifact.is_file():
            raise SoakError(f"C2 manifest artifact is absent/non-regular: {raw_name}")
        if _sha256_file(artifact) != raw_digest:
            raise SoakError(f"C2 manifest checksum mismatch: {raw_name}")
    return payload


def build_real_runtime_stack(
    *,
    initial_metadata: Mapping[str, Any],
    c1_checkpoint: Path,
    c2_bundle: Path,
    device: str,
) -> RuntimeStack:
    """Load C1/C2 once; session resets never reconstruct their models."""

    from C1.runtime import StudentTTCRuntime
    from drive_state.phase_2.replay import DMSBundle, GeneralDMS
    from safeloop.contextual_risk import ContextualRiskPolicy
    from safeloop.drive_quality import DriveQualityAccumulator

    checkpoint = c1_checkpoint.resolve(strict=True)
    dms_root = c2_bundle.resolve(strict=True)
    # DMSBundle.validate() checks existence. Verify the checksum inventory first
    # so the manifest digest below actually identifies the loaded artifacts.
    _validate_c2_bundle_manifest(dms_root)
    bundle = DMSBundle.at(dms_root)
    bundle.validate()
    c1_runtime = StudentTTCRuntime(
        checkpoint,
        metadata=initial_metadata,
        source_fps=float(SOURCE_HZ),
        stride=2,
        device=device,
    )
    try:
        c2_runtime = GeneralDMS(
            bundle,
            device=device,
            fps=float(SOURCE_HZ),
        )
    except BaseException:
        c1_runtime.close()
        raise
    c1 = _SessionC1Runtime(c1_runtime)
    return RuntimeStack(
        c1=c1,
        c2=c2_runtime,
        c3=_SessionC3Runtime(),
        drive_quality=DriveQualityAccumulator(),
        contextual_risk=ContextualRiskPolicy(),
        models=MappingProxyType(_model_map(checkpoint, dms_root)),
        closeables=(c2_runtime, c1),
    )


@dataclass(frozen=True)
class PublisherFrameRef:
    """Small publish reference; decoded images never enter the broker queue."""

    session_id: str
    generation: int
    sequence: int
    source_sequence: int | None
    capture_timestamp_ms: int | None
    health: str


class LocalMockDecisionSink:
    """Serialize decision-v2 on the mirror worker and simulate broker behavior."""

    def __init__(
        self,
        metrics: SoakMetrics,
        *,
        delay_ms: float = 0.0,
        fail_every: int = 0,
        epoch_ms: Callable[[], int] | None = None,
    ) -> None:
        self.metrics = metrics
        self.delay_ms = _finite_non_negative(delay_ms, field="publisher delay")
        self.fail_every = _non_negative_int(fail_every, field="publisher fail_every")
        self.epoch_ms = epoch_ms or (lambda: time.time_ns() // 1_000_000)
        self._lock = threading.Lock()
        self._attempts = 0
        self._serialized = 0
        self._successful = 0
        self._simulated_failures = 0
        self._data_attempts = 0
        self._heartbeat_attempts = 0
        self._closed = False
        self._threads: set[str] = set()
        self._max_payload_bytes = 0
        self._last_identity: tuple[str, int, int] | None = None
        self._duplicates = 0
        self._reorders = 0
        self._gaps = 0
        self._clock_statuses: Counter[str] = Counter()
        self._health_statuses: Counter[str] = Counter()

    def publish(self, frame: PublisherFrameRef, envelope: Any) -> None:
        started = time.perf_counter_ns()
        if (frame.session_id, frame.generation, frame.sequence) != (
            envelope.session_id,
            envelope.generation,
            envelope.sequence,
        ):
            self.metrics.increment("publisher_identity_errors")
            raise SoakError("publisher frame/envelope identity mismatch")
        if (
            frame.source_sequence is not None
            and frame.source_sequence != envelope.source_sequence
        ):
            self.metrics.increment("publisher_identity_errors")
            raise SoakError("publisher source_sequence identity mismatch")
        if (
            frame.capture_timestamp_ms is not None
            and frame.capture_timestamp_ms != envelope.capture_timestamp_ms
        ):
            self.metrics.increment("publisher_identity_errors")
            raise SoakError("publisher capture timestamp identity mismatch")
        serialization_started = started
        payload = envelope.to_mqtt_json_bytes()
        serialization_ms = (
            time.perf_counter_ns() - serialization_started
        ) / 1_000_000.0
        self.metrics.observe("publisher_serialize_ms", serialization_ms)
        self.metrics.maximum("max_envelope_bytes", len(payload))
        with self._lock:
            self._attempts += 1
            attempt = self._attempts
            self._serialized += 1
            if frame.source_sequence is None:
                self._heartbeat_attempts += 1
            else:
                self._data_attempts += 1
            self._threads.add(threading.current_thread().name)
            self._max_payload_bytes = max(self._max_payload_bytes, len(payload))
            current = (envelope.session_id, envelope.generation, envelope.sequence)
            previous = self._last_identity
            if previous is not None and current[:2] == previous[:2]:
                if current[2] == previous[2]:
                    self._duplicates += 1
                elif current[2] < previous[2]:
                    self._reorders += 1
                elif current[2] > previous[2] + 1:
                    self._gaps += current[2] - previous[2] - 1
            self._last_identity = current
            self._clock_statuses[str(envelope.clock["status"])] += 1
            self._health_statuses[str(envelope.health["status"])] += 1
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        if frame.capture_timestamp_ms is not None:
            age = self.epoch_ms() - frame.capture_timestamp_ms
            if age >= 0:
                self.metrics.observe("end_to_end_publish_ms", float(age))
        self.metrics.observe(
            "publisher_call_ms", (time.perf_counter_ns() - started) / 1_000_000.0
        )
        if self.fail_every and attempt % self.fail_every == 0:
            with self._lock:
                self._simulated_failures += 1
            raise ConnectionError(f"simulated local broker outage at attempt {attempt}")
        with self._lock:
            self._successful += 1

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "kind": "LOCAL_MOCK_DECISION_SINK",
                "attempts": self._attempts,
                "serialized": self._serialized,
                "successful": self._successful,
                "simulated_failures": self._simulated_failures,
                "data_attempts": self._data_attempts,
                "heartbeat_attempts": self._heartbeat_attempts,
                "closed": self._closed,
                "publisher_threads": sorted(self._threads),
                "max_payload_bytes": self._max_payload_bytes,
                "duplicates": self._duplicates,
                "reorders": self._reorders,
                "sequence_gaps_from_coalescing": self._gaps,
                "clock_status_counts": dict(self._clock_statuses),
                "health_status_counts": dict(self._health_statuses),
            }


@dataclass(frozen=True)
class DecodedTick:
    session_id: str
    generation: int
    source_sequence: int
    source_media_timestamp_ms: int
    server_receive_timestamp_ms: int
    road: RoadFrameEnvelope
    cabin: CabinFrameEnvelope
    ego: EgoTelemetryEnvelope


@dataclass
class _ControlRequest:
    operation: str
    payload: Any
    done: threading.Event
    result: Any = None
    error: BaseException | None = None


@dataclass(frozen=True)
class _StartPayload:
    session: TransportSession
    metadata: Mapping[str, Any]
    expected_frames: int


def _disposition_bucket(disposition: str) -> str | None:
    upper = disposition.upper()
    if "DUPLICATE" in upper:
        return "synchronizer_duplicates"
    if "REORDER" in upper:
        return "synchronizer_reorders"
    if "GAP" in upper:
        return "synchronizer_gaps"
    if upper not in {"PENDING", "READY"}:
        return "synchronizer_other_drops"
    return None


class InferenceWorker:
    """Own all controller calls on one thread behind a capacity-one slot."""

    def __init__(
        self,
        stack: RuntimeStack,
        mirror: LatestOnlyKuksaMirror,
        metrics: SoakMetrics,
        *,
        device: str,
        epoch_ms: Callable[[], int] | None = None,
        ttl_ms: int = 250,
    ) -> None:
        self.stack = stack
        self.mirror = mirror
        self.metrics = metrics
        self.device = device
        self.epoch_ms = epoch_ms or (lambda: time.time_ns() // 1_000_000)
        self.c1 = _TimedC1(stack.c1, metrics)
        self.c2 = _TimedC2(stack.c2, metrics)
        self.c3 = _TimedC3(stack.c3, metrics)
        self.drive_quality = _TimedDriveQuality(stack.drive_quality, metrics)
        self.contextual_risk = _TimedRisk(stack.contextual_risk, metrics)
        self.handoff = LatestRuntimeSnapshotSlot()
        self.session = LiveInferenceSession(
            c1=self.c1,
            c2=self.c2,
            c3=self.c3,
            drive_quality=self.drive_quality,
            contextual_risk=self.contextual_risk,
        )
        self.controller = LiveInferenceController(
            self.session,
            handoff=self.handoff,
            coherent_dwell_ticks=2,
            max_skew_ms=25,
        )
        self.builder = LiveDecisionV2Builder(
            video_source=VIDEO_SOURCE,
            models=stack.models,
            ttl_ms=ttl_ms,
            clock_status="SYNCHRONIZED",
            clock_offset_uncertainty_ms=0,
        )
        self._clock_unhealthy = False
        self._condition = threading.Condition()
        self._pending: DecodedTick | None = None
        self._in_flight = False
        self._command: _ControlRequest | None = None
        self._active_session: TransportSession | None = None
        self._session_error: BaseException | None = None
        self._current_expected_frames = 0
        self._current_processed = 0
        self._current_started_ns: int | None = None
        self._session_summaries: list[dict[str, Any]] = []
        self._stopping = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="carsky-live-inference",
            daemon=True,
        )
        self._thread.start()

    @property
    def thread_name(self) -> str:
        return self._thread.name

    @property
    def session_summaries(self) -> list[dict[str, Any]]:
        with self._condition:
            return [dict(item) for item in self._session_summaries]

    @property
    def max_queue_depth(self) -> int:
        return int(self.metrics.count("ingress_queue_ever_nonempty") > 0)

    def _submit(
        self,
        operation: str,
        payload: Any = None,
        *,
        timeout_s: float = 120.0,
    ) -> Any:
        request = _ControlRequest(operation, payload, threading.Event())
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._command is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"inference {operation} command slot timed out")
                self._condition.wait(remaining)
            if self._closed:
                raise SoakError("inference worker is closed")
            self._command = request
            self._condition.notify_all()
        remaining = max(0.0, deadline - time.monotonic())
        if not request.done.wait(remaining):
            raise TimeoutError(f"inference {operation} timed out")
        if request.error is not None:
            raise request.error
        return request.result

    def start_session(
        self,
        session: TransportSession,
        metadata: Mapping[str, Any],
        *,
        expected_frames: int,
    ) -> None:
        self._submit(
            "start",
            _StartPayload(session, metadata, expected_frames),
        )

    def offer(self, tick: DecodedTick) -> None:
        if not isinstance(tick, DecodedTick):
            raise TypeError("inference worker accepts DecodedTick")
        with self._condition:
            if self._closed or self._stopping:
                raise SoakError("inference worker is closed")
            if self._active_session is None:
                raise SoakError("no inference source session is active")
            expected = self._active_session
            if (tick.session_id, tick.generation) != (
                expected.session_id,
                expected.generation,
            ):
                raise SoakError("decoded tick belongs to a stale/foreign session")
            if self._session_error is not None:
                raise SessionInferenceError(
                    f"active inference session faulted: {self._session_error}"
                )
            if self._pending is not None:
                self.metrics.increment("ingress_queue_drops")
                self.metrics.increment("input_drops")
                raise InferenceBackpressure(
                    "capacity-one inference queue is full; source session must end"
                )
            self._pending = tick
            self.metrics.increment("ingress_offered")
            self.metrics.increment("ingress_queue_ever_nonempty")
            self.metrics.maximum("max_ingress_queue_depth", 1)
            self._condition.notify_all()

    def wait_until_idle(self, timeout_s: float = 120.0) -> bool:
        timeout = _finite_non_negative(timeout_s, field="inference idle timeout")
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._pending is not None or self._in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return True

    def current_error(self) -> BaseException | None:
        with self._condition:
            return self._session_error

    def finish_session(self, *, timeout_s: float = 120.0) -> dict[str, Any]:
        if not self.wait_until_idle(timeout_s):
            raise TimeoutError("inference did not drain before session end")
        return self._submit("finish", timeout_s=timeout_s)

    def report_state_loss(
        self, reason: str = "INFERENCE_BACKLOG", *, timeout_s: float = 120.0
    ) -> None:
        if not self.wait_until_idle(timeout_s):
            raise TimeoutError("inference did not drain before state-loss reset")
        self._submit("state_loss", reason, timeout_s=timeout_s)

    def mark_clock_unhealthy(self, *, timeout_s: float = 120.0) -> None:
        self._submit("clock_unhealthy", timeout_s=timeout_s)

    def close(self, *, timeout_s: float = 120.0) -> None:
        if self._closed:
            return
        if not self.wait_until_idle(timeout_s):
            raise TimeoutError("inference did not drain before worker close")
        try:
            self._submit("stop", timeout_s=timeout_s)
        finally:
            self._thread.join(timeout_s)
            with self._condition:
                self._closed = True
        if self._thread.is_alive():
            raise TimeoutError("inference worker did not stop")

    def _publish_snapshot(
        self,
        snapshot: Any,
        *,
        tick: DecodedTick | None,
    ) -> None:
        if tick is None:
            received = self.epoch_ms()
            decided = received
            source_media_timestamp_ms = None
        else:
            received = tick.server_receive_timestamp_ms
            decided_raw = self.epoch_ms()
            if decided_raw < received:
                self.metrics.increment("clock_regressions")
                self._set_clock_unhealthy()
            decided = max(decided_raw, received)
            source_media_timestamp_ms = tick.source_media_timestamp_ms
        started = time.perf_counter_ns()
        envelope = self.builder.build(
            snapshot,
            server_receive_timestamp_ms=received,
            decision_timestamp_ms=decided,
            source_media_timestamp_ms=source_media_timestamp_ms,
        )
        self.metrics.observe(
            "decision_build_ms", (time.perf_counter_ns() - started) / 1_000_000.0
        )
        if tick is not None and snapshot.capture_timestamp_ms is not None:
            age = decided - snapshot.capture_timestamp_ms
            if age >= 0:
                self.metrics.observe("capture_to_decision_timestamp_ms", float(age))
            built_at = self.epoch_ms()
            if built_at < decided:
                self.metrics.increment("clock_regressions")
                if not self._clock_unhealthy:
                    # The just-built envelope still claimed a trusted clock.
                    # Suppress it and fence the generation; its reset heartbeat
                    # and every subsequent envelope are explicitly untrusted.
                    self._set_clock_unhealthy()
                    raise SoakError(
                        "wall clock regressed during decision construction; "
                        "trusted envelope suppressed"
                    )
            built_age = max(built_at, decided) - snapshot.capture_timestamp_ms
            if built_age >= 0:
                self.metrics.observe("capture_to_decision_built_ms", float(built_age))
        reference = PublisherFrameRef(
            session_id=snapshot.session_id,
            generation=snapshot.generation,
            sequence=snapshot.sequence,
            source_sequence=snapshot.source_sequence,
            capture_timestamp_ms=snapshot.capture_timestamp_ms,
            health=snapshot.health,
        )
        enqueue_started = time.perf_counter_ns()
        receipt = self.mirror.publish(reference, envelope)
        self.metrics.observe(
            "publisher_enqueue_ms",
            (time.perf_counter_ns() - enqueue_started) / 1_000_000.0,
        )
        self.metrics.increment("publisher_accepted")
        if receipt.coalesced_previous:
            self.metrics.increment("publisher_coalesced")
        if self.mirror.status.pending:
            self.metrics.maximum("max_publisher_queue_depth", 1)
        self.metrics.maximum("max_controller_handoff_depth", 1)
        if tick is None:
            self.metrics.increment("heartbeat_envelopes")
        else:
            self.metrics.increment("data_envelopes")

    def _start(self, payload: _StartPayload) -> None:
        if self._active_session is not None:
            raise SoakError("an inference source session is already active")
        if payload.expected_frames <= 0:
            raise ValueError("expected_frames must be positive")
        started = time.perf_counter_ns()
        self.c1.configure_session(
            payload.metadata, expected_frames=payload.expected_frames
        )
        self.c3.configure_session(
            payload.metadata, expected_frames=payload.expected_frames
        )
        self.controller.start_session(
            payload.session.session_id,
            generation=payload.session.generation,
        )
        self.builder.start_generation(
            payload.session.session_id,
            payload.session.generation,
        )
        self._active_session = payload.session
        self._session_error = None
        self._current_expected_frames = payload.expected_frames
        self._current_processed = 0
        self._current_started_ns = time.perf_counter_ns()
        heartbeat = self.handoff.pop_latest()
        if heartbeat is None:
            raise SoakError("controller did not emit its startup heartbeat")
        self._publish_snapshot(heartbeat, tick=None)
        self.metrics.increment("sessions_started")
        self.metrics.observe(
            "session_start_ms", (time.perf_counter_ns() - started) / 1_000_000.0
        )

    def _record_offer(self, result: Any) -> None:
        disposition = str(result.disposition)
        self.metrics.increment(f"synchronizer_disposition_{disposition.lower()}")
        bucket = _disposition_bucket(disposition)
        if bucket is not None:
            self.metrics.increment(bucket)
        if result.reset_required:
            raise SoakError(
                f"synchronizer requested reset: {result.disposition}"
            )

    def _process(self, tick: DecodedTick) -> None:
        session = self._active_session
        if session is None:
            raise SoakError("decoded tick reached inactive inference worker")
        if (tick.session_id, tick.generation) != (
            session.session_id,
            session.generation,
        ):
            raise SoakError("decoded tick identity changed before inference")
        self._record_offer(self.controller.offer_road(tick.road))
        self._record_offer(self.controller.offer_cabin(tick.cabin))
        self._record_offer(self.controller.offer_ego(tick.ego))
        self.metrics.maximum("max_controller_input_depth", 1)
        _cuda_synchronize(self.device)
        started = time.perf_counter_ns()
        prediction = self.controller.process_latest()
        _cuda_synchronize(self.device)
        self.metrics.observe(
            "controller_model_total_ms",
            (time.perf_counter_ns() - started) / 1_000_000.0,
        )
        if prediction is None:
            raise SoakError("controller consumed a tick without a prediction")
        snapshot = self.handoff.pop_latest()
        if snapshot is None or snapshot.source_sequence != tick.source_sequence:
            raise SoakError("controller output lost exact source-sequence mapping")
        self._current_processed += 1
        self.metrics.increment("inference_outputs")
        if (self._current_processed - 1) % 2 == 0:
            self.metrics.increment("c1_expected_model_updates")
        self._publish_snapshot(snapshot, tick=tick)

    def _publish_reset_heartbeat(self) -> None:
        heartbeat = self.handoff.pop_latest()
        if heartbeat is None:
            return
        self.builder.start_generation(heartbeat.session_id, heartbeat.generation)
        self._publish_snapshot(heartbeat, tick=None)
        self.metrics.increment("controller_resets")

    def _tick_failed(self, tick: DecodedTick, error: BaseException) -> None:
        self.metrics.error(
            "inference_tick",
            error,
            session_id=tick.session_id,
            source_sequence=tick.source_sequence,
        )
        self.metrics.increment("inference_exceptions")
        if _is_oom(error):
            self.metrics.increment("oom_exceptions")
        try:
            # Model exceptions advance generation inside process_latest.  An
            # adapter/synchronizer failure may happen earlier, so fence it now.
            if self.controller.generation == tick.generation:
                self.controller.report_state_loss("STATE_LOSS")
            self._publish_reset_heartbeat()
        except BaseException as reset_error:
            self.metrics.error(
                "inference_reset",
                reset_error,
                session_id=tick.session_id,
                source_sequence=tick.source_sequence,
            )
            self.metrics.increment("reset_exceptions")
        with self._condition:
            self._session_error = error

    def _finish(self) -> dict[str, Any]:
        active = self._active_session
        if active is None:
            raise SoakError("no inference source session is active")
        synchronizer = self.controller.synchronizer_stats
        controller_before = self.controller.controller_snapshot
        session_snapshot = self.controller.end_session()
        # ENDED is intentionally not published: the next recorded trip is a
        # distinct source session, and its startup heartbeat establishes the
        # next mirror identity. Drain it so it cannot overwrite later output.
        ended_heartbeat = self.handoff.pop_latest()
        elapsed_ms = (
            (time.perf_counter_ns() - self._current_started_ns) / 1_000_000.0
            if self._current_started_ns is not None
            else 0.0
        )
        summary = {
            "session_id": active.session_id,
            "generation_started": active.generation,
            "generation_ended": controller_before.generation,
            "expected_frames": self._current_expected_frames,
            "processed_frames": self._current_processed,
            "elapsed_ms": round(elapsed_ms, 6),
            "error": None
            if self._session_error is None
            else f"{type(self._session_error).__name__}: {self._session_error}"[:500],
            "session_snapshot": _json_value(session_snapshot),
            "controller_snapshot": _json_value(controller_before),
            "synchronizer_cumulative_stats": _json_value(synchronizer),
            "ended_heartbeat_drained": ended_heartbeat is not None,
        }
        with self._condition:
            self._session_summaries.append(summary)
            self._active_session = None
            self._session_error = None
            self._current_expected_frames = 0
            self._current_processed = 0
            self._current_started_ns = None
        self.metrics.increment("sessions_finished")
        return summary

    def _stop(self) -> None:
        if self._active_session is not None:
            self._finish()
        self.stack.close()
        self._stopping = True

    def _state_loss(self, reason: str) -> None:
        if self._active_session is None:
            raise SoakError("no inference source session is active")
        self.controller.report_state_loss(reason)
        self._publish_reset_heartbeat()
        self.metrics.increment("backpressure_state_loss_resets")

    def _set_clock_unhealthy(self) -> None:
        self.builder.set_clock_health(
            "UNHEALTHY", offset_uncertainty_ms=60_000
        )
        self._clock_unhealthy = True

    def _run_command(self, request: _ControlRequest) -> None:
        try:
            if request.operation == "start":
                self._start(request.payload)
            elif request.operation == "finish":
                request.result = self._finish()
            elif request.operation == "state_loss":
                self._state_loss(str(request.payload))
            elif request.operation == "clock_unhealthy":
                self._set_clock_unhealthy()
            elif request.operation == "stop":
                self._stop()
            else:
                raise SoakError(f"unknown inference command: {request.operation}")
        except BaseException as exc:
            request.error = exc
        finally:
            request.done.set()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._command is not None
                    or self._pending is not None
                    or self._stopping
                )
                if self._stopping and self._command is None and self._pending is None:
                    break
                if self._command is not None:
                    request = self._command
                    self._command = None
                    self._condition.notify_all()
                    tick = None
                else:
                    request = None
                    tick = self._pending
                    self._pending = None
                    self._in_flight = tick is not None
                    self._condition.notify_all()
            if request is not None:
                self._run_command(request)
                continue
            if tick is not None:
                try:
                    self._process(tick)
                except BaseException as exc:
                    self._tick_failed(tick, exc)
                finally:
                    with self._condition:
                        self._in_flight = False
                        self._condition.notify_all()


def _decode_image(payload: bytes, *, field: str) -> np.ndarray:
    if not isinstance(payload, bytes) or not payload:
        raise SoakError(f"{field} payload must be non-empty immutable bytes")
    encoded = np.frombuffer(payload, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if (
        decoded is None
        or decoded.dtype != np.uint8
        or decoded.ndim != 3
        or decoded.shape[2] != 3
        or decoded.shape[0] <= 0
        or decoded.shape[1] <= 0
    ):
        raise SoakError(f"{field} payload is not a decodable BGR image")
    if not decoded.flags.c_contiguous:
        decoded = np.ascontiguousarray(decoded)
    return decoded


class RecordedSoakAdapter:
    """Decode one sender session and hand exact identities to inference."""

    def __init__(
        self,
        worker: InferenceWorker,
        bundle: TruthFreeRecordedBundle,
        metrics: SoakMetrics,
        *,
        expected_frames: int,
        epoch_ms: Callable[[], int] | None = None,
        close_timeout_s: float = 120.0,
        drain_each_frame_for_test: bool = False,
    ) -> None:
        if not isinstance(bundle, TruthFreeRecordedBundle):
            raise TypeError("bundle must be TruthFreeRecordedBundle")
        if expected_frames <= 0 or expected_frames > len(bundle.frames):
            raise ValueError("expected_frames is outside the bundle")
        self.worker = worker
        self.bundle = bundle
        self.metrics = metrics
        self.expected_frames = expected_frames
        self.epoch_ms = epoch_ms or (lambda: time.time_ns() // 1_000_000)
        self.close_timeout_s = _positive_duration(
            close_timeout_s, field="adapter close timeout"
        )
        self.drain_each_frame_for_test = bool(drain_each_frame_for_test)
        self._session: TransportSession | None = None
        self._opened = False
        self._closed = False
        self._last_sequence: int | None = None
        self._last_capture_timestamp_ms: int | None = None
        self.frames_received = 0
        self.frames_decoded = 0
        self.frames_offered = 0
        self.session_summary: dict[str, Any] | None = None
        self.first_capture_timestamp_ms: int | None = None
        self.last_capture_timestamp_ms: int | None = None
        self._backpressure_state_loss = False

    def open(self, session: TransportSession) -> None:
        if self._opened or self._closed:
            raise SoakError("recorded soak adapter cannot be reopened")
        if session.source_hz != SOURCE_HZ:
            raise SoakError("soak adapter requires a 20 Hz source")
        if session.video_source != VIDEO_SOURCE or session.inference_mode != INFERENCE_MODE:
            raise SoakError("sender source/inference labels do not match live soak")
        self.worker.start_session(
            session,
            self.bundle.metadata,
            expected_frames=self.expected_frames,
        )
        if self.drain_each_frame_for_test:
            # Test-only deterministic drain; CLI never enables this path.
            self.worker.mirror.wait_until_idle(self.close_timeout_s)
        self._session = session
        self._opened = True

    def _validate_media(self, frame: TransportFrame, media: Any, *, stream: str) -> None:
        assert self._session is not None
        expected_media_id = (
            self._session.road_media_id
            if stream == "road"
            else self._session.cabin_media_id
        )
        if media.stream_id != stream or media.media_id != expected_media_id:
            raise SoakError(f"{stream} media identity does not match negotiated session")
        if media.source_sequence != frame.source_sequence:
            raise SoakError(f"{stream} source_sequence does not match frame metadata")
        if media.rtp_clock_rate_hz != RTP_CLOCK_RATE_HZ:
            raise SoakError(f"{stream} RTP clock must be 90 kHz")
        expected_rtp = (
            frame.source_media_timestamp_ms * RTP_CLOCK_RATE_HZ // 1000
        ) % RTP_MODULUS
        if media.rtp_timestamp != expected_rtp:
            raise SoakError(f"{stream} RTP timestamp is not bound to media timestamp")
        if media.content_type not in {"image/jpeg", "image/png"}:
            raise SoakError(f"{stream} codec is not an allow-listed image file")
        if hashlib.sha256(media.payload).hexdigest() != media.payload_sha256:
            raise SoakError(f"{stream} payload digest changed before decode")

    def _validate_frame(self, frame: TransportFrame) -> None:
        if not isinstance(frame, TransportFrame):
            raise TypeError("soak adapter accepts TransportFrame")
        if not self._opened or self._session is None:
            raise SoakError("soak adapter is not open")
        session = self._session
        if frame.schema != FRAME_SCHEMA:
            raise SoakError("recorded stream frame schema changed")
        if (frame.session_id, frame.generation) != (
            session.session_id,
            session.generation,
        ):
            raise SoakError("transport frame belongs to a stale/foreign session")
        if frame.video_source != VIDEO_SOURCE or frame.inference_mode != INFERENCE_MODE:
            raise SoakError("transport frame lacks RECORDED_STREAM + LIVE_MODEL labels")
        expected_sequence = 0 if self._last_sequence is None else self._last_sequence + 1
        if frame.source_sequence != expected_sequence:
            if self._last_sequence is not None and frame.source_sequence == self._last_sequence:
                self.metrics.increment("input_duplicates")
            elif frame.source_sequence < expected_sequence:
                self.metrics.increment("input_reorders")
            else:
                self.metrics.increment("input_gaps", frame.source_sequence - expected_sequence)
            raise SoakError(
                f"source_sequence discontinuity: got {frame.source_sequence}, "
                f"expected {expected_sequence}"
            )
        if (
            self._last_capture_timestamp_ms is not None
            and frame.capture_timestamp_ms <= self._last_capture_timestamp_ms
        ):
            self.metrics.increment("capture_timestamp_reorders")
            raise SoakError("capture timestamps must increase within a source session")
        self._validate_media(frame, frame.road, stream="road")
        self._validate_media(frame, frame.cabin, stream="cabin")

    def emit(self, frame: TransportFrame) -> None:
        self.frames_received += 1
        self.metrics.increment("source_ticks_attempted")
        try:
            self._validate_frame(frame)
            received_raw = self.epoch_ms()
            if received_raw < frame.capture_timestamp_ms:
                self.metrics.increment("clock_regressions")
                self.worker.mark_clock_unhealthy(timeout_s=self.close_timeout_s)
            received = max(received_raw, frame.capture_timestamp_ms)

            pair_started = time.perf_counter_ns()
            road_started = time.perf_counter_ns()
            road_bgr = _decode_image(frame.road.payload, field="road")
            self.metrics.observe(
                "road_decode_ms", (time.perf_counter_ns() - road_started) / 1_000_000.0
            )
            cabin_started = time.perf_counter_ns()
            cabin_bgr = _decode_image(frame.cabin.payload, field="cabin")
            self.metrics.observe(
                "cabin_decode_ms", (time.perf_counter_ns() - cabin_started) / 1_000_000.0
            )
            self.metrics.observe(
                "decode_pair_ms", (time.perf_counter_ns() - pair_started) / 1_000_000.0
            )

            envelope_started = time.perf_counter_ns()
            key = SourceTickKey(
                frame.session_id,
                frame.generation,
                frame.source_sequence,
            )
            road_identity = RtpFrameIdentity(frame.road.rtp_timestamp, key)
            cabin_identity = RtpFrameIdentity(frame.cabin.rtp_timestamp, key)
            road = RoadFrameEnvelope(
                road_identity,
                frame.capture_timestamp_ms,
                road_bgr,
            )
            cabin = CabinFrameEnvelope(
                cabin_identity,
                frame.capture_timestamp_ms,
                cabin_bgr,
            )
            ego = EgoTelemetryEnvelope(
                key,
                frame.capture_timestamp_ms,
                frame.ego,
            )
            self.metrics.observe(
                "envelope_copy_ms",
                (time.perf_counter_ns() - envelope_started) / 1_000_000.0,
            )
            timestamps = (
                road.capture_timestamp_ms,
                cabin.capture_timestamp_ms,
                ego.capture_timestamp_ms,
            )
            self.metrics.observe(
                "source_metadata_capture_skew_ms",
                float(max(timestamps) - min(timestamps)),
            )
            tick = DecodedTick(
                session_id=frame.session_id,
                generation=frame.generation,
                source_sequence=frame.source_sequence,
                source_media_timestamp_ms=frame.source_media_timestamp_ms,
                server_receive_timestamp_ms=received,
                road=road,
                cabin=cabin,
                ego=ego,
            )
            self.frames_decoded += 1
            self.metrics.increment("input_decoded_ticks")
            self.worker.offer(tick)
            self.frames_offered += 1
            self.metrics.increment("input_offered_ticks")
            self._last_sequence = frame.source_sequence
            self._last_capture_timestamp_ms = frame.capture_timestamp_ms
            if self.first_capture_timestamp_ms is None:
                self.first_capture_timestamp_ms = frame.capture_timestamp_ms
            self.last_capture_timestamp_ms = frame.capture_timestamp_ms
            if self.drain_each_frame_for_test and not self.worker.wait_until_idle(
                self.close_timeout_s
            ):
                raise TimeoutError("test inference drain timed out")
            if self.drain_each_frame_for_test and not self.worker.mirror.wait_until_idle(
                self.close_timeout_s
            ):
                raise TimeoutError("test publisher drain timed out")
            error = self.worker.current_error()
            if error is not None:
                raise SessionInferenceError(
                    f"inference failed for source sequence {frame.source_sequence}: {error}"
                )
        except BaseException as exc:
            if isinstance(exc, InferenceBackpressure):
                self._backpressure_state_loss = True
            if isinstance(exc, SoakError) and not isinstance(
                exc, (InferenceBackpressure, SessionInferenceError)
            ):
                self.metrics.increment("input_contract_errors")
            if "decod" in str(exc).lower():
                self.metrics.increment("decode_errors")
            self.metrics.error(
                "recorded_adapter_emit",
                exc,
                session_id=frame.session_id,
                source_sequence=frame.source_sequence,
            )
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._opened:
            return
        try:
            if self._backpressure_state_loss:
                self.worker.report_state_loss(
                    "INFERENCE_BACKLOG", timeout_s=self.close_timeout_s
                )
            self.session_summary = self.worker.finish_session(
                timeout_s=self.close_timeout_s
            )
        finally:
            self._opened = False


@dataclass(frozen=True)
class SoakRunResult:
    report: Mapping[str, Any]
    report_path: Path


def _bundle_report(bundle: TruthFreeRecordedBundle) -> dict[str, Any]:
    return {
        "trip_id": bundle.trip_id,
        "bundle_root": str(bundle.root),
        "manifest_path": str(bundle.manifest_path),
        "manifest_sha256": _sha256_file(bundle.manifest_path),
        "frames": len(bundle.frames),
        "metadata": _json_value(bundle.metadata),
    }


def _rate(count: int, seconds: float) -> float | None:
    if seconds <= 0.0:
        return None
    return round(float(count) / seconds, 6)


def _write_report(report: Mapping[str, Any], output_dir: Path) -> Path:
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = root / f"carsky-live-soak-{stamp}-{os.getpid()}.json"
    temporary = root / f".{destination.name}.tmp"
    payload = _strict_json_bytes(report)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def run_soak(
    bundle_paths: Sequence[Path | str],
    *,
    duration_seconds: float,
    device: str = "cuda",
    c1_checkpoint: Path | str = DEFAULT_C1_CHECKPOINT,
    c2_bundle: Path | str = DEFAULT_C2_BUNDLE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    publisher_delay_ms: float = 0.0,
    publisher_fail_every: int = 0,
    ttl_ms: int = 250,
    close_timeout_s: float = 120.0,
    clock: SenderClock | None = None,
    runtime_stack: RuntimeStack | None = None,
    drain_each_frame_for_test: bool = False,
) -> SoakRunResult:
    """Run bounded recorded-stream live inference and persist one strict report.

    ``runtime_stack`` and ``drain_each_frame_for_test`` exist solely to keep
    unit tests short and model-free.  The CLI always constructs the real C1,
    C2, C3, drive-quality, and contextual-risk stack and leaves the ingress
    asynchronous.
    """

    requested_seconds = _positive_duration(
        duration_seconds, field="duration_seconds"
    )
    delay_ms = _finite_non_negative(
        publisher_delay_ms, field="publisher_delay_ms"
    )
    fail_every = _non_negative_int(
        publisher_fail_every, field="publisher_fail_every"
    )
    if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or not 1 <= ttl_ms <= 1000:
        raise ValueError("ttl_ms must be an integer in [1, 1000]")
    timeout = _positive_duration(close_timeout_s, field="close_timeout_s")
    if not bundle_paths:
        raise ValueError("at least one prepared bundle is required")

    # This is the sole input-path read. The loader rejects raw/mixed trees and
    # verifies the exact prepare-tool allowlist, inventory, and every digest.
    bundles = tuple(TruthFreeRecordedBundle.load(path) for path in bundle_paths)
    if any(not bundle.frames for bundle in bundles):
        raise SoakError("every soak bundle must contain at least one source tick")
    sender_clock = clock or SystemClock()
    epoch_ms = lambda: sender_clock.time_ns() // 1_000_000
    metrics = SoakMetrics()
    overall_started_ns = time.perf_counter_ns()
    resources = ResourceSampler(device)
    model_load_started_ns = time.perf_counter_ns()
    injected_runtime = runtime_stack is not None
    stack = runtime_stack
    if stack is None:
        stack = build_real_runtime_stack(
            initial_metadata=bundles[0].metadata,
            c1_checkpoint=Path(c1_checkpoint),
            c2_bundle=Path(c2_bundle),
            device=device,
        )
    model_load_ms = (time.perf_counter_ns() - model_load_started_ns) / 1_000_000.0
    resources.sample("model_resident")
    model_resident_memory = {
        "process": _proc_memory_bytes(),
        "cuda": _cuda_memory(device),
    }
    _reset_cuda_peak(device)

    sink = LocalMockDecisionSink(
        metrics,
        delay_ms=delay_ms,
        fail_every=fail_every,
        epoch_ms=epoch_ms,
    )
    mirror = LatestOnlyKuksaMirror(
        sink,
        close_timeout_s=timeout,
        thread_name="carsky-live-local-publisher",
    )
    worker: InferenceWorker | None = None
    session_records: list[dict[str, Any]] = []
    transition_gaps_ms: list[int] = []
    target_frames = max(1, int(math.ceil(requested_seconds * SOURCE_HZ)))
    source_attempts = 0
    sender_emitted = 0
    sender_late_frames = 0
    sender_max_lateness_ms = 0.0
    fatal_error: BaseException | None = None
    source_started_ns = sender_clock.monotonic_ns()
    soak_started_perf_ns = time.perf_counter_ns()
    started_at = datetime.now(timezone.utc).isoformat()
    previous_capture_ms: int | None = None

    try:
        worker = InferenceWorker(
            stack,
            mirror,
            metrics,
            device=device,
            epoch_ms=epoch_ms,
            ttl_ms=ttl_ms,
        )
        session_index = 0
        while source_attempts < target_frames:
            # RecordedStreamSender paces absolutely within a trip. This outer
            # deadline preserves the missing 50 ms period across trip/session
            # transitions instead of emitting the next trip immediately.
            session_deadline_ns = source_started_ns + source_attempts * PERIOD_NS
            remaining_ns = session_deadline_ns - sender_clock.monotonic_ns()
            if remaining_ns > 0:
                sender_clock.sleep(remaining_ns / 1_000_000_000.0)

            bundle = bundles[session_index % len(bundles)]
            planned = min(target_frames - source_attempts, len(bundle.frames))
            session_id = (
                f"soak-{os.getpid()}-{session_index:05d}-{bundle.trip_id}"
            )
            adapter = RecordedSoakAdapter(
                worker,
                bundle,
                metrics,
                expected_frames=planned,
                epoch_ms=epoch_ms,
                close_timeout_s=timeout,
                drain_each_frame_for_test=drain_each_frame_for_test,
            )
            sender = RecordedStreamSender(
                bundle,
                adapter,
                session_id=session_id,
                generation=0,
                clock=sender_clock,
            )
            sender_stats: Any | None = None
            run_error: BaseException | None = None
            try:
                sender_stats = sender.run(limit=planned)
            except BaseException as exc:
                run_error = exc
                metrics.increment("sender_run_exceptions")
                metrics.error("recorded_sender", exc, session_id=session_id)

            attempted_this_session = adapter.frames_received
            source_attempts += attempted_this_session
            if sender_stats is not None:
                sender_emitted += sender_stats.frames_emitted
                sender_late_frames += sender_stats.late_frames
                sender_max_lateness_ms = max(
                    sender_max_lateness_ms, sender_stats.max_lateness_ms
                )
            if adapter.first_capture_timestamp_ms is not None:
                if previous_capture_ms is not None:
                    gap = adapter.first_capture_timestamp_ms - previous_capture_ms
                    transition_gaps_ms.append(gap)
                    metrics.observe("session_transition_gap_ms", float(max(0, gap)))
                    if gap != round(1000 / SOURCE_HZ):
                        metrics.increment("session_transition_gap_anomalies")
                previous_capture_ms = adapter.last_capture_timestamp_ms
            record = {
                "session_index": session_index,
                "trip_id": bundle.trip_id,
                "bundle_manifest_sha256": _sha256_file(bundle.manifest_path),
                "planned_frames": planned,
                "source_ticks_attempted": attempted_this_session,
                "decoded_frames": adapter.frames_decoded,
                "offered_frames": adapter.frames_offered,
                "first_capture_timestamp_ms": adapter.first_capture_timestamp_ms,
                "last_capture_timestamp_ms": adapter.last_capture_timestamp_ms,
                "sender": None if sender_stats is None else sender_stats.metadata(),
                "inference": adapter.session_summary,
                "error": None
                if run_error is None
                else f"{type(run_error).__name__}: {run_error}"[:500],
            }
            session_records.append(record)
            resources.sample(f"session_{session_index:05d}_end")
            session_index += 1

            if attempted_this_session == 0:
                fatal_error = run_error or SoakError(
                    "source session made no progress"
                )
                break
            if run_error is not None and not isinstance(
                run_error, InferenceBackpressure
            ):
                fatal_error = run_error
                break
            if adapter.session_summary is not None and adapter.session_summary["error"]:
                fatal_error = SessionInferenceError(
                    str(adapter.session_summary["error"])
                )
                break

        if fatal_error is None and source_attempts >= target_frames:
            requested_deadline_ns = source_started_ns + int(
                round(requested_seconds * 1_000_000_000.0)
            )
            remaining_ns = requested_deadline_ns - sender_clock.monotonic_ns()
            if remaining_ns > 0:
                sender_clock.sleep(remaining_ns / 1_000_000_000.0)
    finally:
        if worker is not None:
            try:
                worker.close(timeout_s=timeout)
            except BaseException as exc:
                metrics.error("inference_worker_close", exc)
                metrics.increment("close_exceptions")
                if fatal_error is None:
                    fatal_error = exc
        else:
            try:
                stack.close()
            except BaseException as exc:
                metrics.error("runtime_close", exc)
                metrics.increment("close_exceptions")
                if fatal_error is None:
                    fatal_error = exc
        mirror_closed = mirror.close(timeout_s=timeout)
        if not mirror_closed:
            error = TimeoutError("local publisher worker did not stop")
            metrics.error("publisher_close", error)
            metrics.increment("close_exceptions")
            if fatal_error is None:
                fatal_error = error

    ended_at = datetime.now(timezone.utc).isoformat()
    soak_wall_seconds = (time.perf_counter_ns() - soak_started_perf_ns) / 1_000_000_000.0
    overall_wall_seconds = (time.perf_counter_ns() - overall_started_ns) / 1_000_000_000.0
    source_clock_seconds = (
        sender_clock.monotonic_ns() - source_started_ns
    ) / 1_000_000_000.0
    counts, maxima, latency, errors = metrics.snapshot()
    mirror_summary = mirror.summary()
    sink_summary = sink.summary()
    publisher_errors = int(mirror_summary["errors"])
    processed = counts.get("inference_outputs", 0)
    cadence = {
        "source_ticks_processed": processed,
        "c1_calls": counts.get("c1_calls", 0),
        "c1_model_updates": counts.get("c1_model_updates", 0),
        "c1_expected_model_updates": counts.get("c1_expected_model_updates", 0),
        "c2_calls": counts.get("c2_calls", 0),
        "c3_calls": counts.get("c3_calls", 0),
        "drive_quality_calls": counts.get("drive_quality_calls", 0),
        "contextual_risk_calls": counts.get("contextual_risk_calls", 0),
    }
    cadence["exact"] = bool(
        cadence["c1_calls"] == processed
        and cadence["c1_model_updates"] == cadence["c1_expected_model_updates"]
        and cadence["c2_calls"] == processed
        and cadence["c3_calls"] == processed
        and cadence["drive_quality_calls"] == processed
        and cadence["contextual_risk_calls"] == processed
    )

    resource_report = resources.report(overall_wall_seconds)
    resident_rss = model_resident_memory["process"].get("rss")
    end_rss = resource_report["memory"].get("rss_end_bytes")
    peak_rss = resource_report["memory"].get("rss_max_sampled_bytes")
    resident_cuda = model_resident_memory["cuda"].get("allocated_bytes")
    end_cuda = resource_report["vram"]["end"].get("allocated_bytes")
    peak_cuda = resource_report["vram"].get("max_sampled_allocated_bytes")

    def delta(current: Any, baseline: Any) -> int | None:
        if isinstance(current, int) and isinstance(baseline, int):
            return current - baseline
        return None

    session_resource_samples = [
        sample
        for sample in resource_report["samples"]
        if str(sample.get("phase", "")).startswith("session_")
    ]
    post_warm_samples = session_resource_samples[1:]
    rss_growth: int | None = None
    cuda_allocated_growth: int | None = None
    cuda_reserved_growth: int | None = None
    rss_slope_bytes_per_minute: float | None = None
    cuda_allocated_slope_bytes_per_minute: float | None = None
    stability_status = "NOT_EVALUATED"
    stability_reason = "at least three completed source sessions are required"
    if len(session_resource_samples) >= 3:
        baseline = session_resource_samples[1]
        final = session_resource_samples[-1]
        required_fields_present = all(
            isinstance(sample.get(field), int)
            for sample in post_warm_samples
            for field in (
                "rss_bytes",
                "cuda_allocated_bytes",
                "cuda_reserved_bytes",
            )
        )
        if required_fields_present:
            rss_growth = int(final["rss_bytes"]) - int(baseline["rss_bytes"])
            cuda_allocated_growth = int(final["cuda_allocated_bytes"]) - int(
                baseline["cuda_allocated_bytes"]
            )
            cuda_reserved_growth = int(final["cuda_reserved_bytes"]) - int(
                baseline["cuda_reserved_bytes"]
            )
            elapsed_minutes = (
                float(final["elapsed_seconds"])
                - float(baseline["elapsed_seconds"])
            ) / 60.0
            if elapsed_minutes > 0.0:
                rss_slope_bytes_per_minute = round(
                    rss_growth / elapsed_minutes, 3
                )
                cuda_allocated_slope_bytes_per_minute = round(
                    cuda_allocated_growth / elapsed_minutes, 3
                )
            stability_status = (
                "PASS"
                if rss_growth <= 128 * 1024 * 1024
                and cuda_allocated_growth <= 64 * 1024 * 1024
                and cuda_reserved_growth <= 64 * 1024 * 1024
                else "FAIL"
            )
            stability_reason = "post-warm end growth evaluated"
        else:
            stability_reason = (
                "RSS and PyTorch CUDA allocated/reserved values are required "
                "for every post-warm session sample"
            )
    resource_report["stability"] = {
        "status": stability_status,
        "reason": stability_reason,
        "threshold_scope": "HARNESS_LOCAL_POC_NOT_PRODUCTION_SLA",
        "warmup_sessions_excluded": 2,
        "thresholds": {
            "rss_end_growth_bytes_at_most": 128 * 1024 * 1024,
            "pytorch_cuda_allocated_end_growth_bytes_at_most": 64 * 1024 * 1024,
            "pytorch_cuda_reserved_end_growth_bytes_at_most": 64 * 1024 * 1024,
        },
        "observed": {
            "completed_session_samples": len(session_resource_samples),
            "post_warm_samples": len(post_warm_samples),
            "rss_end_growth_bytes": rss_growth,
            "pytorch_cuda_allocated_end_growth_bytes": cuda_allocated_growth,
            "pytorch_cuda_reserved_end_growth_bytes": cuda_reserved_growth,
            "rss_slope_bytes_per_minute": rss_slope_bytes_per_minute,
            "pytorch_cuda_allocated_slope_bytes_per_minute": (
                cuda_allocated_slope_bytes_per_minute
            ),
            "rss_end_minus_model_resident_bytes": delta(end_rss, resident_rss),
            "rss_peak_minus_model_resident_bytes": delta(peak_rss, resident_rss),
            "pytorch_cuda_end_minus_model_resident_bytes": delta(
                end_cuda, resident_cuda
            ),
            "pytorch_cuda_peak_minus_model_resident_bytes": delta(
                peak_cuda, resident_cuda
            ),
        },
    }
    cuda_end = resource_report["vram"]["end"]
    cuda_device_name = cuda_end.get("device_name")
    hardware_status = (
        "NOT_EVALUATED_TEST_RUNTIME"
        if injected_runtime
        else "PASS"
        if cuda_end.get("available") is True
        and isinstance(cuda_device_name, str)
        and "t4" in cuda_device_name.lower()
        else "FAIL"
    )
    resource_report["cpu"]["measurement_scope"] = (
        "MODEL_LOAD_PLUS_SOAK_PUBLISHER_DRAIN_AND_SHUTDOWN"
    )
    pytorch_cuda_allocator = resource_report.pop("vram")
    pytorch_cuda_allocator["measurement_scope"] = (
        "PYTORCH_CUDA_ALLOCATOR_ONLY_NOT_TOTAL_PROCESS_OR_DEVICE_VRAM"
    )
    resource_report["pytorch_cuda_allocator"] = pytorch_cuda_allocator

    inference_drop_count = max(0, source_attempts - processed)
    drop_rate_pct = (
        100.0 * inference_drop_count / source_attempts
        if source_attempts > 0
        else 100.0
    )
    decision_p95 = latency.get("capture_to_decision_built_ms", {}).get("p95")
    queue_maxima = {
        "ingress_waiting": int(maxima.get("max_ingress_queue_depth", 0)),
        "controller_input": int(maxima.get("max_controller_input_depth", 0)),
        "controller_handoff": int(maxima.get("max_controller_handoff_depth", 0)),
        "publisher_pending": int(maxima.get("max_publisher_queue_depth", 0)),
    }
    functional_checks = {
        "target_source_ticks_completed": source_attempts == target_frames,
        "drop_rate_below_1_percent": drop_rate_pct < 1.0,
        "capture_to_decision_p95_at_most_150_ms": (
            decision_p95 is not None and float(decision_p95) <= 150.0
        ),
        "wall_at_least_requested": source_clock_seconds + 1e-6 >= requested_seconds,
        "all_queue_depths_at_most_one": all(
            value <= 1 for value in queue_maxima.values()
        ),
        "exact_model_cadence": bool(cadence["exact"]),
        "no_oom": counts.get("oom_exceptions", 0) == 0,
        "no_clock_regression": counts.get("clock_regressions", 0) == 0,
        "publisher_identity_exact": counts.get("publisher_identity_errors", 0) == 0,
    }
    functional_pass = all(functional_checks.values())
    qualification_required = not injected_runtime
    qualification_checks: dict[str, bool | None] = {
        "cuda_device_is_t4": (
            None if injected_runtime else hardware_status == "PASS"
        ),
        "post_warm_memory_growth_within_poc_thresholds": (
            None
            if injected_runtime or stability_status == "NOT_EVALUATED"
            else stability_status == "PASS"
        ),
    }
    qualification_pass = (
        True
        if not qualification_required
        else all(value is True for value in qualification_checks.values())
    )
    acceptance_pass = functional_pass and qualification_pass
    hard_failure = any(
        (
            fatal_error is not None,
            counts.get("input_contract_errors", 0) > 0,
            counts.get("inference_exceptions", 0) > 0,
            counts.get("close_exceptions", 0) > 0,
            not functional_pass,
            stability_status == "FAIL" and qualification_required,
        )
    )
    degraded = any(
        (
            publisher_errors > 0,
            int(mirror_summary["coalesced"]) > 0,
            counts.get("session_transition_gap_anomalies", 0) > 0,
        )
    )
    qualification_incomplete = qualification_required and not qualification_pass
    outcome = (
        "FAIL"
        if hard_failure
        else "INCONCLUSIVE"
        if qualification_incomplete
        else "DEGRADED"
        if degraded
        else "PASS"
    )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generated_at": ended_at,
        "outcome": outcome,
        "claim_boundary": {
            "video_source": VIDEO_SOURCE,
            "inference_mode": INFERENCE_MODE,
            "media_codec": MEDIA_CODEC,
            "h264_included": H264_INCLUDED,
            "aws_connectivity": AWS_CONNECTIVITY,
            "transport": "LOCAL_IN_PROCESS_EXACT_RTP_METADATA",
            "image_decode": "cv2.imdecode(IMREAD_COLOR)",
            "synthetic_frames": False,
            "synthetic_ego": False,
            "raw_or_mixed_dataset_access": False,
            "ground_truth_depth_labels_events_targets_predictions": False,
            "pair_skew_semantics": (
                "SOURCE_METADATA_CAPTURE_SKEW_ZERO_BY_ATOMIC_SENDER_CONSTRUCTION"
            ),
            "independent_webrtc_decode_arrival_skew_measured": False,
            "clock_claim": "SAME_HOST_SYNCHRONIZED_CLOCK_ONLY",
        },
        "configuration": {
            "requested_duration_seconds": requested_seconds,
            "source_hz": SOURCE_HZ,
            "target_source_ticks": target_frames,
            "device": device,
            "ingress_queue_capacity": INGRESS_CAPACITY,
            "ingress_queue_policy": "REJECT_AND_END_SOURCE_SESSION_WHEN_FULL",
            "publisher_queue_policy": "LATEST_ONLY_CAPACITY_ONE_NO_RETRY",
            "publisher_delay_ms": delay_ms,
            "publisher_fail_every": fail_every,
            "decision_ttl_ms": ttl_ms,
            "runtime_injected_for_test": injected_runtime,
            "test_drain_each_frame": bool(drain_each_frame_for_test),
            "started_at": started_at,
            "ended_at": ended_at,
        },
        "bundles": [_bundle_report(bundle) for bundle in bundles],
        "models": {
            "identities": _json_value(stack.models),
            "loaded_once": True,
            "load_performed_by_harness": not injected_runtime,
            "load_ms": round(model_load_ms, 6),
            "resident_memory_after_load": model_resident_memory,
            "resident_cuda_measurement_scope": (
                "PYTORCH_CUDA_ALLOCATOR_ONLY_NOT_TOTAL_PROCESS_OR_DEVICE_VRAM"
            ),
            "c2_manifest_artifacts_verified_before_load": not injected_runtime,
        },
        "timing": {
            "requested_duration_seconds": requested_seconds,
            "source_clock_elapsed_seconds": round(source_clock_seconds, 6),
            "soak_wall_seconds": round(soak_wall_seconds, 6),
            "overall_with_model_load_seconds": round(overall_wall_seconds, 6),
            "wall_requirement_met": source_clock_seconds + 1e-6 >= requested_seconds,
            "session_transition_gaps_ms": transition_gaps_ms,
            "expected_transition_gap_ms": round(1000 / SOURCE_HZ),
        },
        "throughput": {
            "source_ticks_attempted": source_attempts,
            "sender_frames_emitted_completed_runs_only": sender_emitted,
            "decoded_ticks": counts.get("input_decoded_ticks", 0),
            "offered_ticks": counts.get("input_offered_ticks", 0),
            "inference_outputs": processed,
            "data_envelopes_accepted": counts.get("data_envelopes", 0),
            "publisher_data_attempts": sink_summary["data_attempts"],
            "publisher_successful_total": sink_summary["successful"],
            "input_fps_wall": _rate(counts.get("input_decoded_ticks", 0), soak_wall_seconds),
            "output_fps_wall": _rate(processed, soak_wall_seconds),
            "publisher_data_fps_wall": _rate(
                int(sink_summary["data_attempts"]), soak_wall_seconds
            ),
            "sender_late_frames_successful_sessions": sender_late_frames,
            "sender_max_lateness_ms_successful_sessions": round(
                sender_max_lateness_ms, 6
            ),
        },
        "latency_ms": latency,
        "latency_telemetry": {
            "method": "EXACT_COUNT_MEAN_MAX_PLUS_BOUNDED_DETERMINISTIC_RESERVOIR",
            "reservoir_per_series": MAX_LATENCY_RESERVOIR,
            "unbounded_per_tick_lists": False,
            "capture_to_decision_acceptance_series": "capture_to_decision_built_ms",
            "decision_contract_timestamp_series": "capture_to_decision_timestamp_ms",
        },
        "acceptance": {
            "passed": acceptance_pass,
            "status": (
                "PASS"
                if acceptance_pass
                else "NOT_EVALUATED"
                if qualification_incomplete and functional_pass
                else "FAIL"
            ),
            "thresholds": {
                "drop_rate_pct_strictly_below": 1.0,
                "capture_to_decision_p95_ms_at_most": 150.0,
                "minimum_wall_seconds": requested_seconds,
                "maximum_queue_depth": 1,
                "exact_cadence_required": True,
                "oom_allowed": 0,
                "hardware": "CUDA Tesla T4",
                "memory_growth": resource_report["stability"]["thresholds"],
            },
            "observed": {
                "inference_drop_count": inference_drop_count,
                "drop_rate_pct": round(drop_rate_pct, 6),
                "capture_to_decision_p95_ms": decision_p95,
                "source_clock_elapsed_seconds": round(source_clock_seconds, 6),
                "queue_maxima": queue_maxima,
                "exact_cadence": cadence["exact"],
                "oom_count": counts.get("oom_exceptions", 0),
                "hardware_status": hardware_status,
                "cuda_device_name": cuda_device_name,
                "memory_stability_status": stability_status,
            },
            "functional_checks": functional_checks,
            "qualification": {
                "required": qualification_required,
                "passed": qualification_pass,
                "checks": qualification_checks,
            },
        },
        "integrity": {
            "input_drops": counts.get("input_drops", 0),
            "ingress_queue_drops": counts.get("ingress_queue_drops", 0),
            "input_duplicates": counts.get("input_duplicates", 0),
            "input_reorders": counts.get("input_reorders", 0),
            "input_gaps": counts.get("input_gaps", 0),
            "capture_timestamp_reorders": counts.get("capture_timestamp_reorders", 0),
            "synchronizer_duplicates": counts.get("synchronizer_duplicates", 0),
            "synchronizer_reorders": counts.get("synchronizer_reorders", 0),
            "synchronizer_gaps": counts.get("synchronizer_gaps", 0),
            "synchronizer_other_drops": counts.get("synchronizer_other_drops", 0),
            "publisher_duplicates": sink_summary["duplicates"],
            "publisher_reorders": sink_summary["reorders"],
            "publisher_sequence_gaps_from_latest_only": sink_summary[
                "sequence_gaps_from_coalescing"
            ],
            "controller_fault_resets": counts.get("controller_resets", 0),
            "backpressure_state_loss_resets": counts.get(
                "backpressure_state_loss_resets", 0
            ),
            "backpressure_policy": (
                "DRAIN_PENDING_THEN_CONTROLLER_STATE_LOSS_HEARTBEAT_THEN_END_SESSION"
            ),
            "normal_session_state_resets": counts.get("sessions_started", 0),
            "oom_exceptions": counts.get("oom_exceptions", 0),
            "inference_exceptions": counts.get("inference_exceptions", 0),
            "clock_regressions": counts.get("clock_regressions", 0),
            "publisher_identity_errors": counts.get(
                "publisher_identity_errors", 0
            ),
        },
        "cadence": cadence,
        "queues": {
            "ingress_capacity": INGRESS_CAPACITY,
            "max_ingress_waiting": int(maxima.get("max_ingress_queue_depth", 0)),
            "max_controller_input": int(maxima.get("max_controller_input_depth", 0)),
            "max_controller_handoff": int(
                maxima.get("max_controller_handoff_depth", 0)
            ),
            "publisher_capacity": 1,
            "max_publisher_pending": int(
                maxima.get("max_publisher_queue_depth", 0)
            ),
        },
        "publisher": {
            "mirror": mirror_summary,
            "sink": sink_summary,
            "max_serialized_published_envelope_bytes": int(
                maxima.get("max_envelope_bytes", 0)
            ),
            "heartbeat_envelopes_accepted_total": counts.get(
                "heartbeat_envelopes", 0
            ),
            "startup_heartbeats_accepted": max(
                0,
                counts.get("heartbeat_envelopes", 0)
                - counts.get("controller_resets", 0),
            ),
            "reset_heartbeats": counts.get("controller_resets", 0),
            "startup_or_reset_may_be_coalesced": True,
            "ordering_guard_accepts_first_observed_sequence_per_generation": True,
        },
        "resources": resource_report,
        "sessions": session_records,
        "exceptions": {
            "retained_count": len(errors),
            "omitted_count": counts.get("metric_error_items_omitted", 0),
            "oom_count": counts.get("oom_exceptions", 0),
            "items": errors,
            "fatal": None
            if fatal_error is None
            else f"{type(fatal_error).__name__}: {fatal_error}"[:500],
        },
        "limitations": [
            "JPEG/PNG file decode is measured; H.264 encode, packetization, network, and decode are not included.",
            "The publisher is a local mock sink; no AWS, MQTT, KUKSA server, or Android network hop is included.",
            "A prepared trip boundary starts a new inference session while keeping model weights resident.",
            "Latest-only publication intentionally drops superseded observability snapshots and never retries stale state.",
            "Source metadata capture skew is zero by construction; independent WebRTC arrival/decode skew is unavailable.",
            "CUDA memory values come from the PyTorch allocator, not total device/process VRAM.",
            "The decision-v2 contract timestamp precedes builder execution; capture_to_decision_built_ms is the complete local construction latency used for acceptance.",
        ],
    }
    report_path = _write_report(report, Path(output_dir))
    return SoakRunResult(report=MappingProxyType(report), report_path=report_path)


def _non_negative_cli_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run truth-free C1+C2+C3 live inference from prepared CarSky "
            "JPEG/PNG bundles and publish decision-v2 to a local mock sink. "
            "No AWS call is made and H.264 is not included."
        )
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        required=True,
        help="requested paced source duration; e.g. 30 for smoke or 600 for soak",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        type=Path,
        dest="bundles",
        help=(
            "prepared truth-free bundle (repeat to control alternation); "
            "defaults to ignored T01-Sample and T02-Sample bundles"
        ),
    )
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    parser.add_argument(
        "--c1-checkpoint", type=Path, default=DEFAULT_C1_CHECKPOINT
    )
    parser.add_argument("--c2-bundle", type=Path, default=DEFAULT_C2_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--publisher-delay-ms",
        type=float,
        default=0.0,
        help="local mock broker delay per serialized snapshot",
    )
    parser.add_argument(
        "--publisher-fail-every",
        type=_non_negative_cli_int,
        default=0,
        help="simulate a local broker outage on every Nth call (0 disables)",
    )
    parser.add_argument("--ttl-ms", type=int, default=250)
    parser.add_argument("--close-timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    bundles = tuple(arguments.bundles or DEFAULT_BUNDLES)
    try:
        result = run_soak(
            bundles,
            duration_seconds=arguments.duration_seconds,
            device=arguments.device,
            c1_checkpoint=arguments.c1_checkpoint,
            c2_bundle=arguments.c2_bundle,
            output_dir=arguments.output_dir,
            publisher_delay_ms=arguments.publisher_delay_ms,
            publisher_fail_every=arguments.publisher_fail_every,
            ttl_ms=arguments.ttl_ms,
            close_timeout_s=arguments.close_timeout_seconds,
        )
    except (OSError, ValueError, SoakError) as exc:
        print(f"carsky live soak failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    outcome = str(result.report["outcome"])
    print(
        json.dumps(
            {
                "outcome": outcome,
                "report": str(result.report_path),
                "media_codec": MEDIA_CODEC,
                "h264_included": H264_INCLUDED,
                "aws_connectivity": AWS_CONNECTIVITY,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
