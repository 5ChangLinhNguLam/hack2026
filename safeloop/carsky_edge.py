"""CarSky edge entry point for unified C1/C2/C3 replay.

Each model result is converted once into the versioned Android decision
envelope.  The exact same envelope is then sent to the AAOS dashboard over
room-local UDP and mirrored, together with standard VSS signals, to KUKSA.
Both transports are optional so the full contract can be smoke-tested on a
developer machine without a running CarSky room.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from tripkit import TripLoader

from . import replay_models
from .carsky_decision import (
    DEFAULT_TTL_MS,
    DecisionEnvelope,
    DecisionEnvelopeBuilder,
    DecisionValidity,
)
from .carsky_hmi import HmiTransportError, UdpDecisionPublisher
from .carsky_kuksa import (
    KuksaGrpcBackend,
    KuksaPublishError,
    KuksaSignalPublisher,
)


DEFAULT_ANDROID_HOST = "10.99.0.14"
DEFAULT_ANDROID_PORT = 48_100
DEFAULT_KUKSA_HOST = "127.0.0.10"
DEFAULT_KUKSA_PORT = 55_555
MODEL_VERSION_SHA256_PREFIX_HEX = 12


def _sha256_artifact(path: Path) -> tuple[str, str]:
    """Hash one file or a directory tree using an auditable stable framing."""

    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"model provenance refuses symlink: {path}")
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), "file-sha256"
    if not path.is_dir():
        raise FileNotFoundError(f"model provenance source does not exist: {path}")

    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"model provenance directory is empty: {path}")
    digest = hashlib.sha256(b"safeloop-tree-sha256-v1\0")
    for item in files:
        if item.is_symlink():
            raise ValueError(f"model provenance refuses symlink: {item}")
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), "safeloop-tree-sha256-v1"


def build_model_provenance(
    c1_checkpoint: Path,
    c2_bundle: Path,
) -> tuple[dict[str, str], dict[str, object]]:
    """Build compact wire IDs plus full hashes for the local audit report."""

    package_dir = Path(__file__).resolve().parent
    sources = {
        "c1": Path(c1_checkpoint),
        "c2": Path(c2_bundle),
        "c3": package_dir / "c3.py",
        "dq": package_dir / "drive_quality.py",
        "risk": package_dir / "contextual_risk.py",
    }
    artifacts: dict[str, dict[str, str]] = {}
    wire: dict[str, str] = {}
    for name, source in sources.items():
        full_digest, digest_kind = _sha256_artifact(source)
        wire[name] = full_digest[:MODEL_VERSION_SHA256_PREFIX_HEX]
        artifacts[name] = {
            "source": str(source),
            "digest_kind": digest_kind,
            "sha256": full_digest,
            "wire_id": wire[name],
        }
    return wire, {
        "wire_contract": "health.model_versions",
        "wire_id_encoding": (
            f"first-{MODEL_VERSION_SHA256_PREFIX_HEX}-hex-of-sha256"
        ),
        "artifacts": artifacts,
    }


class UdpPublisher(Protocol):
    def publish(self, envelope: DecisionEnvelope) -> Any: ...

    def close(self) -> None: ...


class KuksaPublisher(Protocol):
    def publish(self, frame: Any, envelope: DecisionEnvelope) -> Any: ...

    def close(self) -> None: ...


class EnvelopeRecordError(RuntimeError):
    """A strict decision JSONL artifact could not be recorded atomically."""


@dataclass(frozen=True)
class EnvelopeRecordReceipt:
    session_id: str
    sequence: int
    bytes_written: int


class AtomicEnvelopeJsonlRecorder:
    """Write one strict decision packet per line and publish on clean close."""

    def __init__(self, destination: str | Path) -> None:
        self.destination = Path(destination)
        if self.destination.exists() and self.destination.is_dir():
            raise ValueError("record-envelopes destination must be a file")
        self.temporary = self.destination.with_name(
            self.destination.name + ".tmp"
        )
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._handle = self.temporary.open("w", encoding="utf-8")
        except OSError as exc:
            raise EnvelopeRecordError(
                f"cannot open envelope recording {self.temporary}"
            ) from exc
        self._closed = False
        self._last_session_id: str | None = None
        self._last_sequence: int | None = None
        self.records = 0

    def publish(self, envelope: DecisionEnvelope) -> EnvelopeRecordReceipt:
        if self._closed:
            raise EnvelopeRecordError("envelope recorder is closed")
        if not isinstance(envelope, DecisionEnvelope):
            raise TypeError("recorder requires a DecisionEnvelope")
        if self._last_session_id == envelope.session_id:
            if (
                self._last_sequence is not None
                and envelope.sequence <= self._last_sequence
            ):
                raise EnvelopeRecordError(
                    "refusing duplicate/out-of-order envelope sequence"
                )
        elif self._last_session_id is not None or envelope.sequence != 0:
            raise EnvelopeRecordError(
                "one JSONL recording must contain exactly one ordered session"
            )
        payload = envelope.to_json_bytes()
        try:
            text = payload.decode("utf-8")
            self._handle.write(text)
            self._handle.write("\n")
            self._handle.flush()
        except (OSError, UnicodeDecodeError) as exc:
            raise EnvelopeRecordError("cannot write decision JSONL") from exc
        self._last_session_id = envelope.session_id
        self._last_sequence = envelope.sequence
        self.records += 1
        return EnvelopeRecordReceipt(
            session_id=envelope.session_id,
            sequence=envelope.sequence,
            bytes_written=len(payload) + 1,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except OSError as exc:
            self._handle.close()
            self._closed = True
            self.temporary.unlink(missing_ok=True)
            raise EnvelopeRecordError(
                f"cannot finalize envelope recording {self.destination}"
            ) from exc
        else:
            self._handle.close()
            self._closed = True
        try:
            self.temporary.replace(self.destination)
        except OSError as exc:
            self.temporary.unlink(missing_ok=True)
            raise EnvelopeRecordError(
                f"cannot finalize envelope recording {self.destination}"
            ) from exc

    def abort(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True
        self.temporary.unlink(missing_ok=True)

    def __enter__(self) -> "AtomicEnvelopeJsonlRecorder":
        return self

    def __exit__(
        self, exc_type: object, _exc: object, _traceback: object
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


@dataclass(frozen=True)
class EdgeFrameReceipt:
    session_id: str
    sequence: int
    udp_sent: bool
    kuksa_sent: bool
    transport_errors: tuple[str, ...]


class CarSkyEdgeSink:
    """Synchronous latest-frame fan-out with independent transport branches."""

    def __init__(
        self,
        builder: DecisionEnvelopeBuilder,
        *,
        udp_publisher: UdpPublisher | None = None,
        kuksa_publisher: KuksaPublisher | None = None,
        envelope_recorder: AtomicEnvelopeJsonlRecorder | None = None,
        validity_provider: Callable[[Any], DecisionValidity] | None = None,
        transport_error_handler: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(builder, DecisionEnvelopeBuilder):
            raise TypeError("builder must be a DecisionEnvelopeBuilder")
        self.builder = builder
        self.udp_publisher = udp_publisher
        self.kuksa_publisher = kuksa_publisher
        self.envelope_recorder = envelope_recorder
        self._validity_provider = validity_provider or self._replay_validity
        self._transport_error_handler = transport_error_handler
        self._closed = False
        self._frames = 0
        self._udp_sent = 0
        self._kuksa_sent = 0
        self._udp_errors = 0
        self._kuksa_errors = 0
        self._recorded = 0
        self._last_errors: dict[str, str] = {}
        self._latest: DecisionEnvelope | None = None

    @property
    def latest(self) -> DecisionEnvelope | None:
        return self._latest

    def _replay_validity(self, frame: Any) -> DecisionValidity:
        """Treat completed same-frame inference as fresh, including C1 holds.

        C1 is intentionally forward-filled between model updates.  A held
        result remains valid only while its derived age is below the envelope
        TTL; contextual risk is suppressed automatically once C1 goes stale.
        """

        frame_id = frame.frame_id
        model_frame_id = frame.c1.model_frame_id
        try:
            c1_age_ms = round(
                (frame_id - model_frame_id)
                * 1_000.0
                / self.builder.source_fps
            )
        except (TypeError, ValueError, ZeroDivisionError):
            c1_age_ms = self.builder.ttl_ms
        c1_fresh = 0 <= c1_age_ms < self.builder.ttl_ms
        return DecisionValidity(
            ego=True,
            front_camera=True,
            driver_camera=True,
            c1=c1_fresh,
            c2=True,
            c3=True,
            drive_quality=True,
            contextual_risk=c1_fresh,
        )

    def _record_transport_error(self, branch: str, error: Exception) -> str:
        message = f"{branch}: {type(error).__name__}: {error}"
        self._last_errors[branch] = message
        if self._transport_error_handler is not None:
            self._transport_error_handler(message)
        return message

    def publish(self, frame: Any) -> EdgeFrameReceipt:
        if self._closed:
            raise RuntimeError("CarSky edge sink is closed")
        validity = self._validity_provider(frame)
        if not isinstance(validity, DecisionValidity):
            raise TypeError("validity_provider must return DecisionValidity")
        envelope = self.builder.build(frame, validity=validity)
        errors: list[str] = []
        udp_sent = False
        kuksa_sent = False

        # The direct HMI branch is latency-critical and always runs first.
        if self.udp_publisher is not None:
            try:
                self.udp_publisher.publish(envelope)
                self._udp_sent += 1
                udp_sent = True
            except HmiTransportError as exc:
                self._udp_errors += 1
                errors.append(self._record_transport_error("udp", exc))

        # KUKSA is an independent observability/VSS mirror. A transient write
        # failure must not stop inference or prevent the next UDP snapshot.
        if self.kuksa_publisher is not None:
            try:
                self.kuksa_publisher.publish(frame, envelope)
                self._kuksa_sent += 1
                kuksa_sent = True
            except KuksaPublishError as exc:
                self._kuksa_errors += 1
                errors.append(self._record_transport_error("kuksa", exc))

        if self.envelope_recorder is not None:
            self.envelope_recorder.publish(envelope)
            self._recorded += 1

        self._frames += 1
        self._latest = envelope
        return EdgeFrameReceipt(
            session_id=envelope.session_id,
            sequence=envelope.sequence,
            udp_sent=udp_sent,
            kuksa_sent=kuksa_sent,
            transport_errors=tuple(errors),
        )

    def __call__(self, frame: Any) -> None:
        self.publish(frame)

    def summary(self) -> dict[str, object]:
        live_transport_enabled = (
            self.udp_publisher is not None or self.kuksa_publisher is not None
        )
        live_transport_errors = self._udp_errors + self._kuksa_errors
        live_transport_complete = (
            self._frames > 0
            and live_transport_errors == 0
            and (
                self.udp_publisher is None
                or self._udp_sent == self._frames
            )
            and (
                self.kuksa_publisher is None
                or self._kuksa_sent == self._frames
            )
        )
        return {
            "session_id": self.builder.session_id,
            "source_mode": self.builder.source_mode,
            "frames_enveloped": self._frames,
            "udp_enabled": self.udp_publisher is not None,
            "udp_sent": self._udp_sent,
            "udp_errors": self._udp_errors,
            "kuksa_enabled": self.kuksa_publisher is not None,
            "kuksa_sent": self._kuksa_sent,
            "kuksa_errors": self._kuksa_errors,
            "recording_enabled": self.envelope_recorder is not None,
            "recorded_envelopes": self._recorded,
            "recording_path": (
                str(self.envelope_recorder.destination)
                if self.envelope_recorder is not None
                else None
            ),
            "last_transport_errors": dict(self._last_errors),
            "live_transport_enabled": live_transport_enabled,
            "live_transport_pass": (
                live_transport_complete
                if live_transport_enabled
                else None
            ),
            "ttl_ms": self.builder.ttl_ms,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        for publisher in (
            self.envelope_recorder,
            self.kuksa_publisher,
            self.udp_publisher,
        ):
            if publisher is None:
                continue
            try:
                publisher.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                errors.append(exc)
        if errors:
            raise RuntimeError(
                "cannot close all CarSky transports: "
                + "; ".join(str(error) for error in errors)
            ) from errors[0]

    def abort(self) -> None:
        """Close transports but keep the previous complete recording intact."""

        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        if self.envelope_recorder is not None:
            try:
                self.envelope_recorder.abort()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                errors.append(exc)
        for publisher in (self.kuksa_publisher, self.udp_publisher):
            if publisher is None:
                continue
            try:
                publisher.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                errors.append(exc)
        if errors:
            raise RuntimeError(
                "cannot abort all CarSky transports: "
                + "; ".join(str(error) for error in errors)
            ) from errors[0]

    def __enter__(self) -> "CarSkyEdgeSink":
        return self

    def __exit__(
        self, exc_type: object, _exc: object, _traceback: object
    ) -> None:
        # A JSONL file is evidence only when the surrounding replay completed
        # cleanly. Preserve the last complete artifact on every exceptional
        # context exit while still closing both live transport branches.
        if exc_type is None:
            self.close()
        else:
            self.abort()


def create_edge_sink(
    *,
    trip_id: str,
    source_fps: float,
    ttl_ms: int = DEFAULT_TTL_MS,
    model_versions: Mapping[str, str] | None = None,
    enable_udp: bool = True,
    udp_host: str = DEFAULT_ANDROID_HOST,
    udp_port: int = DEFAULT_ANDROID_PORT,
    enable_kuksa: bool = True,
    kuksa_host: str = DEFAULT_KUKSA_HOST,
    kuksa_port: int = DEFAULT_KUKSA_PORT,
    kuksa_timeout_s: float = 2.0,
    record_envelopes: Path | None = None,
    transport_error_handler: Callable[[str], None] | None = None,
) -> CarSkyEdgeSink:
    """Construct and preflight all requested transports for one trip."""

    # A compact 64-bit run nonce leaves MTU headroom while preserving a
    # practically unique session for the receiver ordering guard.
    session_id = f"r:{trip_id}:{uuid4().hex[:16]}"
    builder = DecisionEnvelopeBuilder(
        session_id=session_id,
        source_mode="replay",
        source_fps=source_fps,
        ttl_ms=ttl_ms,
        model_versions=model_versions,
    )
    udp: UdpDecisionPublisher | None = None
    kuksa: KuksaSignalPublisher | None = None
    recorder: AtomicEnvelopeJsonlRecorder | None = None
    try:
        if record_envelopes is not None:
            recorder = AtomicEnvelopeJsonlRecorder(record_envelopes)
        if enable_udp:
            udp = UdpDecisionPublisher(udp_host, udp_port)
        if enable_kuksa:
            kuksa = KuksaSignalPublisher(
                KuksaGrpcBackend(
                    kuksa_host,
                    kuksa_port,
                    timeout_s=kuksa_timeout_s,
                )
            )
            kuksa.connect()
        return CarSkyEdgeSink(
            builder,
            udp_publisher=udp,
            kuksa_publisher=kuksa,
            envelope_recorder=recorder,
            transport_error_handler=transport_error_handler,
        )
    except BaseException:
        for resource, abort in (
            (kuksa, False),
            (udp, False),
            (recorder, True),
        ):
            if resource is None:
                continue
            try:
                if abort:
                    resource.abort()
                else:
                    resource.close()
            except Exception:
                pass
        raise


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = replay_models.build_parser()
    parser.prog = "safeloop-carsky-edge"
    parser.description = (
        "Chạy unified replay và phát decision thật sang CarSky KUKSA/AAOS."
    )
    parser.set_defaults(mode="realtime")
    parser.add_argument("--no-udp", action="store_true")
    parser.add_argument("--udp-host", default=DEFAULT_ANDROID_HOST)
    parser.add_argument("--udp-port", type=int, default=DEFAULT_ANDROID_PORT)
    parser.add_argument("--no-kuksa", action="store_true")
    parser.add_argument("--kuksa-host", default=DEFAULT_KUKSA_HOST)
    parser.add_argument("--kuksa-port", type=int, default=DEFAULT_KUKSA_PORT)
    parser.add_argument("--kuksa-timeout", type=float, default=2.0)
    parser.add_argument("--decision-ttl-ms", type=int, default=DEFAULT_TTL_MS)
    parser.add_argument(
        "--record-envelopes",
        type=Path,
        help="ghi atomic strict decision JSONL; một envelope cho mỗi frame",
    )
    return parser


def _headless_show_requested(args: argparse.Namespace) -> bool:
    return args.show and os.name != "nt" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 0:
        print("Lỗi: --limit phải >= 0", file=sys.stderr)
        return 2
    if _headless_show_requested(args):
        print(
            "Lỗi: --show cần desktop display; trên server headless hãy dùng "
            "--write-video",
            file=sys.stderr,
        )
        return 2

    trip_ids = args.trip or [f"T{index:02d}d" for index in range(1, 11)]
    if args.record_envelopes is not None and len(trip_ids) != 1:
        print(
            "Lỗi: --record-envelopes cần đúng một --trip để tránh ghi đè",
            file=sys.stderr,
        )
        return 2
    summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    try:
        wire_model_versions, model_provenance = build_model_provenance(
            args.c1_checkpoint,
            args.c2_bundle,
        )
    except (OSError, ValueError) as exc:
        print(f"Lỗi model provenance: {exc}", file=sys.stderr)
        return 2
    for trip_id in trip_ids:
        sink: CarSkyEdgeSink | None = None
        completed_without_exception = False
        commit_recording = False
        try:
            trip_path = args.dataset / trip_id
            loader = TripLoader(trip_path)
            sink = create_edge_sink(
                trip_id=loader.trip_id,
                source_fps=loader.fps,
                ttl_ms=args.decision_ttl_ms,
                model_versions=wire_model_versions,
                enable_udp=not args.no_udp,
                udp_host=args.udp_host,
                udp_port=args.udp_port,
                enable_kuksa=not args.no_kuksa,
                kuksa_host=args.kuksa_host,
                kuksa_port=args.kuksa_port,
                kuksa_timeout_s=args.kuksa_timeout,
                record_envelopes=args.record_envelopes,
                transport_error_handler=lambda message: print(
                    f"Cảnh báo CarSky: {message}", file=sys.stderr, flush=True
                ),
            )
            summary = replay_models.run_trip(
                trip_path,
                c1_checkpoint=args.c1_checkpoint,
                c2_bundle=args.c2_bundle,
                output_dir=args.output_dir,
                device=args.device,
                mode=args.mode,
                speed=args.speed,
                limit=args.limit,
                c1_stride=args.c1_stride,
                c1_ema=args.c1_ema,
                c1_warning_on=args.c1_warning_on,
                c1_warning_off=args.c1_warning_off,
                show=args.show,
                write_video=args.write_video,
                video_fourcc=args.video_fourcc,
                frame_sink=sink,
            )
            transport_summary = sink.summary()
            stopped_by_user = bool(summary["stopped_by_user"])
            transport_summary["recording_committed"] = (
                not stopped_by_user
                if sink.envelope_recorder is not None
                else None
            )
            summary["carsky_transport"] = transport_summary
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            completed_without_exception = True
            commit_recording = not stopped_by_user
            if (
                transport_summary["live_transport_enabled"]
                and transport_summary["live_transport_pass"] is not True
            ):
                failures.append(
                    {
                        "trip_id": trip_id,
                        "error": (
                            "live transport incomplete: "
                            f"frames={transport_summary['frames_enveloped']}, "
                            f"udp_sent={transport_summary['udp_sent']}, "
                            f"kuksa_sent={transport_summary['kuksa_sent']}, "
                            f"udp={transport_summary['udp_errors']}, "
                            f"kuksa={transport_summary['kuksa_errors']}"
                        ),
                    }
                )
        except Exception as exc:
            failure = {
                "trip_id": trip_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(
                json.dumps(failure, ensure_ascii=False),
                file=sys.stderr,
                flush=True,
            )
        finally:
            if sink is not None:
                try:
                    if completed_without_exception and commit_recording:
                        sink.close()
                    else:
                        sink.abort()
                except Exception as exc:
                    failure = {
                        "trip_id": trip_id,
                        "error": f"transport close: {type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    print(
                        json.dumps(failure, ensure_ascii=False),
                        file=sys.stderr,
                        flush=True,
                    )

    report: dict[str, object] = {
        "pipeline": "SafeLoop unified replay -> decision envelope -> CarSky",
        "android_udp": {
            "enabled": not args.no_udp,
            "host": args.udp_host,
            "port": args.udp_port,
        },
        "kuksa": {
            "enabled": not args.no_kuksa,
            "host": args.kuksa_host,
            "port": args.kuksa_port,
        },
        "record_envelopes": (
            str(args.record_envelopes)
            if args.record_envelopes is not None
            else None
        ),
        "model_provenance": model_provenance,
        "completed_trips": len(summaries),
        "processed_frames": sum(int(item["frames"]) for item in summaries),
        "summaries": summaries,
        "failures": failures,
    }
    report_path = args.output_dir / "carsky_edge_report.json"
    _write_json_atomic(report_path, report)
    print(f"REPORT={report_path}", flush=True)
    stopped = any(bool(item["stopped_by_user"]) for item in summaries)
    return 0 if not failures and not stopped else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AtomicEnvelopeJsonlRecorder",
    "CarSkyEdgeSink",
    "DEFAULT_ANDROID_HOST",
    "DEFAULT_ANDROID_PORT",
    "DEFAULT_KUKSA_HOST",
    "DEFAULT_KUKSA_PORT",
    "EdgeFrameReceipt",
    "EnvelopeRecordError",
    "EnvelopeRecordReceipt",
    "MODEL_VERSION_SHA256_PREFIX_HEX",
    "build_model_provenance",
    "build_parser",
    "create_edge_sink",
    "main",
]
