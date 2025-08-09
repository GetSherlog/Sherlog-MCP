import os
import io
import json
import time
import uuid
import shutil
import signal
import threading
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import redis

from fastmcp import Context

from sherlog_mcp.session import app, logger


SHERLOG_ROOT = Path(os.getenv("SHERLOG_ROOT", "/var/sherlog")).resolve()
JOBS_ROOT = (SHERLOG_ROOT / "jobs").resolve()

WORKSPACE_ROOT_ENV = os.getenv("WORKSPACE_ROOT")
WORKSPACE_ROOT = Path(WORKSPACE_ROOT_ENV).resolve() if WORKSPACE_ROOT_ENV else None

DEFAULT_GRADLE_ARGS = [
    "--no-daemon",
    "--stacktrace",
    "--console=plain",
    "--configuration-cache",
    "--build-cache",
]

REDIS_URL = os.getenv("REDIS_URL") or "redis://redis:6379/0"

_redis_client: Optional[redis.Redis] = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
    return _redis_client


def _redis_state_key(job_id: str) -> str:
    return f"job:{job_id}:state"


def _redis_logs_key(job_id: str) -> str:
    return f"logs:{job_id}"


def _redis_update_state(job_id: str, payload: Dict[str, Any]) -> None:
    client = _get_redis()
    client.hset(_redis_state_key(job_id), mapping={k: json.dumps(v) for k, v in payload.items()})


def _redis_read_state(job_id: str) -> Optional[Dict[str, Any]]:
    client = _get_redis()
    data = cast(Dict[str, str], client.hgetall(_redis_state_key(job_id)))
    if not data:
        return None
    out: Dict[str, Any] = {}
    for k, v in data.items():
        try:
            out[k] = json.loads(v)
        except Exception:
            out[k] = v
    return out


def _redis_xadd_log(job_id: str, line: str) -> None:
    client = _get_redis()
    client.xadd(_redis_logs_key(job_id), {"line": line})


@dataclass
class JobMeta:
    status: str
    code: Optional[int] = None
    startedAt: Optional[int] = None
    finishedAt: Optional[int] = None
    pid: Optional[int] = None
    rootDir: Optional[str] = None
    task: Optional[str] = None
    args: Optional[List[str]] = None
    artifacts: Optional[Dict[str, List[str]]] = None  # kind -> list of absolute paths


def _ensure_dirs(job_id: str) -> Tuple[Path, Path, Path]:
    job_dir = (JOBS_ROOT / job_id).resolve()
    artifacts_dir = (job_dir / "artifacts").resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return job_dir, artifacts_dir, job_dir / "build.log"


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_root_dir(root_dir: str) -> Path:
    p = Path(root_dir).resolve()
    if not p.exists() or not p.is_dir():
        raise ValueError(f"rootDir does not exist or is not a directory: {root_dir}")
    if WORKSPACE_ROOT and not str(p).startswith(str(WORKSPACE_ROOT)):
        raise ValueError(
            f"rootDir must be under WORKSPACE_ROOT: {WORKSPACE_ROOT}; got {p}"
        )
    return p


def _discover_artifacts(project_root: Path) -> Dict[str, List[str]]:
    artifacts: Dict[str, List[str]] = {
        "apk": [],
        "aab": [],
        "mapping": [],
        "test-report": [],
    }

    search_roots = [project_root]

    for root in search_roots:
        for path in root.rglob("*.apk"):
            artifacts["apk"].append(str(path.resolve()))
        for path in root.rglob("*.aab"):
            artifacts["aab"].append(str(path.resolve()))
        for path in root.rglob("mapping.txt"):
            artifacts["mapping"].append(str(path.resolve()))
        for path in root.rglob("build/reports/tests"):
            if path.is_dir():
                artifacts["test-report"].append(str(path.resolve()))

    for k, v in list(artifacts.items()):
        seen = set()
        deduped: List[str] = []
        for item in v:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        artifacts[k] = deduped

    return artifacts


def _copy_artifact_safely(src: Path, dest_dir: Path) -> Path:
    dest = (dest_dir / src.name).resolve()
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if src.is_dir():
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp)
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    else:
        with src.open("rb") as rf, tmp.open("wb") as wf:
            shutil.copyfileobj(rf, wf, length=1024 * 1024)
            wf.flush()
            os.fsync(wf.fileno())
        os.replace(tmp, dest)
    return dest


def _collect_and_copy_artifacts(project_root: Path, artifacts_dir: Path) -> Dict[str, List[str]]:
    discovered = _discover_artifacts(project_root)
    collected: Dict[str, List[str]] = {"apk": [], "aab": [], "mapping": [], "test-report": []}
    for kind, paths in discovered.items():
        for p in paths:
            try:
                copied = _copy_artifact_safely(Path(p), artifacts_dir)
                collected[kind].append(str(copied))
            except Exception as e:
                logger.warning(f"Failed to copy artifact {p}: {e}")
    return collected


def _update_meta(job_dir: Path, **updates: Any) -> None:
    meta_path = job_dir / "meta.json"
    meta = _read_json(meta_path)
    meta.update(updates)
    _write_json_atomic(meta_path, meta)


def _run_build_job(job_id: str, root_dir: Path, task: str, args: List[str], extra_env: Dict[str, str]) -> None:
    job_dir, artifacts_dir, log_path = _ensure_dirs(job_id)

    started = int(time.time())
    _write_json_atomic(
        job_dir / "meta.json",
        JobMeta(
            status="provisioning",
            startedAt=started,
            rootDir=str(root_dir),
            task=task,
            args=args,
        ).__dict__,
    )
    _redis_update_state(job_id, {"status": "provisioning", "startedAt": started})

    env = os.environ.copy()
    env.update(extra_env or {})

    gradlew_path = "./gradlew"
    cmd = [gradlew_path, task, *args]

    # Prepare process: combine stdout+stderr
    proc: Optional[subprocess.Popen[bytes]] = None
    code: Optional[int] = None
    bytes_written = 0
    last_heartbeat = 0.0
    try:
        with open(log_path, "ab", buffering=0) as lf:
            # Start process in its own process group for easier cancellation
            proc = subprocess.Popen(
                cmd,
                cwd=str(root_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                bufsize=1,
            )
            _update_meta(job_dir, status="running", pid=proc.pid)
            _redis_update_state(job_id, {"status": "running", "pid": proc.pid})

            # Stream output to file + redis stream (line-buffered)
            assert proc.stdout is not None
            stdout = proc.stdout
            line_buffer = b""
            while True:
                chunk = stdout.read(8192)
                if not chunk:
                    break
                lf.write(chunk)
                bytes_written += len(chunk)

                # Push completed lines to redis stream
                line_buffer += chunk
                parts = line_buffer.split(b"\n")
                line_buffer = parts[-1]
                for part in parts[:-1]:
                    try:
                        _redis_xadd_log(job_id, part.decode("utf-8", errors="ignore"))
                    except Exception:
                        pass

                # Heartbeat every ~2s
                now = time.time()
                if now - last_heartbeat > 2.0:
                    _redis_update_state(job_id, {"status": "running", "lastOffset": bytes_written})
                    last_heartbeat = now

            code = proc.wait()

        # Finalizing: collect artifacts regardless of code (for diagnostics)
        _update_meta(job_dir, status="finalizing")
        _redis_update_state(job_id, {"status": "finalizing"})
        collected = _collect_and_copy_artifacts(root_dir, artifacts_dir)
        finished = int(time.time())
        final_status = "completed" if code == 0 else "failed"
        _update_meta(
            job_dir,
            status=final_status,
            code=code,
            finishedAt=finished,
            artifacts=collected,
        )
        _redis_update_state(job_id, {"status": final_status, "code": code, "finishedAt": finished})
    except Exception as e:
        logger.error(f"Job {job_id} failed with exception: {e}", exc_info=True)
        finished = int(time.time())
        _update_meta(job_dir, status="failed", code=code if code is not None else -1, finishedAt=finished)
        _redis_update_state(job_id, {"status": "failed", "code": code if code is not None else -1, "finishedAt": finished})
        # Best-effort: append error to log
        try:
            with open(log_path, "ab", buffering=0) as lf:
                lf.write(f"\n[FATAL] Job failed: {e}\n".encode("utf-8", errors="ignore"))
        except Exception:
            pass


@app.tool()
async def android_start_build(
    rootDir: str,
    task: str = "assembleDebug",
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """Start an Android Gradle build as a background job.

    Inputs:
    - rootDir: Absolute path to the Android project root (validated against WORKSPACE_ROOT if set)
    - task: Gradle task (e.g., assembleDebug)
    - args: Additional Gradle args; defaults optimized for long runs
    - env: Extra environment variables to pass to the build

    Returns: {"jobId": str, "status": "queued|running"}
    """
    project_root = _validate_root_dir(rootDir)

    job_id = str(uuid.uuid4())
    job_dir, _, _ = _ensure_dirs(job_id)

    _write_json_atomic((job_dir / "meta.json"), JobMeta(status="queued").__dict__)

    build_args = list(DEFAULT_GRADLE_ARGS if not args else args)

    thread = threading.Thread(
        target=_run_build_job,
        args=(job_id, project_root, task, build_args, env or {}),
        name=f"android-build-{job_id[:8]}",
        daemon=True,
    )
    thread.start()

    return {"jobId": job_id, "status": "queued"}


@app.tool()
async def android_job_status(jobId: str) -> Dict[str, Any]:
    """Get job status snapshot for a given jobId."""
    # Prefer Redis state if present, otherwise fallback to meta.json
    state = _redis_read_state(jobId)
    if state and "status" in state:
        return {
            "status": state.get("status", "queued"),
            "code": state.get("code"),
            "startedAt": state.get("startedAt"),
            "finishedAt": state.get("finishedAt"),
        }

    job_dir = (JOBS_ROOT / jobId).resolve()
    meta_path = job_dir / "meta.json"
    if not meta_path.exists():
        return {"status": "not_found"}
    meta = _read_json(meta_path)
    # Normalize output per contract
    status = meta.get("status", "queued")
    return {
        "status": status,
        "code": meta.get("code"),
        "startedAt": meta.get("startedAt"),
        "finishedAt": meta.get("finishedAt"),
    }


@app.tool()
async def android_tail_logs(jobId: str, offset: int = 0, limit: int = 65536) -> Dict[str, Any]:
    """Tail the canonical build log from a byte offset."""
    job_dir = (JOBS_ROOT / jobId).resolve()
    log_path = job_dir / "build.log"
    if not log_path.exists():
        return {"text": "", "nextOffset": offset}

    with open(log_path, "rb") as f:
        try:
            f.seek(offset)
        except OSError:
            offset = 0
            f.seek(0)
        data = f.read(max(0, int(limit)))

    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    return {"text": text, "nextOffset": offset + len(data)}


def _read_pid(job_dir: Path) -> Optional[int]:
    pid = _read_json(job_dir / "meta.json").get("pid")
    if isinstance(pid, int) and pid > 0:
        return pid
    return None


@app.tool()
async def android_cancel_job(jobId: str) -> Dict[str, Any]:
    """Attempt to cancel a running job. Returns {"cancelled": true} if the signal was sent."""
    job_dir = (JOBS_ROOT / jobId).resolve()
    pid = _read_pid(job_dir)
    if not pid:
        return {"cancelled": False}

    try:
        # Try killing the whole process group if possible
        if hasattr(os, "getpgid"):
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
        _update_meta(job_dir, status="cancelled")
        _redis_update_state(jobId, {"status": "cancelled"})
        return {"cancelled": True}
    except ProcessLookupError:
        _update_meta(job_dir, status="cancelled")
        _redis_update_state(jobId, {"status": "cancelled"})
        return {"cancelled": True}
    except Exception as e:
        logger.warning(f"Failed to cancel job {jobId}: {e}")
        return {"cancelled": False}


@app.tool()
async def android_fetch_artifact(jobId: str, kind: str) -> Dict[str, Any]:
    """Return the first matching artifact path for a given kind (apk|aab|mapping|test-report)."""
    job_dir = (JOBS_ROOT / jobId).resolve()
    meta = _read_json(job_dir / "meta.json")
    artifacts = meta.get("artifacts") or {}
    paths: List[str] = artifacts.get(kind, [])
    if not paths:
        artifacts_dir = (job_dir / "artifacts").resolve()
        if artifacts_dir.exists():
            candidates: List[str] = []
            if kind == "apk":
                candidates = [str(p) for p in artifacts_dir.glob("*.apk")]
            elif kind == "aab":
                candidates = [str(p) for p in artifacts_dir.glob("*.aab")]
            elif kind == "mapping":
                candidates = [str(p) for p in artifacts_dir.rglob("mapping.txt")]
            elif kind == "test-report":
                candidates = [str(p) for p in artifacts_dir.glob("*") if Path(p).is_dir()]
            if candidates:
                paths = candidates
                meta.setdefault("artifacts", {}).setdefault(kind, paths)
                _write_json_atomic(job_dir / "meta.json", meta)

    if not paths:
        return {"path": "", "sizeBytes": 0}

    p = Path(paths[0])
    size = 0
    try:
        if p.is_file():
            size = p.stat().st_size
        elif p.is_dir():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except Exception:
        size = 0

    return {"path": str(p), "sizeBytes": size} 