from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlopen


BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"
BIN_BASE_DIR = BACKEND_DIR / "bin" / "postgres"
RUNTIME_DIR = BACKEND_DIR / "runtime" / "postgres"


class EmbeddedPostgresError(RuntimeError):
    pass


def _is_windows() -> bool:
    return platform.system().lower() == "windows"


def _exe(name: str) -> str:
    return f"{name}.exe" if _is_windows() else name


def _platform_artifact() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        return "embedded-postgres-binaries-windows-amd64"

    if system == "linux":
        if machine in {"x86_64", "amd64"}:
            return "embedded-postgres-binaries-linux-amd64"
        if machine in {"aarch64", "arm64"}:
            return "embedded-postgres-binaries-linux-arm64v8"

    if system == "darwin":
        if machine in {"x86_64", "amd64"}:
            return "embedded-postgres-binaries-darwin-amd64"
        if machine in {"aarch64", "arm64"}:
            return "embedded-postgres-binaries-darwin-arm64v8"

    raise EmbeddedPostgresError(f"Unsupported platform for embedded postgres: {system}/{machine}")


def _default_pg_bin_dir() -> Path:
    artifact = _platform_artifact()
    version = (os.getenv("EMBEDDED_PG_VERSION") or "17.6.0").strip()
    install_dir = BIN_BASE_DIR / f"{artifact}-{version}"
    return install_dir / "bin"


def _resolve_pg_bin_dir() -> Path:
    override = (os.getenv("EMBEDDED_PG_BIN_DIR") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _default_pg_bin_dir()


def _binary_path(bin_dir: Path, name: str) -> Path:
    return bin_dir / _exe(name)


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, timeout_seconds: int = 30) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _is_port_open(host, port):
            return True
        time.sleep(0.25)
    return False


def _pid_exists(pid: int) -> bool:
    try:
        if int(pid) <= 0:
            return False
    except Exception:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _clear_stale_postmaster_pid(data_dir: Path) -> bool:
    """Remove stale postmaster.pid when referenced PID is no longer alive."""
    pid_file = data_dir / "postmaster.pid"
    if not pid_file.exists():
        return False
    try:
        first_line = (pid_file.read_text(encoding="utf-8", errors="replace").splitlines() or [""])[0].strip()
        pid = int(first_line)
    except Exception:
        # Corrupt pid file: remove it so startup can proceed.
        try:
            pid_file.unlink()
            return True
        except Exception:
            return False
    if _pid_exists(pid):
        return False
    try:
        pid_file.unlink()
        return True
    except Exception:
        return False


def _kill_processes_on_port(port: int) -> None:
    if not _is_windows():
        return
    try:
        result = subprocess.run(
            ["netstat", "-aon"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode != 0:
            return
        pids: set[int] = set()
        needle = f":{int(port)}"
        for raw_line in (result.stdout or "").splitlines():
            line = raw_line.strip()
            if not line or needle not in line or "LISTENING" not in line.upper():
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                pid = int(parts[-1])
            except Exception:
                continue
            if pid > 0:
                pids.add(pid)
        for pid in pids:
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
    except Exception:
        pass


def _find_available_port(host: str, start_port: int, max_tries: int = 40) -> int | None:
    port = int(start_port)
    for _ in range(max(1, int(max_tries))):
        if not _is_port_open(host, port):
            return port
        port += 1
    return None


def _resolve_writable_log_path(preferred: Path) -> Path:
    preferred.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(preferred, "a", encoding="utf-8"):
            pass
        return preferred
    except Exception:
        pass
    # Fallback to a timestamped logfile when the canonical one is locked.
    alt = preferred.parent / f"postgres-{int(time.time())}.log"
    try:
        with open(alt, "a", encoding="utf-8"):
            pass
        return alt
    except Exception:
        # Let caller fail with original path semantics if even fallback is not writable.
        return preferred


def _kill_embedded_postgres_processes(data_dir: Path) -> None:
    """Best-effort kill of postgres.exe processes tied to this embedded data dir."""
    if not _is_windows():
        return
    data = str(data_dir).replace("'", "''")
    data_fwd = str(data_dir).replace("\\", "/").replace("'", "''")
    ps = (
        "$d='" + data + "'; $df='" + data_fwd + "'; "
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -ieq 'postgres.exe' -and ("
        "($_.CommandLine -like ('*' + $d + '*')) -or "
        "($_.CommandLine -like ('*' + $df + '*')) -or "
        "($_.CommandLine -like '*embedded-postgres-binaries*')"
        ") } | "
        "ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {} }"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        pass


def _download_and_extract_binaries(bin_dir: Path) -> None:
    artifact = _platform_artifact()
    version = (os.getenv("EMBEDDED_PG_VERSION") or "17.6.0").strip()
    maven_base = (os.getenv("EMBEDDED_PG_MAVEN_BASE") or "https://repo1.maven.org/maven2/io/zonky/test/postgres").rstrip("/")
    jar_name = f"{artifact}-{version}.jar"
    jar_url = f"{maven_base}/{artifact}/{version}/{jar_name}"

    install_root = bin_dir.parent
    install_root.mkdir(parents=True, exist_ok=True)
    marker = install_root / ".installed"
    if marker.exists() and _binary_path(bin_dir, "postgres").exists():
        return

    print(f"[postgres] Downloading embedded PostgreSQL binaries from {jar_url} ...", flush=True)
    with tempfile.TemporaryDirectory(prefix="chatalogue-pg-") as tmp:
        tmp_dir = Path(tmp)
        jar_path = tmp_dir / "postgres-binaries.jar"
        txz_path = tmp_dir / "postgres-binaries.txz"

        with urlopen(jar_url, timeout=120) as resp, open(jar_path, "wb") as out:
            out.write(resp.read())
        print("[postgres] Download complete. Extracting...", flush=True)

        with zipfile.ZipFile(jar_path, "r") as zf:
            txz_entries = [n for n in zf.namelist() if n.endswith(".txz")]
            if not txz_entries:
                raise EmbeddedPostgresError(f"Embedded postgres artifact did not include .txz payload: {jar_url}")
            with zf.open(txz_entries[0]) as src, open(txz_path, "wb") as out:
                out.write(src.read())

        if install_root.exists():
            shutil.rmtree(install_root)
        install_root.mkdir(parents=True, exist_ok=True)

        with tarfile.open(txz_path, "r:xz") as tf:
            root_real = install_root.resolve()
            for member in tf.getmembers():
                target = (install_root / member.name).resolve()
                if root_real not in target.parents and target != root_real:
                    raise EmbeddedPostgresError(f"Blocked unsafe tar entry: {member.name}")
            tf.extractall(path=install_root)

        if not _is_windows():
            for exe in ("postgres", "pg_ctl", "initdb"):
                path = _binary_path(bin_dir, exe)
                if path.exists():
                    path.chmod(path.stat().st_mode | 0o111)

        marker.write_text(f"{artifact}:{version}", encoding="utf-8")
        print("[postgres] Binaries extracted.", flush=True)


def _ensure_binaries() -> Path:
    bin_dir = _resolve_pg_bin_dir()
    postgres_bin = _binary_path(bin_dir, "postgres")
    pg_ctl_bin = _binary_path(bin_dir, "pg_ctl")
    initdb_bin = _binary_path(bin_dir, "initdb")

    if postgres_bin.exists() and pg_ctl_bin.exists() and initdb_bin.exists():
        return bin_dir

    auto_download = (os.getenv("EMBEDDED_PG_AUTO_DOWNLOAD") or "true").strip().lower() in {"1", "true", "yes", "on"}
    if auto_download:
        _download_and_extract_binaries(bin_dir)

    if not (postgres_bin.exists() and pg_ctl_bin.exists() and initdb_bin.exists()):
        raise EmbeddedPostgresError(
            "Embedded PostgreSQL binaries are not available. "
            "Set EMBEDDED_PG_BIN_DIR to a postgres bin directory or enable EMBEDDED_PG_AUTO_DOWNLOAD=true."
        )

    return bin_dir


def _run(cmd: list[str], *, env: Optional[dict[str, str]] = None, timeout_seconds: int = 120, detach_child: bool = False) -> None:
    kwargs: dict[str, object] = {"text": True, "env": env, "timeout": timeout_seconds}
    if detach_child and _is_windows():
        # On Windows, pg_ctl start spawns postgres as a child that inherits
        # stdout/stderr pipes. capture_output keeps the pipes open waiting for
        # ALL child processes to close them, causing subprocess.run to hang
        # even after pg_ctl itself exits. Use CREATE_NEW_PROCESS_GROUP and
        # redirect to DEVNULL so the pipe isn't inherited.
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["capture_output"] = True
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        raise EmbeddedPostgresError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\nstdout: {stdout.strip()}\nstderr: {stderr.strip()}"
        )


def _ensure_database_exists(
    host: str,
    port: int,
    user: str,
    database: str,
    *,
    connect_timeout_seconds: int = 5,
) -> None:
    try:
        import psycopg
        from psycopg import sql
    except Exception as exc:
        raise EmbeddedPostgresError(
            "psycopg is required to initialize embedded PostgreSQL databases. "
            "Install dependency 'psycopg[binary]' in the backend venv."
        ) from exc

    password = (os.getenv("EMBEDDED_PG_PASSWORD") or "").strip()
    timeout_s = max(2, int(connect_timeout_seconds or 5))
    dsn = f"host={host} port={port} user={user} dbname=postgres connect_timeout={timeout_s}"
    if password:
        dsn += f" password={password}"
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (database,))
            exists = cur.fetchone() is not None
            if not exists:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))


def _wait_for_database_ready(
    host: str,
    port: int,
    user: str,
    database: str,
    *,
    timeout_seconds: int = 120,
) -> None:
    """Wait until we can connect and verify/create target DB.

    This handles restart/recovery windows where TCP port is open but postgres is still
    replaying WAL or briefly rejecting connections.
    """
    deadline = time.time() + max(10, int(timeout_seconds or 120))
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            _ensure_database_exists(
                host,
                port,
                user,
                database,
                connect_timeout_seconds=3,
            )
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise EmbeddedPostgresError(
        f"Embedded PostgreSQL on {host}:{port} did not become query-ready in time. "
        f"Last error: {last_error}"
    )


def ensure_embedded_postgres() -> None:
    if (os.getenv("EMBEDDED_PG_ENABLED") or "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    host = (os.getenv("EMBEDDED_PG_HOST") or "127.0.0.1").strip()
    port = int((os.getenv("EMBEDDED_PG_PORT") or "55432").strip())
    user = (os.getenv("EMBEDDED_PG_USER") or "chatalogue").strip()
    database = (os.getenv("EMBEDDED_PG_DATABASE") or "chatalogue").strip()
    password = (os.getenv("EMBEDDED_PG_PASSWORD") or "").strip()
    data_dir = Path((os.getenv("EMBEDDED_PG_DATA_DIR") or str(DATA_DIR / "postgres")).strip())
    data_dir.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    ready_timeout = int((os.getenv("EMBEDDED_PG_READY_TIMEOUT_SECONDS") or "120").strip())

    if _is_port_open(host, port):
        try:
            _wait_for_database_ready(host, port, user, database, timeout_seconds=ready_timeout)
            return
        except Exception as first_ready_error:
            print(
                f"[postgres] Port {port} is open but DB is not query-ready. "
                f"Attempting embedded postgres restart... ({first_ready_error})",
                flush=True,
            )
            bin_dir = _ensure_binaries()
            pg_ctl_bin = _binary_path(bin_dir, "pg_ctl")
            log_path = _resolve_writable_log_path(RUNTIME_DIR / "postgres.log")

            # Best effort stop: stale/post-recovery states can leave the port open
            # but unresponsive to libpq connection attempts.
            try:
                _run(
                    [str(pg_ctl_bin), "-D", str(data_dir), "stop", "-m", "immediate"],
                    timeout_seconds=45,
                )
            except Exception:
                pass
            if _is_port_open(host, port):
                _kill_processes_on_port(port)
                time.sleep(1.0)
            _kill_embedded_postgres_processes(data_dir)
            _clear_stale_postmaster_pid(data_dir)

            start_cmd = [
                str(pg_ctl_bin),
                "-D",
                str(data_dir),
                "-l",
                str(log_path),
                "-w",
                "-t",
                "90",
                "-o",
                f"-h {host} -p {port}",
                "start",
            ]
            try:
                _run(start_cmd, timeout_seconds=150, detach_child=True)
            except Exception as start_error:
                # If original port remains unusable, move embedded postgres to the
                # next available local port for this process.
                alt_port = _find_available_port(host, port + 1, max_tries=60)
                if alt_port:
                    print(
                        f"[postgres] Port {port} unavailable after restart attempt; "
                        f"switching embedded postgres to port {alt_port}.",
                        flush=True,
                    )
                    os.environ["EMBEDDED_PG_PORT"] = str(alt_port)
                    port = int(alt_port)
                    log_path = _resolve_writable_log_path(RUNTIME_DIR / "postgres.log")
                    start_cmd = [
                        str(pg_ctl_bin),
                        "-D",
                        str(data_dir),
                        "-l",
                        str(log_path),
                        "-w",
                        "-t",
                        "90",
                        "-o",
                        f"-h {host} -p {port}",
                        "start",
                    ]
                    _run(start_cmd, timeout_seconds=150, detach_child=True)
                else:
                    pg_log_tail = ""
                    try:
                        log_tail = (log_path.read_text(encoding="utf-8", errors="replace") or "").splitlines()[-40:]
                        pg_log_tail = "\n".join(log_tail)
                    except Exception:
                        pg_log_tail = ""
                    raise EmbeddedPostgresError(
                        f"Embedded PostgreSQL restart failed after stale-port recovery: {start_error}\n"
                        f"postgres.log tail:\n{pg_log_tail}"
                    ) from start_error
            if not _wait_for_port(host, port, timeout_seconds=45):
                raise EmbeddedPostgresError(f"Embedded PostgreSQL did not reopen {host}:{port} after restart.")
            _wait_for_database_ready(host, port, user, database, timeout_seconds=ready_timeout)
            return

    bin_dir = _ensure_binaries()
    initdb_bin = _binary_path(bin_dir, "initdb")
    pg_ctl_bin = _binary_path(bin_dir, "pg_ctl")

    if not (data_dir / "PG_VERSION").exists():
        print(f"[postgres] Initializing database cluster in {data_dir} ...", flush=True)
        env = os.environ.copy()
        if password:
            env["PGPASSWORD"] = password
        _run(
            [
                str(initdb_bin),
                "-D",
                str(data_dir),
                "-U",
                user,
                "--encoding",
                "UTF8",
                "--auth",
                "trust",
            ],
            env=env,
            timeout_seconds=180,
        )
        print("[postgres] Database cluster initialized.", flush=True)

    # Recover from abrupt terminations leaving stale lock files.
    _kill_embedded_postgres_processes(data_dir)
    if _clear_stale_postmaster_pid(data_dir):
        print("[postgres] Cleared stale postmaster.pid.", flush=True)

    log_path = _resolve_writable_log_path(RUNTIME_DIR / "postgres.log")
    print(f"[postgres] Starting PostgreSQL on {host}:{port} ...", flush=True)
    start_cmd = [
        str(pg_ctl_bin),
        "-D",
        str(data_dir),
        "-l",
        str(log_path),
        "-w",
        "-t",
        "60",
        "-o",
        f"-h {host} -p {port}",
        "start",
    ]
    _run(start_cmd, timeout_seconds=120, detach_child=True)

    if not _wait_for_port(host, port, timeout_seconds=30):
        raise EmbeddedPostgresError(f"Embedded PostgreSQL did not open {host}:{port} in time.")

    print(f"[postgres] PostgreSQL is ready on port {port}.", flush=True)
    _wait_for_database_ready(host, port, user, database, timeout_seconds=ready_timeout)


def build_embedded_postgres_url() -> str:
    host = (os.getenv("EMBEDDED_PG_HOST") or "127.0.0.1").strip()
    port = int((os.getenv("EMBEDDED_PG_PORT") or "55432").strip())
    user = (os.getenv("EMBEDDED_PG_USER") or "chatalogue").strip()
    database = (os.getenv("EMBEDDED_PG_DATABASE") or "chatalogue").strip()
    password = (os.getenv("EMBEDDED_PG_PASSWORD") or "").strip()

    if password:
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
    return f"postgresql+psycopg://{user}@{host}:{port}/{database}"


def stop_embedded_postgres() -> None:
    data_dir = Path((os.getenv("EMBEDDED_PG_DATA_DIR") or str(DATA_DIR / "postgres")).strip())
    pg_ctl_bin = _binary_path(_resolve_pg_bin_dir(), "pg_ctl")
    if not pg_ctl_bin.exists() or not data_dir.exists():
        return

    result = subprocess.run(
        [str(pg_ctl_bin), "-D", str(data_dir), "stop", "-m", "fast"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().lower()
        if "is not running" in stderr or "no such process" in stderr:
            _clear_stale_postmaster_pid(data_dir)
            return
        raise EmbeddedPostgresError(f"Failed to stop embedded PostgreSQL: {(result.stderr or '').strip()}")
