from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from typing import Iterable


def timed_get(base_url: str, path: str, timeout: float) -> tuple[float, int, dict[str, str]]:
    started = time.perf_counter()
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=timeout) as response:
        body = response.read()
        elapsed = time.perf_counter() - started
        headers = {key.lower(): value for key, value in response.headers.items()}
        return elapsed, len(body), headers


def benchmark_paths(base_url: str, paths: Iterable[str], timeout: float) -> int:
    failures = 0
    for path in paths:
        try:
            elapsed, size_bytes, headers = timed_get(base_url, path, timeout)
            extra = ""
            total = headers.get("x-total-count")
            if total is not None:
                extra = f" total={total}"
            print(f"{path} OK {elapsed:.3f}s bytes={size_bytes}{extra}")
        except urllib.error.HTTPError as exc:
            failures += 1
            print(f"{path} HTTP {exc.code} {exc.reason}", file=sys.stderr)
        except Exception as exc:
            failures += 1
            print(f"{path} ERR {exc!r}", file=sys.stderr)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the speaker endpoints that have historically regressed."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8011", help="Backend base URL.")
    parser.add_argument("--channel-id", type=int, required=True, help="Channel id to benchmark.")
    parser.add_argument("--speaker-id", type=int, required=True, help="Speaker id to benchmark.")
    parser.add_argument("--page-size", type=int, default=48, help="Speaker page size to request.")
    parser.add_argument("--detail-page-size", type=int, default=50, help="Appearance/profile page size.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds.")
    args = parser.parse_args()

    paths = [
        f"/speakers?channel_id={args.channel_id}&limit={args.page_size}",
        f"/speakers/stats?channel_id={args.channel_id}",
        f"/speakers/{args.speaker_id}",
        f"/speakers/{args.speaker_id}/appearances?limit={args.detail_page_size}",
        f"/speakers/{args.speaker_id}/profiles?limit={args.detail_page_size}",
    ]
    return benchmark_paths(args.base_url, paths, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
