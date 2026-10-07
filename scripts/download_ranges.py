"""Download a large HTTP file with resumable parallel byte ranges.

This is intentionally small and dependency-free so Zenodo archives can be
resumed when a single connection is throttled or interrupted.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--md5", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=32)
    return parser.parse_args()


def download_part(url: str, part_path: Path, start: int, end: int, retries: int = 6) -> None:
    expected = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected:
        return
    for attempt in range(retries):
        tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
        try:
            request = Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                    "User-Agent": "car-tse-dataset-downloader/1.0",
                },
            )
            with urlopen(request, timeout=120) as response, open(tmp_path, "wb") as out:
                status = getattr(response, "status", response.getcode())
                if status != 206:
                    raise RuntimeError(f"server returned HTTP {status} for range {start}-{end}")
                shutil.copyfileobj(response, out, length=1024 * 1024)
            if tmp_path.stat().st_size != expected:
                raise RuntimeError(
                    f"short range {start}-{end}: got {tmp_path.stat().st_size}, expected {expected}"
                )
            os.replace(tmp_path, part_path)
            return
        except (HTTPError, URLError, OSError, RuntimeError) as exc:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            if attempt == retries - 1:
                raise RuntimeError(f"range {start}-{end} failed: {exc}") from exc
            time.sleep(min(30, 2**attempt))


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    part_dir = output.with_name(output.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, args.chunk_mib) * 1024 * 1024
    ranges = [
        (start, min(args.size - 1, start + chunk_size - 1))
        for start in range(0, args.size, chunk_size)
    ]
    print(f"Downloading {args.size:,} bytes in {len(ranges)} ranges with {args.workers} workers", flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(download_part, args.url, part_dir / f"part-{i:06d}.bin", start, end): i
            for i, (start, end) in enumerate(ranges)
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed == len(ranges) or completed % 4 == 0:
                print(f"ranges {completed}/{len(ranges)}", flush=True)
    temp_output = output.with_suffix(output.suffix + ".assembling")
    with open(temp_output, "wb") as stream:
        for i in range(len(ranges)):
            with open(part_dir / f"part-{i:06d}.bin", "rb") as part:
                shutil.copyfileobj(part, stream, length=8 * 1024 * 1024)
    if temp_output.stat().st_size != args.size:
        raise RuntimeError(f"assembled file has size {temp_output.stat().st_size}, expected {args.size}")
    os.replace(temp_output, output)
    if args.md5:
        actual = md5_file(output)
        if actual.lower() != args.md5.lower():
            raise RuntimeError(f"MD5 mismatch: got {actual}, expected {args.md5}")
        print(f"MD5 OK: {actual}")
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; range files are retained for resume.", file=sys.stderr)
        raise SystemExit(130)
