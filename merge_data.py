#!/usr/bin/env python3
"""Restore original gzip archives from ordinary Git files (Python 3.9+)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
BUFFER_SIZE = 4 * 1024 * 1024


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside data directory: {relative}")
    return path


def digest(path: Path) -> tuple[int, str]:
    total = 0
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER_SIZE), b""):
            total += len(block)
            sha.update(block)
    return total, sha.hexdigest()


def merge(manifest: dict, source: Path, destination: Path) -> None:
    if manifest.get("version") != 1:
        raise ValueError("Unsupported data manifest version")
    for entry in manifest["files"]:
        target = safe_path(destination, entry["path"])
        expected = (entry["size"], entry["sha256"])
        if target.exists():
            if digest(target) != expected:
                raise ValueError(f"Existing file differs; move it aside before retrying: {target}")
            print(f"Verified existing: {entry['path']}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            full_sha = hashlib.sha256()
            full_size = 0
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                for part in entry["parts"]:
                    path = safe_path(source, part["path"])
                    part_sha = hashlib.sha256()
                    part_size = 0
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(BUFFER_SIZE), b""):
                            output.write(block)
                            part_sha.update(block)
                            full_sha.update(block)
                            part_size += len(block)
                            full_size += len(block)
                    if (part_size, part_sha.hexdigest()) != (part["size"], part["sha256"]):
                        raise ValueError(f"Corrupt or incomplete chunk: {part['path']}")
                if (full_size, full_sha.hexdigest()) != expected:
                    raise ValueError(f"Merged checksum mismatch: {entry['path']}")
            os.replace(temporary, target)
            print(f"Restored: {entry['path']} ({full_size:,} bytes)")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT, help="Restore under this directory (default: repository root)")
    args = parser.parse_args()
    manifest = {"version": 1, "files": []}
    for group in ("github_all", "top10"):
        group_manifest = json.loads((ROOT / group / "raw/metadata/chunks.json").read_text(encoding="utf-8"))
        if group_manifest.get("version") != 1:
            raise ValueError("Unsupported data manifest version")
        manifest["files"].extend(group_manifest["files"])
    merge(manifest, ROOT, args.output_dir.resolve())
    print(f"Processed {len(manifest['files'])} split archive(s). Smaller archives are stored directly. Run: python3 verify_submission.py")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Data restore failed: {error}")
