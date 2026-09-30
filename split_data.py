#!/usr/bin/env python3
"""Split only gzip archives over 90 MB; verify chunks, then remove originals."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHUNK_SIZE = 90_000_000


def main() -> None:
    archives = []
    for group, field in (("github_all", "stored_path"), ("top10", "compressed_path")):
        metadata = json.loads((ROOT / group / "raw/metadata/raw_archives.json").read_text())
        archives.extend(ROOT / group / row[field] for row in metadata if row[field].endswith(".gz"))
    # Validate inputs before writing any chunks.
    for archive in archives:
        with archive.open("rb") as stream:
            if stream.read(2) != b"\x1f\x8b":
                raise ValueError(f"Not a gzip archive (possibly an LFS pointer): {archive}")
    entries = []
    for archive in sorted(archives):
        if archive.stat().st_size <= CHUNK_SIZE:
            continue
        relative = archive.relative_to(ROOT)
        part_dir = archive.parent
        part_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        full_sha = hashlib.sha256()
        total = 0
        with archive.open("rb") as stream:
            for index, block in enumerate(iter(lambda: stream.read(CHUNK_SIZE), b""), 1):
                path = part_dir / f"{archive.name}.part{index:03d}"
                path.write_bytes(block)
                if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(block).digest():
                    raise ValueError(f"Written chunk checksum mismatch: {path}")
                full_sha.update(block)
                total += len(block)
                parts.append({"path": path.relative_to(ROOT).as_posix(), "size": len(block), "sha256": hashlib.sha256(block).hexdigest()})
        entries.append({"path": relative.as_posix(), "size": total, "sha256": full_sha.hexdigest(), "parts": parts})
    manifest = {"version": 1, "chunk_size_bytes": CHUNK_SIZE, "files": entries}
    for group in ("github_all", "top10"):
        grouped = {**manifest, "files": [entry for entry in entries if entry["path"].startswith(group + "/")]}
        (ROOT / group / "raw/metadata/chunks.json").write_text(json.dumps(grouped, indent=2) + "\n", encoding="utf-8")
    for entry in entries:
        archive = ROOT / entry["path"]
        if hashlib.sha256(archive.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Original changed during splitting: {archive}")
        archive.unlink()
    print(f"Created {sum(len(entry['parts']) for entry in entries)} chunks for {len(entries)} archives; maximum {CHUNK_SIZE:,} bytes each.")


if __name__ == "__main__":
    main()
