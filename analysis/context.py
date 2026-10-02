"""Portable paths for the offline replication pipeline."""
from pathlib import Path
import gzip
import os

ARTIFACT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("AGENTHISTORY_OUTPUT", ARTIFACT / "build/analysis")).resolve()

def stage(name):
    path = OUTPUT / name
    path.mkdir(parents=True, exist_ok=True)
    return path

def raw_open(path, mode="rb"):
    path = Path(path)
    if not path.exists():
        path = path.with_name(path.name + ".gz")
    return gzip.open(path, mode) if path.suffix == ".gz" else path.open(mode)
