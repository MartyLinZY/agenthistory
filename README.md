# File descriptions

| Path | Contents |
| --- | --- |
| `top10/raw/` | Original records and collection metadata for ten repositories. |
| `top10/code/` | Collection, feature extraction, analysis and verification scripts. |
| `github_all/raw/` | Original cross-repository PR export, supplied annotations and collection metadata. |
| `github_all/code/` | AgentBench collection, analysis, classification and verification scripts. |
| `requirements.txt` | Python runtime dependencies. |
| `requirements-dev.txt` | Test dependencies. |
| `verify_submission.py` | Verifies both raw snapshots against their archived checksums. |

## Restore the data

Download or clone this repository, then run these commands from its root directory:

```bash
python3 merge_data.py
python3 verify_submission.py
```

Python 3.9 or later is required; both commands use only the standard library. Git LFS is not required. Archives no larger than **90,000,000 bytes (90 MB)** are kept intact in their original locations. Only `github_all/raw/prs.jsonl.gz` (411,312,583 bytes) exceeds this limit: it is stored alongside its original location as `prs.jsonl.gz.part001` through `.part005`. The first four chunks are 90 MB each; the last is 51,312,583 bytes. The other 14 gzip archives are stored directly without splitting.

The merge script restores `github_all/raw/prs.jsonl.gz` and verifies each chunk and the completed file against the sizes and SHA-256 hashes in `github_all/raw/metadata/chunks.json`. It does not extract the gzip archive or modify the smaller archives. Existing correct output is verified and skipped; differing output must be moved aside before retrying. Missing or damaged chunks cause an error, and incomplete output is removed. Allow approximately 412 MB of additional disk space for restoration.

`verify_submission.py` additionally checks all 16 raw data files (15 gzip archives and one metadata file) against their recorded uncompressed SHA-256 checksums. For an alternative restoration directory, use `python3 merge_data.py --output-dir /path/to/output`; the original verification command expects archives in this repository's default locations.

## Rebuild the chunks

After restoring the archives, run `python3 split_data.py` to regenerate chunks for files larger than 90 MB. Smaller files remain intact. The script verifies the written chunks, saves the manifests, and deletes only the successfully split large originals. Commit the ordinary `.jsonl.gz` files, the `.partNNN` files, and the manifests in the original `raw/` directories. The restored large `github_all/raw/prs.jsonl.gz` is ignored by Git. Chunk files are byte segments and cannot be decompressed individually.
