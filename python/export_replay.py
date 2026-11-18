"""Stream a checksummed Parquet manifest into the raw C++ replay CSV format."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pyarrow.parquet as pq

if __package__:
    from .export_contract import validate_raw
    from .lob_features import RAW_COLS, MAX_GAP_MS
else:
    from export_contract import validate_raw
    from lob_features import RAW_COLS, MAX_GAP_MS

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export_replay(manifest_path, output, root=ROOT, batch_size=10000):
    """Preserve manifest/row order, fail on corruption, publish only on success.

    Requires recordings already ordered across files; never silently sorts them.
    Memory use is bounded by one Arrow/Pandas batch, rather than the dataset.
    """
    manifest_path, output, root = Path(manifest_path), Path(output), Path(root).resolve()
    if not 1 <= batch_size <= 1_000_000:
        raise ValueError('batch_size must be 1..1000000')
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest.get('files'), list) or not manifest['files']:
        raise ValueError('Manifest must have a nonempty files list')
    if manifest.get('max_gap_ms', MAX_GAP_MS) != MAX_GAP_MS:
        raise ValueError('Manifest gap cutoff differs from the feature contract')
    paths, seen = [], set()
    for item in manifest['files']:
        relative = Path(item['path'])
        path = (root / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root) or path in seen:
            raise ValueError(f'Invalid/repeated manifest path: {relative}')
        if output.resolve() in (path, manifest_path.resolve()):
            raise ValueError('Output must not replace a source file or manifest')
        if not isinstance(item['rows'], int) or item['rows'] <= 0:
            raise ValueError('Manifest source row counts must be positive integers')
        seen.add(path)
        paths.append((path, item))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output already exists: {output}; choose a new path')
    temporary = None
    count, previous_ts, previous_id = 0, None, None
    first_ts = last_ts = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=output.parent, prefix='.replay-', suffix='.csv', delete=False) as stream:
            temporary = Path(stream.name)
            for path, item in paths:
                before = path.stat()
                if sha256(path) != item['sha256']:
                    raise ValueError(f'Checksum mismatch: {item["path"]}')
                rows = 0
                for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=RAW_COLS):
                    frame = batch.to_pandas()[RAW_COLS]
                    validate_raw(frame)
                    ts, update_id = int(frame.timestamp_ms.iloc[0]), int(frame.last_update_id.iloc[0])
                    if previous_ts is not None and (ts < previous_ts or update_id <= previous_id):
                        raise ValueError(f'Out-of-order snapshots across batches/files: {item["path"]}')
                    previous_ts = last_ts = int(frame.timestamp_ms.iloc[-1])
                    previous_id = int(frame.last_update_id.iloc[-1])
                    if first_ts is None:
                        first_ts = ts
                    frame.insert(0, 'row_id', range(count, count + len(frame)))
                    frame.to_csv(stream, index=False, header=count == 0, float_format='%.17g', lineterminator='\n')
                    rows += len(frame)
                    count += len(frame)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f'Source changed during export: {item["path"]}')
                if rows != item['rows']:
                    raise ValueError(f'Row count mismatch: {item["path"]}')
        # Refuse to replace an output created while export was running.
        # Hard linking publishes a complete file exclusively on the same filesystem.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'rows': count, 'files': len(paths), 'first_timestamp_ms': first_ts,
            'last_timestamp_ms': last_ts, 'output': str(output),
            'output_sha256': sha256(output),
            'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'batch_size': batch_size, 'max_gap_ms': MAX_GAP_MS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'contract/training_manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT, help='Root for relative manifest paths')
    parser.add_argument('--batch-size', type=int, default=10000)
    args = parser.parse_args()
    try:
        result = export_replay(args.manifest, args.output, args.root, args.batch_size)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Export error: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
