import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from python.export_replay import export_replay, sha256
from python.tests.test_export_contract import book_sequence


class ReplayExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.a = book_sequence(8)
        self.b = book_sequence(7)
        self.b.timestamp_ms += 5000
        self.b.last_update_id += 8
        self.manifest = self.root / 'manifest.json'
        self.output = self.root / 'out.csv'
        self.write_manifest()

    def write_manifest(self):
        items = []
        for name, frame in [('a.parquet', self.a), ('b.parquet', self.b)]:
            path = self.root / name
            frame.to_parquet(path, index=False)
            items.append({'path': name, 'rows': len(frame), 'sha256': sha256(path)})
        self.manifest.write_text(json.dumps({'files': items, 'max_gap_ms': 1000}))

    def test_streaming_batches_preserve_precision_and_ids(self):
        result = export_replay(self.manifest, self.output, self.root, batch_size=3)
        self.assertEqual(result['rows'], 15)
        converters = {c: float for c in self.a if c not in ['timestamp_ms', 'last_update_id']}
        actual = pd.read_csv(self.output, converters=converters)
        self.assertEqual(actual.row_id.tolist(), list(range(15)))
        expected = pd.concat([self.a, self.b], ignore_index=True)
        pd.testing.assert_frame_equal(actual.drop(columns='row_id'), expected, check_dtype=False)
        self.assertEqual(sha256(self.output), result['output_sha256'])

    def test_existing_source_and_output_protected(self):
        self.output.write_text('keep')
        with self.assertRaises(FileExistsError):
            export_replay(self.manifest, self.output, self.root)
        self.assertEqual(self.output.read_text(), 'keep')
        with self.assertRaises(ValueError):
            export_replay(self.manifest, self.manifest, self.root)

    def test_corrupt_or_out_of_order_sources_leave_no_output(self):
        for kind in ['hash', 'order', 'rows']:
            with self.subTest(kind=kind):
                self.write_manifest()
                if kind == 'hash':
                    with (self.root / 'b.parquet').open('ab') as stream:
                        stream.write(b'changed')
                elif kind == 'order':
                    self.b.last_update_id -= 8
                    self.write_manifest()
                    self.b.last_update_id += 8
                else:
                    manifest = json.loads(self.manifest.read_text())
                    manifest['files'][1]['rows'] += 1
                    self.manifest.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    export_replay(self.manifest, self.output, self.root, batch_size=3)
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.root.glob('.replay-*')), [])

    def test_unsafe_paths_and_invalid_batch_size(self):
        manifest = json.loads(self.manifest.read_text())
        manifest['files'][0]['path'] = '../a.parquet'
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            export_replay(self.manifest, self.output, self.root)
        with self.assertRaises(ValueError):
            export_replay(self.manifest, self.output, self.root, batch_size=0)


if __name__ == '__main__':
    unittest.main()
