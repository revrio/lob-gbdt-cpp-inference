"""Executable-level integration checks, requiring only Python's standard library."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

parser = argparse.ArgumentParser()
parser.add_argument('--exe', type=Path, required=True)
parser.add_argument('--contract', type=Path, required=True)
args, remaining = parser.parse_known_args()
EXE, CONTRACT = args.exe.resolve(), args.contract.resolve()
sys.argv = [sys.argv[0], *remaining]


class ReplayCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.input = CONTRACT / 'golden_input_lob.csv'
        self.tau = json.loads((CONTRACT / 'metadata.json').read_text())['conviction_tau']

    def run_replay(self, *extra, input_path=None, success=True):
        command = [str(EXE), '--model', str(CONTRACT / 'model.txt'), '--input',
                   str(input_path or self.input), '--tau', str(self.tau), *map(str, extra)]
        result = subprocess.run(command, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result

    def assert_no_staging(self):
        self.assertEqual(list(self.root.glob('.lob-replay-*')), [])

    def test_golden_predictions_and_summary(self):
        output, summary = self.root / 'predictions.csv', self.root / 'summary.json'
        stats = self.run_replay('--output', output, '--summary', summary)
        self.assertEqual(stats, json.loads(summary.read_text()))
        self.assertEqual((stats['snapshots'], stats['warmup_rows'], stats['predictions']), (5000,300,4700))
        self.assertEqual(stats['gap_resets'], 0)
        with output.open() as f:
            actual = list(csv.DictReader(f))
        with (CONTRACT / 'golden_predictions.csv').open() as f:
            expected = list(csv.DictReader(f))
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertEqual(a['row_id'], b['row_id'])
            self.assertEqual(a['class_id'], b['class_id'])
            for key in ['raw_down','raw_flat','raw_up','p_down','p_flat','p_up']:
                self.assertTrue(math.isfinite(float(a[key])))
                self.assertLess(abs(float(a[key])-float(b[key])), 1e-6)
        self.assertEqual(sum(stats['classes_down_flat_up']), 4700)
        self.assert_no_staging()

    def test_no_output_and_profile_agree(self):
        a = self.run_replay('--no-output')
        b = self.run_replay('--no-output', '--profile')
        self.assertEqual(a['checksum'], b['checksum'])
        self.assertEqual(a['profile_stage_ns'], dict.fromkeys(['parse_validate','features','inference','output'], 0))
        self.assertGreater(b['profile_stage_ns']['inference'], 0)
        self.assertGreater(b['profile_stage_ns']['parse_validate'], 0)

    def test_gap_reset_and_exact_boundary(self):
        with self.input.open() as f:
            rows = list(csv.reader(f))
        base = rows[1]
        data = []
        for i in range(641):
            row = base.copy()
            # Interval at row 320 is exactly 1000ms; row 340 is 1001ms.
            t = i*100 + (900 if i>=320 else 0) + (901 if i>=340 else 0)
            row[:3] = [str(i), str(t), str(i+1)]
            data.append(row)
        path = self.root / 'gaps.csv'
        with path.open('w', newline='') as f:
            writer = csv.writer(f); writer.writerow(rows[0]); writer.writerows(data)
        stats = self.run_replay('--no-output', input_path=path)
        self.assertEqual(stats['gap_resets'], 1)
        self.assertEqual(stats['warmup_rows'], 600)
        self.assertEqual(stats['predictions'], 41)

    def test_partial_invalid_input_never_publishes_output(self):
        path = self.root / 'broken.csv'
        with self.input.open() as f:
            first = [next(f) for _ in range(302)]
        path.write_text(''.join(first)+'bad row\n')
        output = self.root / 'must-not-exist.csv'
        error = self.run_replay('--output', output, input_path=path, success=False)
        self.assertIn('broken.csv:303:', error.stderr)
        self.assertFalse(output.exists())
        self.assert_no_staging()

    def test_existing_outputs_and_inputs_are_protected(self):
        path = self.root / 'existing.csv';path.write_text('keep me')
        self.run_replay('--output', path, success=False)
        self.assertEqual(path.read_text(), 'keep me')
        before = self.input.read_bytes()
        self.run_replay('--output', self.input, success=False)
        self.assertEqual(self.input.read_bytes(), before)
        self.assert_no_staging()

    def test_invalid_arguments_missing_input_and_output_failure(self):
        self.run_replay('--no-output', '--output', self.root/'x', success=False)
        self.run_replay('--no-output', '--tau', 'nan', success=False)
        self.run_replay('--no-output', input_path=self.root/'missing.csv', success=False)
        self.run_replay('--output', self.root/'missing-parent'/'out.csv', success=False)
        self.run_replay('--no-output', '--unknown', 'value', success=False)
        result = subprocess.run([str(EXE), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('Usage:', result.stdout)
        self.assert_no_staging()

    def test_header_only_and_short_warmup_stream(self):
        with self.input.open() as f:
            lines = [next(f) for _ in range(101)]
        path = self.root/'short.csv'
        path.write_text(lines[0])
        self.assertEqual(self.run_replay('--no-output', input_path=path)['snapshots'], 0)
        path.write_text(''.join(lines))
        s = self.run_replay('--no-output', input_path=path)
        self.assertEqual((s['snapshots'],s['warmup_rows'],s['predictions']), (100,100,0))


if __name__ == '__main__':
    unittest.main()
