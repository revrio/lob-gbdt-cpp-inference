"""Executable-level integration checks, requiring only Python's standard library."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import pty
import select
import signal
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

    def command(self, *extra, input_path=None):
        return [str(EXE), '--model', str(CONTRACT / 'model.txt'), '--input',
                str(input_path or self.input), '--tau', str(self.tau), *map(str, extra)]

    def run_replay(self, *extra, input_path=None, success=True):
        result = subprocess.run(self.command(*extra, input_path=input_path),
                                capture_output=True, text=True, timeout=30)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result

    def assert_no_staging(self):
        self.assertEqual(list(self.root.glob('.lob-replay-*')), [])

    def write_timestamps(self, timestamps):
        with self.input.open() as stream:
            reader = csv.reader(stream)
            header, base = next(reader), next(reader)
        path = self.root / 'timestamps.csv'
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for i, timestamp in enumerate(timestamps):
                writer.writerow([str(i), str(timestamp), str(i + 1), *base[3:]])
        return path

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

    def test_watch_preserves_every_prediction_and_keeps_stdout_json(self):
        fast_path = self.root / 'fast.csv'
        watch_path = self.root / 'watch.csv'
        fast = self.run_replay('--output', fast_path)
        result = subprocess.run(self.command('--watch', '--speed', '1000', '--output', watch_path),
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        watch = json.loads(result.stdout)
        self.assertTrue(watch['playback'])
        self.assertEqual(watch['playback_speed'], 1000)
        self.assertFalse(fast['playback'])
        self.assertEqual(fast['pacing_wait_ns'], 0)
        self.assertEqual(fast['display_ns'], 0)
        for key in ['snapshots', 'warmup_rows', 'predictions', 'classes_down_flat_up', 'checksum']:
            self.assertEqual(watch[key], fast[key])
        self.assertEqual(watch_path.read_bytes(), fast_path.read_bytes())
        self.assertIn('WARMUP', result.stderr)
        self.assertIn('READY', result.stderr)
        self.assertNotIn('\033', result.stderr)
        self.assertGreater(watch['display_ns'], 0)
        duration_ns = (watch['last_timestamp_ms'] - watch['first_timestamp_ms']) * 1e6 / 1000
        self.assertGreaterEqual(watch['replay_wall_ns'], duration_ns)

    def test_watch_skips_gap_wait_but_resets_prediction_and_warmup(self):
        path = self.write_timestamps([*range(304), *range(100000, 100004)])
        result = subprocess.run(self.command('--watch', '--speed', '1000', input_path=path),
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        stats = json.loads(result.stdout)
        self.assertEqual((stats['snapshots'], stats['warmup_rows'], stats['predictions']), (308,304,4))
        self.assertEqual(stats['gap_resets'], 1)
        self.assertEqual(stats['gap_waits_skipped'], 1)
        self.assertFalse(stats['output_enabled'])
        self.assertIn('READY', result.stderr)
        last_frame = result.stderr.splitlines()[-1]
        self.assertIn('WARMUP 4/300', last_frame)
        self.assertNotIn('decision=', last_frame)

    def test_watch_preserves_gaps_when_requested(self):
        path = self.write_timestamps([0, 0, 2000])
        stats = self.run_replay('--watch', '--speed', '100', '--keep-gaps', input_path=path)
        self.assertEqual(stats['gap_resets'], 1)
        self.assertEqual(stats['gap_waits_skipped'], 0)
        self.assertTrue(stats['preserve_gap_waits'])
        self.assertGreaterEqual(stats['replay_wall_ns'], 20_000_000)

    def test_watch_interrupt_during_long_gap_cleans_partial_outputs(self):
        path = self.write_timestamps([0, 3_600_000])
        output, summary = self.root / 'partial.csv', self.root / 'partial.json'
        process = subprocess.Popen(self.command('--watch', '--keep-gaps', '--output', output,
                                                 '--summary', summary, input_path=path),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertTrue(select.select([process.stderr], [], [], 5)[0], 'No first playback frame')
            self.assertIn('WARMUP 1/300', process.stderr.readline())
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=3)
            self.assertEqual(process.returncode, 130, stderr)
            self.assertEqual(stdout, '')
            self.assertIn('Playback interrupted', stderr)
            self.assertFalse(output.exists())
            self.assertFalse(summary.exists())
            self.assert_no_staging()
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_invalid_watch_options(self):
        for extra in [('--speed', '10'), ('--keep-gaps',), ('--watch', '--profile'),
                      ('--watch', '--speed', 'nan'), ('--watch', '--speed', '0'),
                      ('--watch', '--speed', '1001')]:
            with self.subTest(extra=extra):
                self.run_replay('--no-output', *extra, success=False)

    def test_watch_restores_real_terminal_on_interrupt(self):
        master, slave = pty.openpty()
        process = subprocess.Popen(self.command('--watch'), stdout=subprocess.PIPE, stderr=slave,
                                   env={**os.environ, 'TERM': 'xterm'})
        os.close(slave)
        transcript = bytearray()
        try:
            self.assertTrue(select.select([master], [], [], 5)[0], 'No interactive playback frame')
            transcript.extend(os.read(master, 65536))
            process.send_signal(signal.SIGINT)
            stdout, _ = process.communicate(timeout=3)
            while True:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break  # A closed PTY reports EIO on Linux.
                if not chunk:
                    break
                transcript.extend(chunk)
            self.assertEqual(process.returncode, 130)
            self.assertEqual(stdout, b'')
            self.assertIn(b'\033[?1049h', transcript)
            self.assertIn(b'\033[?25l', transcript)
            self.assertIn(b'\033[?25h\033[?1049l', transcript)
            self.assertIn(b'Playback interrupted', transcript)
            self.assertLess(transcript.index(b'\033[?25h\033[?1049l'),
                            transcript.index(b'Playback interrupted'))
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            os.close(master)


if __name__ == '__main__':
    unittest.main()
