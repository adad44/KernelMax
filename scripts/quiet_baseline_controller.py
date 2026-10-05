"""Bounded checkpoint acknowledgements; no inference, filtering, or app suspension."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def run(command):
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=65)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout.strip()


def acknowledge(gate, output, worker_pid, sleep_display=False):
    """Equal 12-second untimed hold for every lane; evaluator settles afterward."""
    if gate.get('status') != 'paused' or not gate.get('token'):
        raise ValueError('Only an evaluator-owned paused checkpoint may be acknowledged')
    time.sleep(12)
    if sleep_display:
        run(['/usr/bin/pmset', 'displaysleepnow'])
    receipt = output.parent / (output.name + '-local-controller.jsonl')
    with receipt.open('a') as stream:
        stream.write(json.dumps({'token': gate['token'], 'at': time.time(),
            'controller': 'local; display sleep; no AppleEvents' if sleep_display else 'local; no display or app control',
            'worker_pid': worker_pid, 'display_sleep': sleep_display}) + '\n')
    return ['--resume', gate['token']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-pid', type=int, required=True)
    parser.add_argument('--session-limit-seconds', type=int, default=10800)
    parser.add_argument('--sleep-display', action='store_true', help='Explicitly permit display sleep, never app suspension')
    args = parser.parse_args()
    if args.worker_pid <= 0 or not 60 <= args.session_limit_seconds <= 10800:
        raise ValueError('Require a positive worker PID and a session bound up to three hours')
    started = time.monotonic()
    gate = {'status': 'waiting_for_worker'}
    while True:
        if time.monotonic() - started > args.session_limit_seconds:
            raise RuntimeError('Controller session bound reached; no automatic restart')
        try:
            os.kill(args.worker_pid, 0)
        except ProcessLookupError:
            gate = json.loads(run([sys.executable, '-B', '-m', 'evaluator.baseline_control',
                '--output', str(args.output), '--wait-seconds', '0']))
            return 0 if gate['status'] == 'completed_pending_review' else 1
        command = [sys.executable, '-B', '-m', 'evaluator.baseline_control',
            '--output', str(args.output), '--wait-seconds', '45']
        if gate['status'] == 'paused':
            print(json.dumps({'event': 'untimed_checkpoint', 'gate': gate}), flush=True)
            command += acknowledge(gate, args.output, args.worker_pid, args.sleep_display)
        gate = json.loads(run(command))
        if gate['status'] not in ('paused', 'waiting_for_worker', 'running'):
            print(json.dumps({'event': 'terminal', 'gate': gate}), flush=True)
            return 0 if gate['status'] == 'completed_pending_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
