"""Bounded checkpoint acknowledgements; no inference, filtering, or app suspension."""
import argparse
import json
import os
from pathlib import Path
import subprocess
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
    from evaluator.baseline_control import current_gate, resume_checkpoint
    resumed_token = None
    while True:
        if time.monotonic() - started > args.session_limit_seconds:
            raise RuntimeError('Controller session bound reached; no automatic restart')
        try:
            _, gate = current_gate(args.output)
        except json.JSONDecodeError:
            time.sleep(.5)
            continue
        try:
            os.kill(args.worker_pid, 0)
        except ProcessLookupError:
            return 0 if gate['status'] == 'completed_pending_review' else 1
        if gate['status'] == 'paused' and gate['token'] != resumed_token:
            print(json.dumps({'event': 'untimed_checkpoint', 'gate': gate}), flush=True)
            acknowledge(gate, args.output, args.worker_pid, args.sleep_display)
            resume_checkpoint(args.output, gate['token'])
            resumed_token = gate['token']
        if gate['status'] not in ('paused', 'waiting_for_worker', 'running'):
            print(json.dumps({'event': 'terminal', 'gate': gate}), flush=True)
            return 0 if gate['status'] == 'completed_pending_review' else 1
        time.sleep(.5)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
