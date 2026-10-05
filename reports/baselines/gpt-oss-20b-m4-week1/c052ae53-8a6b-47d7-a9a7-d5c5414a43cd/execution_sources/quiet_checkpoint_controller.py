"""Bounded local checkpoint control, independent of language-model latency.

No model execution, source edits, result filtering or automatic failed-run restart.
Only evaluator-owned paused checkpoints authorize display control/acknowledgement.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

PYTHON = '/Users/alandiaz/Downloads/KernelMaxxxing/runtime/gpt-oss/.venv/bin/python'
REPO = '/Users/alandiaz/Downloads/KernelMaxxxing/KernelMax'


def run(args):
    result = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=65)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-pid', type=int, required=True)
    parser.add_argument('--session-limit-seconds', type=int, default=7200)
    parser.add_argument('--no-ui-control', action='store_true', help='No AppleEvents; sleep display only for detached jobs')
    args = parser.parse_args()
    receipts = args.output.parent / (args.output.name + '-local-controller.jsonl')
    start = time.monotonic()
    gate = {'status': 'waiting_for_worker'}
    while True:
        if time.monotonic() - start > args.session_limit_seconds:
            raise RuntimeError('Local-controller session bound reached; no automatic restart')
        try:
            os.kill(args.worker_pid, 0)
        except ProcessLookupError:
            gate = json.loads(run([PYTHON, '-B', '-m', 'evaluator.baseline_control', '--output', str(args.output), '--wait-seconds', '0']))
            print(json.dumps({'event': 'terminal', 'gate': gate}), flush=True)
            return 0 if gate['status'] == 'completed_pending_review' else 1
        command = [PYTHON, '-B', '-m', 'evaluator.baseline_control', '--output', str(args.output), '--wait-seconds', '45']
        if gate['status'] == 'paused':
            print(json.dumps({'event': 'untimed_checkpoint', 'gate': gate}), flush=True)
            # Allow the observer's short progress update to finish before acknowledgement.
            # No observer response is needed: a delayed/compacted chat cannot kill the run.
            time.sleep(12)
            if not args.no_ui_control:
                run(['/usr/bin/osascript', '-e', 'tell application "System Events" to set visible of process "ChatGPT" to false',
                     '-e', 'tell application "System Events" to set visible of process "Terminal" to false'])
            asleep = run(['/usr/bin/pmset', 'displaysleepnow'])
            with receipts.open('a') as log:
                log.write(json.dumps({'token': gate['token'], 'at': time.time(),
                    'controller': 'local; display sleep; no AppleEvents' if args.no_ui_control else 'local; hidden UI/display sleep; then waiting only',
                    'ui_commands_succeeded': True, 'worker_pid': args.worker_pid}) + '\n')
            command += ['--resume', gate['token']]
        gate = json.loads(run(command))
        if gate['status'] not in ('paused', 'waiting_for_worker', 'running'):
            print(json.dumps({'event': 'terminal', 'gate': gate}), flush=True)
            return 0 if gate['status'] == 'completed_pending_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
