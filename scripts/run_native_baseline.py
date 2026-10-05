"""Bounded collection supervisor. Never approves results or controls background services."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def worker_command(model, verification, output, policy, diagnostic=None):
    command = [sys.executable, '-u', '-B', '-m', 'evaluator.native_baseline',
        '--model', str(model), '--verification', str(verification), '--output', str(output),
        '--settle-seconds', '60', '--validate-reference', '--paired-aa', '--controlled',
        '--rehash-checkpoint', '--resident-retries', '2', '--swap-policy', policy,
        '--per-request-control']
    if diagnostic is not None:
        command += ['--dense-diagnostic', str(diagnostic)]
    return command


def stop(child):
    if child is None or child.poll() is not None:
        return
    child.send_signal(signal.SIGINT)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def supervise(worker, controller, output, limit=10800, idle_limit=900):
    from evaluator.baseline_control import current_gate
    start = time.monotonic()
    paused_token, paused_at = None, None
    while worker.poll() is None:
        try:
            _, gate = current_gate(output)
        except json.JSONDecodeError:
            gate = {'status': 'running'}  # Atomic attempt writes; gate writes can overlap reads.
        if controller.poll() is not None and gate['status'] not in ('completed_pending_review', 'failed', 'interrupted'):
            raise RuntimeError('Controller exited before a terminal worker state')
        if gate['status'] == 'paused':
            if gate['token'] != paused_token:
                paused_token, paused_at = gate['token'], time.monotonic()
            if time.monotonic() - paused_at > idle_limit:
                raise RuntimeError('Checkpoint idle limit exceeded; retain attempt, no reroll')
        else:
            paused_token, paused_at = None, None
        if time.monotonic() - start > limit:
            raise RuntimeError('Session time limit exceeded; retain attempt, no reroll')
        time.sleep(2)
    return worker.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--swap-policy', choices=('strict', 'paging-aware'), default='strict')
    parser.add_argument('--dense-diagnostic', type=Path)
    parser.add_argument('--environment-ready', action='store_true', required=True,
        help='Caller confirms AC power and approved background-work controls; no services are stopped here')
    parser.add_argument('--sleep-display', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Use a fresh output directory; no existing attempt is overwritten')
    receipt_path = output.parent / (output.name + '-session.json')
    if any(path.exists() for path in (receipt_path,
            output.parent / (output.name + '-worker.log'),
            output.parent / (output.name + '-controller.log'),
            output.parent / (output.name + '-local-controller.jsonl'))):
        raise ValueError('Sibling receipts/logs already exist; choose a fresh session name')
    model, verification = args.model.resolve(strict=True), args.verification.resolve(strict=True)
    diagnostic = args.dense_diagnostic.resolve(strict=True) if args.dense_diagnostic else None
    output.parent.mkdir(parents=True, exist_ok=True)
    worker = controller = awake = None
    receipt = {'scope': 'Collection only; caller-managed environment; no services paused or restored',
        'status': 'starting', 'output': str(output), 'environment_ready_asserted_by_caller': True,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__), REPO/'scripts/quiet_baseline_controller.py', REPO/'scripts/audit_baseline.py')}}
    def save():
        temporary = receipt_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(receipt_path)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        with (output.parent / (output.name + '-worker.log')).open('x') as worker_log, \
             (output.parent / (output.name + '-controller.log')).open('x') as controller_log:
            worker = subprocess.Popen(worker_command(model, verification, output, args.swap_policy, diagnostic),
                cwd=REPO, stdout=worker_log, stderr=subprocess.STDOUT)
            command = [sys.executable, '-u', '-B', str(REPO/'scripts/quiet_baseline_controller.py'),
                '--output', str(output), '--worker-pid', str(worker.pid)]
            if args.sleep_display:
                command.append('--sleep-display')
            controller = subprocess.Popen(command, cwd=REPO, stdout=controller_log, stderr=subprocess.STDOUT)
            awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(worker.pid)])
            receipt.update(status='collecting', benchmark_pid=worker.pid, controller_pid=controller.pid)
            save()
            code = supervise(worker, controller, output)
            controller_code = controller.wait(timeout=65)
            for path, expected in receipt['source_sha256'].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                    raise RuntimeError('External collection/control source changed: ' + path)
            receipt.update(status='benchmark_finished', benchmark_returncode=code, controller_returncode=controller_code)
            if code or controller_code:
                raise RuntimeError('Collection/controller failed; retain evidence, no automatic restart')
    except BaseException as error:
        receipt.update(status='session_error', error=f'{type(error).__name__}: {error}')
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        shutdown_errors = []
        for child in (worker, controller, awake):
            try:
                stop(child)
            except BaseException as error:
                shutdown_errors.append(f'{type(error).__name__}: {error}')
        if shutdown_errors:
            receipt.update(status='session_error', shutdown_errors=shutdown_errors)
        receipt.update(finished_at=time.time(), shutdown_complete=not shutdown_errors)
        save()
        if output.exists():
            (output/'collection-session.json').write_text(json.dumps(receipt, indent=2) + '\n')
            pointer = output/'resident-session.json'
            if pointer.exists():
                from evaluator.baseline_control import current_gate
                current, _ = current_gate(output)
                if current != output:
                    (current/'collection-session.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
    return 0 if receipt['status'] == 'benchmark_finished' else 1


if __name__ == '__main__':
    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
