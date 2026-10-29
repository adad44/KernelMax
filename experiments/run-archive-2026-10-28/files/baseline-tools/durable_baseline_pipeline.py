"""One detached reference session; audit, freeze, audit, then restore services.

No statistical rerolls, gate changes, candidate acceptance or UI suspension.
Designed for a temporary user launchd job, independent of the Codex app/PTY.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

BASE = Path('/Users/alandiaz/Downloads/KernelMaxxxing')
PYTHON = BASE / 'runtime/gpt-oss/.venv/bin/python'
LEASE = BASE / 'baseline-results/background-pause-until-official-2026-10-04.json'


def audit_command(args, destination):
    result = subprocess.run(args, cwd=BASE / 'KernelMax', capture_output=True, text=True, timeout=300)
    destination.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('Verification command failed: ' + ' '.join(map(str, args)))
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if os.getuid() != 501 or args.receipt.exists():
        raise ValueError('Require uid 501 and a fresh pipeline receipt')
    stamp = datetime.datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%dT%H-%M-%S')
    root = BASE / 'baseline-results' / ('official-attempt-' + stamp + '-seed42-paging-aware')
    state = dict(schema='kernelmax_durable_reference_pipeline_v1', status='starting', created_at=time.time(),
                 pid=os.getpid(), parent_pid=os.getppid(), output=str(root), model_loads_expected=1,
                 receipt=str(args.receipt), scope='One complete reference session; no automatic noise retries; original gates unchanged',
                 authorization='approve and continue; stop waiting for approval and do what is needed for the baseline',
                 sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                     Path(__file__), BASE/'baseline-tools/memoryos_baseline_session.py',
                     BASE/'baseline-tools/quiet_checkpoint_controller.py', BASE/'baseline-tools/verify-baseline-evidence.py')})
    def save():
        temporary = args.receipt.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(args.receipt)
    supervisor = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    save()
    try:
        # Existing explicit protocol approval is bound automatically to this
        # fresh run, never synthesized as approval of results or a candidate.
        review = json.loads((BASE/'baseline-tools/official-reference-protocol-review.json').read_text())
        if not review.get('per_request_control_approved') or not review.get('delegated_adjustment_scope'):
            raise ValueError('Required actual protocol approval/delegation missing')
        supervisor = subprocess.Popen([str(PYTHON), '-u', '-B', str(BASE/'baseline-tools/memoryos_baseline_session.py'),
            '--swap-policy', 'paging-aware', '--controller-idle-seconds', '900', '--service-lease', str(LEASE),
            '--display-sleep', '--stability-screen', '--per-request-control', '--session-limit-seconds', '10800',
            '--session-stamp', stamp, '--autonomous-controller'], cwd=BASE)
        state.update(status='collecting', supervisor_pid=supervisor.pid)
        save()
        deadline = time.monotonic() + 60
        raw_path = root/'attempt.json'
        while not raw_path.exists():
            if supervisor.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Worker did not create initial run identity')
            time.sleep(.25)
        raw = json.loads(raw_path.read_text())
        review.update(run_id=raw['run_id'], driver_sha256=raw['driver_sha256'], resident_session_root=str(root),
            environment_control_receipt=str(BASE/'baseline-results'/('memoryos-session-'+stamp+'.json')),
            execution_control='Temporary user launchd job; parent outside Codex/PTY; autonomous controller; no UI-renderer suspension',
            durable_pipeline_receipt=str(args.receipt), ui_renderer_lease=None)
        (root/'human-review.json').write_text(json.dumps(review, indent=2) + '\n')
        state.update(run_id=raw['run_id'], worker_pid=raw['resident_session']['worker_pid'])
        save()
        code = supervisor.wait(timeout=10900)
        state['supervisor_returncode'] = code
        if code:
            raise RuntimeError('Collection failed or interrupted; retained unchanged, no automatic statistical retry')
        for path, expected in state['sources'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise RuntimeError('External collection/control source changed: '+path)
        session = json.loads((root/'resident-session.json').read_text())
        current = Path(session['current_attempt'])
        if current.parent != root.parent or (current != root and not current.name.startswith(root.name+'-retry')):
            raise ValueError('Unexpected resident attempt path')
        state.update(status='auditing_raw', current_attempt=str(current))
        save()
        raw = json.loads((current/'attempt.json').read_text())
        review.update(run_id=raw['run_id'], driver_sha256=raw['driver_sha256'])
        review_path = current/'human-review.json'
        review_path.write_text(json.dumps(review, indent=2) + '\n')
        checker = BASE/'baseline-tools/verify-baseline-evidence.py'
        audit_command([str(PYTHON), '-B', str(checker), '--attempt', str(current/'attempt.json')],
                      current/'independent-audit-raw.json')
        destination = BASE/'KernelMax/reports/baselines/gpt-oss-20b-m4-week1'/raw['run_id']
        state.update(status='freezing', official_destination=str(destination))
        save()
        audit_command([str(PYTHON), '-B', '-m', 'evaluator.baseline_report', '--attempt', str(current/'attempt.json'),
                       '--human-review', str(review_path), '--destination', str(destination)], current/'report-writer.log')
        state['status'] = 'auditing_frozen_report'
        save()
        audit_command([str(PYTHON), '-B', str(checker), '--attempt', str(destination/'raw.json'),
                       '--report', str(destination/'baseline.json')], destination/'independent-audit.json')
        state.update(status='restoring_services', official_verified_at=time.time())
        save()
        audit_command([str(PYTHON), '-B', str(BASE/'baseline-tools/background_service_lease.py'),
                       'restore', '--receipt', str(LEASE)], destination/'background-services-restoration.log')
        observed = json.loads(audit_command([str(PYTHON), '-B', str(BASE/'baseline-tools/background_service_lease.py'),
                       'status', '--receipt', str(LEASE)], destination/'background-services-after.json'))
        if observed['status'] != 'restored' or not all(observed['loaded'].values()):
            raise RuntimeError('Official report passed, but background service restoration is incomplete')
        state.update(status='complete', finished_at=time.time(), background_services_restored=True)
        save()
        (destination/'durable-pipeline.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'official_baseline': str(destination/'README.md'), 'status': 'complete'}), flush=True)
        return 0
    except BaseException as error:
        if supervisor is not None and supervisor.poll() is None:
            supervisor.send_signal(signal.SIGINT)
            try:
                supervisor.wait(timeout=50)
            except subprocess.TimeoutExpired:
                supervisor.terminate()
                supervisor.wait(timeout=20)
        state.update(status='failed', error=type(error).__name__+': '+str(error), finished_at=time.time())
        save()
        print(json.dumps(state), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
