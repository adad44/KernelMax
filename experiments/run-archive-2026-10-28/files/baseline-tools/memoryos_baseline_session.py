"""Approved bounded resident baseline session with reversible MemoryOS pause."""
import datetime
import argparse
import hashlib
import json
import os
import plistlib
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path('/Users/alandiaz/Downloads/KernelMaxxxing')
PYTHON = BASE / 'runtime/gpt-oss/.venv/bin/python'
AGENTS = Path('/Users/alandiaz/Library/LaunchAgents')
LABELS = ('com.memoryos.scheduler', 'com.memoryos.daemon', 'com.memoryos.backend')


def command(args):
    return subprocess.run(args, capture_output=True, text=True, timeout=15)


def wait_unloaded(uid, label, timeout=30):
    deadline = time.monotonic() + timeout
    while command(['/bin/launchctl', 'print', f'gui/{uid}/{label}']).returncode == 0:
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Launch-agent unload did not finish: {label}')
        time.sleep(.25)


def stop_child(child):
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
            child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--swap-policy', choices=('strict', 'paging-aware'), default='strict')
    parser.add_argument('--controller-idle-seconds', type=int, default=180)
    parser.add_argument('--stability-screen', action='store_true')
    parser.add_argument('--service-lease', type=Path)
    parser.add_argument('--quiet-ui', action='store_true')
    parser.add_argument('--display-sleep', action='store_true')
    parser.add_argument('--per-request-control', action='store_true')
    parser.add_argument('--session-limit-seconds', type=int, default=7200)
    parser.add_argument('--session-stamp')
    parser.add_argument('--autonomous-controller', action='store_true')
    options = parser.parse_args()
    if not 180 <= options.controller_idle_seconds <= 900:
        raise ValueError('Controller idle limit must be between 180 and 900 seconds')
    if not 7200 <= options.session_limit_seconds <= 10800:
        raise ValueError('Session limit must be between two and three hours')
    uid = os.getuid()
    if uid != 501:
        raise RuntimeError('This approved session is scoped only to Alan uid 501')
    stamp = datetime.datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%dT%H-%M-%S')
    if options.session_stamp:
        datetime.datetime.strptime(options.session_stamp, '%Y-%m-%dT%H-%M-%S')
        stamp = options.session_stamp
    suffix = '-paging-aware' if options.swap_policy == 'paging-aware' else ''
    output = BASE / 'baseline-results' / f'official-attempt-{stamp}-seed42{suffix}'
    receipt_path = BASE / 'baseline-results' / f'memoryos-session-{stamp}.json'
    log_path = Path('/Users/alandiaz/Library/Logs') / f'kernelmax-official-baseline-{stamp}.log'
    receipt = {'approved_response': 'yea do that, and try for the official baseline again',
               'scope': 'Temporary pause of MemoryOS capture/indexing and scheduler; restore initially loaded jobs on exit. No history deletion.',
               'output': str(output), 'log': str(log_path), 'initially_loaded': [],
               'swap_policy': options.swap_policy,
               'controller_idle_seconds': options.controller_idle_seconds,
               'session_limit_seconds': options.session_limit_seconds,
               'paused': [], 'restored': [], 'restoration_errors': [], 'status': 'preflight'}
    paused, child, caffeinate, controller = [], None, None, None
    controller_log = None
    snapshots = {}
    if options.service_lease:
        lease = json.loads(options.service_lease.read_text())
        if (lease.get('uid') != uid or lease.get('labels') != list(LABELS)
                or lease.get('status') != 'paused_until_official_baseline'):
            raise RuntimeError('Persistent service-pause lease identity mismatch')
        loaded = {label: command(['/bin/launchctl', 'print', f'gui/{uid}/{label}']).returncode == 0 for label in LABELS}
        if any(loaded.values()):
            raise RuntimeError('Approved background-service lease is not currently paused')
        receipt['external_service_lease'] = {'path': str(options.service_lease),
            'sha256': hashlib.sha256(options.service_lease.read_bytes()).hexdigest(),
            'loaded_before_session': loaded,
            'scope': 'Services remain paused across attempts until the official baseline is frozen; this supervisor did not unload or restore them.'}
    def current_output():
        pointer = output / 'resident-session.json'
        if not pointer.exists():
            return output
        current = Path(json.loads(pointer.read_text())['current_attempt'])
        if current.parent != output.parent or (current != output and not current.name.startswith(output.name + '-retry')):
            raise RuntimeError('Unexpected resident attempt path')
        return current
    def save():
        temporary = receipt_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(receipt_path)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        for label in LABELS:
            path = AGENTS / f'{label}.plist'
            payload = path.read_bytes()
            if plistlib.loads(payload).get('Label') != label:
                raise RuntimeError(f'Unexpected launch-agent identity: {path}')
            snapshots[label] = hashlib.sha256(payload).hexdigest()
            loaded = command(['/bin/launchctl', 'print', f'gui/{uid}/{label}'])
            if loaded.returncode == 0:
                match = re.search(r'\bpid = (\d+)', loaded.stdout)
                receipt['initially_loaded'].append({'label': label, 'plist_sha256': snapshots[label],
                                                    'pid': int(match.group(1)) if match else None})
        save()
        for entry in receipt['initially_loaded']:
            label = entry['label']
            result = command(['/bin/launchctl', 'bootout', f'gui/{uid}/{label}'])
            if result.returncode:
                raise RuntimeError(f'Could not temporarily pause {label}: {result.stderr.strip()}')
            paused.append(label)
            receipt['paused'].append({'label': label, 'at': time.time()})
            save()
            wait_unloaded(uid, label)
            print(f'Temporarily paused {label}', flush=True)
        receipt['status'] = 'benchmark_running'
        if options.quiet_ui:
            receipt['ui_control'] = {'mode': 'hide ChatGPT/Codex and Terminal; do not quit apps', 'results': []}
            for name in ('ChatGPT', 'Terminal'):
                result = command(['/usr/bin/osascript', '-e',
                    f'tell application "System Events" to set visible of process "{name}" to false'])
                receipt['ui_control']['results'].append({'process': name, 'returncode': result.returncode, 'error': result.stderr.strip()})
                if result.returncode:
                    raise RuntimeError(f'Could not hide {name}: {result.stderr.strip()}')
        if options.display_sleep:
            result = command(['/usr/bin/pmset', 'displaysleepnow'])
            receipt['display_control'] = {'action': 'display sleep only; system kept awake by caffeinate; no permanent power-setting change',
                'returncode': result.returncode, 'error': result.stderr.strip()}
            if result.returncode:
                raise RuntimeError('Could not put the display to sleep')
        save()
        print(f'Attempt: {output}\nService receipt: {receipt_path}\nLog: {log_path}', flush=True)
        args = [str(PYTHON), '-u', '-B', '-m', 'evaluator.native_baseline',
                '--model', '/Volumes/BankOfSouls/models/gpt-oss-20b',
                '--verification', str(BASE / 'baseline-tools/native-checkpoint-verification.json'),
                '--output', str(output), '--settle-seconds', '60', '--validate-reference',
                '--paired-aa', '--controlled', '--rehash-checkpoint', '--resident-retries', '2',
                '--swap-policy', options.swap_policy, '--dense-diagnostic',
                str(BASE / 'baseline-results/official-attempt-2026-10-03T17-08-11-seed42/attempt.json')]
        if options.stability_screen:
            args.append('--stability-screen')
        if options.per_request_control:
            args.append('--per-request-control')
        child = subprocess.Popen(args, cwd=BASE / 'KernelMax', stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        receipt['benchmark_pid'] = child.pid
        save()
        if options.autonomous_controller:
            controller_log_path = output.parent / (output.name + '-controller.log')
            controller_log = controller_log_path.open('x')
            controller = subprocess.Popen([str(PYTHON), '-u', '-B',
                str(BASE / 'baseline-tools/quiet_checkpoint_controller.py'), '--output', str(output),
                '--worker-pid', str(child.pid), '--session-limit-seconds', str(options.session_limit_seconds),
                '--no-ui-control'], cwd=BASE, stdout=controller_log, stderr=subprocess.STDOUT)
            receipt['autonomous_controller'] = {'pid': controller.pid, 'log': str(controller_log_path),
                'scope': 'Detached local control; display sleep only, no UI suspension or AppleEvents'}
            save()
        caffeinate = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(child.pid)])
        def stream():
            with log_path.open('w') as log:
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
        reader = threading.Thread(target=stream, daemon=True)
        reader.start()
        started = time.monotonic()
        paused_token, paused_since = None, None
        while child.poll() is None:
            if controller is not None and controller.poll() is not None:
                gate = json.loads((current_output() / 'gate.json').read_text()) if (current_output() / 'gate.json').exists() else {}
                if gate.get('status') not in ('completed_pending_review', 'failed', 'interrupted'):
                    raise RuntimeError('Autonomous controller exited before a terminal worker state')
            gate_path = current_output() / 'gate.json'
            if gate_path.exists():
                try:
                    gate = json.loads(gate_path.read_text())
                except json.JSONDecodeError:
                    gate = {}
                if gate.get('status') == 'paused':
                    if gate.get('token') != paused_token:
                        paused_token, paused_since = gate.get('token'), time.monotonic()
                    if time.monotonic() - paused_since > options.controller_idle_seconds:
                        raise RuntimeError(f'Controller checkpoint idle for {options.controller_idle_seconds} seconds; stopping retry and restoring MemoryOS')
                else:
                    paused_token, paused_since = None, None
            if time.monotonic() - started > options.session_limit_seconds:
                raise RuntimeError('Bounded session limit reached; stopping the worker')
            time.sleep(2)
        reader.join(timeout=5)
        receipt['benchmark_returncode'] = child.returncode
        receipt['current_attempt'] = str(current_output())
        receipt['status'] = 'benchmark_finished'
        save()
    except BaseException as error:
        receipt['status'] = 'session_error'
        receipt['error'] = f'{type(error).__name__}: {error}'
        save()
        print(receipt['error'], flush=True)
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        try:
            stop_child(child)
            if controller is not None:
                try:
                    controller.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    stop_child(controller)
                receipt['autonomous_controller']['returncode'] = controller.returncode
            if controller_log is not None:
                controller_log.close()
            if caffeinate is not None and caffeinate.poll() is None:
                caffeinate.terminate()
                try:
                    caffeinate.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    caffeinate.kill()
                    caffeinate.wait()
        except BaseException as error:
            receipt['shutdown_error'] = str(error)
        for label in reversed(paused):
            try:
                path = AGENTS / f'{label}.plist'
                if hashlib.sha256(path.read_bytes()).hexdigest() != snapshots[label]:
                    raise RuntimeError('Launch-agent file changed; refusing a different configuration')
                # A failed pause may still be completing asynchronously.
                if receipt.get('status') == 'session_error' and child is None:
                    wait_unloaded(uid, label)
                if command(['/bin/launchctl', 'print', f'gui/{uid}/{label}']).returncode != 0:
                    result = command(['/bin/launchctl', 'bootstrap', f'gui/{uid}', str(path)])
                    if result.returncode:
                        raise RuntimeError(result.stderr.strip())
                started_job = command(['/bin/launchctl', 'kickstart', f'gui/{uid}/{label}'])
                if started_job.returncode:
                    raise RuntimeError(started_job.stderr.strip())
                if command(['/bin/launchctl', 'print', f'gui/{uid}/{label}']).returncode:
                    raise RuntimeError('Restored service not registered')
                receipt['restored'].append({'label': label, 'at': time.time()})
                print(f'Restored {label}', flush=True)
            except BaseException as error:
                receipt['restoration_errors'].append({'label': label, 'error': str(error)})
            save()
        receipt['finished_at'] = time.time()
        if options.service_lease:
            receipt['external_service_lease']['loaded_after_session'] = {
                label: command(['/bin/launchctl', 'print', f'gui/{uid}/{label}']).returncode == 0 for label in LABELS}
            receipt['external_service_lease']['unchanged'] = hashlib.sha256(options.service_lease.read_bytes()).hexdigest() == receipt['external_service_lease']['sha256']
        receipt['restoration_complete'] = not receipt['restoration_errors'] and len(receipt['restored']) == len(paused)
        save()
        if output.exists():
            (output / 'memoryos-session.json').write_bytes(receipt_path.read_bytes())
            current = current_output()
            if current != output:
                (current / 'memoryos-session.json').write_bytes(receipt_path.read_bytes())
        print(f"Session ended; restoration complete: {receipt['restoration_complete']}", flush=True)
    return 0 if child is not None and child.returncode == 0 and receipt['restoration_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
