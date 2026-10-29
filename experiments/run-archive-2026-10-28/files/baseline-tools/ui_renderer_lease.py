"""Bounded, reversible pause of exact same-user Codex UI renderers only.

Never pause the app main/service, CLI, benchmark worker, or WindowServer.
SIGINT/TERM/HUP and worker exit resume the exact original processes. PID reuse
is checked by uid, launch time and executable before every signal.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

PREFIX = '/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/Versions/'
SUFFIX = '/Helpers/Codex (Renderer).app/Contents/MacOS/Codex (Renderer)'


def identity(pid):
    result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'uid=,lstart=,comm='],
                            capture_output=True, text=True, timeout=5)
    parts = result.stdout.strip().split(None, 6)
    if result.returncode or len(parts) != 7:
        return None
    return {'pid': pid, 'uid': int(parts[0]), 'launched': ' '.join(parts[1:6]), 'executable': parts[6]}


def renderer(entry):
    return (entry and entry['uid'] == os.getuid() == 501
            and entry['executable'].startswith(PREFIX)
            and entry['executable'].endswith(SUFFIX))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--pid', type=int, action='append', required=True)
    parser.add_argument('--worker-pid', type=int)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 10800 or len(set(args.pid)) != len(args.pid):
        raise ValueError('Invalid bound or duplicate target')
    if args.receipt.exists():
        raise ValueError('Fresh receipt required; preserve prior evidence')
    entries = [identity(pid) for pid in args.pid]
    if not all(renderer(entry) for entry in entries):
        raise ValueError('Only verified uid-501 Codex UI renderer executables may be paused')
    for entry in entries:
        state = subprocess.check_output(['/bin/ps', '-p', str(entry['pid']), '-o', 'stat='], text=True).strip()
        if 'T' in state:
            raise ValueError('Do not take ownership of a process already stopped')
    worker = identity(args.worker_pid) if args.worker_pid else None
    if args.worker_pid and (not worker or worker['uid'] != 501 or worker['pid'] in args.pid):
        raise ValueError('Worker identity unavailable or overlaps UI targets')
    receipt = {'schema': 'kernelmax_ui_renderer_lease_v1', 'targets': entries,
               'worker': worker, 'max_seconds': args.max_seconds,
               'authorization': 'approve and continue; stop waiting for my approval and do what is needed for the baseline',
               'scope': 'UI renderers only; reversible SIGSTOP/SIGCONT, no deletion, no main/service/CLI suspension',
               'created_at': time.time(), 'status': 'preflight', 'paused': [], 'resumed': [], 'errors': []}
    def save():
        temporary = args.receipt.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(args.receipt)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    start = time.monotonic()
    try:
        save()
        for entry in entries:
            if identity(entry['pid']) != entry:
                raise RuntimeError('Renderer identity changed before pause')
            os.kill(entry['pid'], signal.SIGSTOP)
            receipt['paused'].append({'pid': entry['pid'], 'at': time.time()})
            save()
        receipt['status'] = 'paused'
        receipt['paused_at'] = time.time()
        save()
        print(json.dumps({'event': 'ui_renderers_paused', 'pids': args.pid, 'receipt': str(args.receipt)}), flush=True)
        while time.monotonic() - start < args.max_seconds:
            if worker and identity(worker['pid']) != worker:
                receipt['end_reason'] = 'worker_exited'
                break
            if any(identity(entry['pid']) != entry for entry in entries):
                raise RuntimeError('Renderer disappeared or changed identity while paused')
            for entry in entries:
                state = subprocess.check_output(['/bin/ps', '-p', str(entry['pid']), '-o', 'stat='], text=True).strip()
                if 'T' not in state:
                    raise RuntimeError('Renderer unexpectedly resumed during controlled session')
            time.sleep(5)
        else:
            receipt['end_reason'] = 'bounded_timeout'
    except BaseException as error:
        receipt['errors'].append(type(error).__name__ + ': ' + str(error))
    finally:
        try:
            if worker and identity(worker['pid']) == worker:
                # Losing the control must not leave an uncontrolled collection.
                os.kill(worker['pid'], signal.SIGINT)
        except ProcessLookupError:
            pass
        except (OSError, subprocess.SubprocessError) as error:
            receipt['errors'].append('Worker stop: ' + str(error))
        for entry in entries:
            if any(x['pid'] == entry['pid'] for x in receipt['paused']):
                try:
                    current = identity(entry['pid'])
                    if current == entry:
                        os.kill(entry['pid'], signal.SIGCONT)
                        receipt['resumed'].append({'pid': entry['pid'], 'at': time.time()})
                    elif current is not None:
                        receipt['errors'].append('PID reused; no signal sent: ' + str(entry['pid']))
                except (OSError, subprocess.SubprocessError) as error:
                    receipt['errors'].append('Renderer restoration: ' + str(error))
        receipt['finished_at'] = time.time()
        receipt['status'] = 'restored' if not receipt['errors'] else 'ended_with_errors'
        save()
        print(json.dumps({'event': 'ui_renderers_resumed', 'receipt': str(args.receipt), 'status': receipt['status']}), flush=True)
    return 0 if receipt['status'] == 'restored' else 1


if __name__ == '__main__':
    raise SystemExit(main())
