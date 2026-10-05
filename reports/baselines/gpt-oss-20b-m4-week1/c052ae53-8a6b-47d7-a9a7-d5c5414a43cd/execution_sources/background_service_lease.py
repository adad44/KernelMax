"""Keep only the three approved MemoryOS jobs paused across baseline attempts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import subprocess
import time

LABELS = ('com.memoryos.scheduler', 'com.memoryos.daemon', 'com.memoryos.backend')
PLISTS = Path('/Users/alandiaz/Library/LaunchAgents')


def run(*args):
    return subprocess.run(args, text=True, capture_output=True, timeout=15)


def save(path, state):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pause', 'status', 'restore'))
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if os.getuid() != 501:
        raise RuntimeError('This lease is scoped to Alan uid 501')
    if args.receipt.exists():
        state = json.loads(args.receipt.read_text())
        if state['labels'] != list(LABELS) or state['uid'] != 501:
            raise RuntimeError('Lease identity mismatch')
    elif args.action == 'pause':
        state = {'uid': 501, 'labels': list(LABELS), 'created_at': time.time(),
                 'authorization': 'keep the background services down until we have our official baseline',
                 'scope': 'MemoryOS scheduler, capture daemon and backend only; no configuration/history deletion',
                 'status': 'pausing', 'initially_loaded': [], 'paused': [], 'restored': []}
        for label in LABELS:
            path = PLISTS / (label + '.plist')
            payload = path.read_bytes()
            if plistlib.loads(payload).get('Label') != label:
                raise RuntimeError('Launch-agent identity mismatch')
            if run('/bin/launchctl', 'print', 'gui/501/' + label).returncode == 0:
                state['initially_loaded'].append({'label': label, 'plist': str(path),
                    'plist_sha256': hashlib.sha256(payload).hexdigest()})
        save(args.receipt, state)
    else:
        raise RuntimeError('Receipt not found')
    if args.action == 'pause':
        if state['status'] == 'restored':
            raise RuntimeError('Use a fresh receipt for a new pause lease')
        for entry in state['initially_loaded']:
            label = entry['label']
            if hashlib.sha256(Path(entry['plist']).read_bytes()).hexdigest() != entry['plist_sha256']:
                raise RuntimeError('Launch-agent configuration changed')
            if run('/bin/launchctl', 'print', 'gui/501/' + label).returncode == 0:
                result = run('/bin/launchctl', 'bootout', 'gui/501/' + label)
                if result.returncode:
                    raise RuntimeError(result.stderr)
            deadline = time.monotonic() + 30
            while run('/bin/launchctl', 'print', 'gui/501/' + label).returncode == 0:
                if time.monotonic() > deadline:
                    raise RuntimeError('Service did not unload: ' + label)
                time.sleep(.25)
            if label not in state['paused']:
                state['paused'].append(label)
                save(args.receipt, state)
        state['status'] = 'paused_until_official_baseline'
        save(args.receipt, state)
    elif args.action == 'restore':
        for entry in reversed(state['initially_loaded']):
            label = entry['label']
            if hashlib.sha256(Path(entry['plist']).read_bytes()).hexdigest() != entry['plist_sha256']:
                raise RuntimeError('Configuration changed; refusing a different configuration')
            if run('/bin/launchctl', 'print', 'gui/501/' + label).returncode:
                result = run('/bin/launchctl', 'bootstrap', 'gui/501', entry['plist'])
                if result.returncode:
                    raise RuntimeError(result.stderr)
            result = run('/bin/launchctl', 'kickstart', 'gui/501/' + label)
            if result.returncode or run('/bin/launchctl', 'print', 'gui/501/' + label).returncode:
                raise RuntimeError('Service restoration failed: ' + label)
            if label not in state['restored']:
                state['restored'].append(label)
                save(args.receipt, state)
        state.update(status='restored', restored_at=time.time())
        save(args.receipt, state)
    observed = {label: run('/bin/launchctl', 'print', 'gui/501/' + label).returncode == 0
                for label in LABELS}
    print(json.dumps({'receipt': str(args.receipt), 'status': state['status'], 'loaded': observed}))


if __name__ == '__main__':
    main()
