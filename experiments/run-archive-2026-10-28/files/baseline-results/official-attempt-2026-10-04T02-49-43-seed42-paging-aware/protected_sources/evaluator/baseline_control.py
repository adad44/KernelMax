"""Coordinate the agent only at evaluator-owned, untimed checkpoints."""
import argparse
import json
import time
from pathlib import Path


def current_gate(root):
    """Follow a resident session without treating a retained failure as its end."""
    pointer = root / 'resident-session.json'
    session = json.loads(pointer.read_text()) if pointer.exists() else None
    output = Path(session['current_attempt']) if session else root
    if session and (output.parent != root.parent or
                    (output != root and not output.name.startswith(root.name + '-retry'))):
        raise ValueError('Unexpected resident attempt path')
    path = output / 'gate.json'
    gate = json.loads(path.read_text()) if path.exists() else {'status': 'waiting_for_worker'}
    if session:
        if session['status'] == 'running' and gate['status'] in ('failed', 'interrupted'):
            gate = {'status': 'waiting_for_worker', 'phase': session['phase']}
        elif session['status'] != 'running':
            gate = {'status': session['status'], 'errors': session.get('errors', [])}
    return output, dict(gate, output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume')
    parser.add_argument('--wait-seconds', type=float, default=50)
    args = parser.parse_args()
    if args.resume:
        output, gate = current_gate(args.output)
        if gate.get('status') != 'paused' or gate.get('token') != args.resume:
            raise ValueError('Refusing stale checkpoint acknowledgement')
        pending = output / 'resume.json.tmp'
        pending.write_text(json.dumps({'token': args.resume, 'acknowledged_at': time.time(),
                                      'controller': 'Codex; untimed checkpoint; then waiting only'}))
        pending.replace(output / 'resume.json')
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        try:
            _, gate = current_gate(args.output)
        except json.JSONDecodeError:
            time.sleep(.5)
            continue
        if gate.get('status') not in ('running', 'waiting_for_worker') and (gate.get('status') != 'paused' or gate.get('token') != args.resume):
            print(json.dumps(gate), flush=True)
            return
        time.sleep(.5)
    print(json.dumps({'status': 'waiting_for_worker'}), flush=True)


if __name__ == '__main__':
    main()
