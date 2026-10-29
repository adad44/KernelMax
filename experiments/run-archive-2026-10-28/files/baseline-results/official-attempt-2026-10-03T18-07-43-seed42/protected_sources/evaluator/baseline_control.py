"""Coordinate the agent only at evaluator-owned, untimed checkpoints."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume')
    parser.add_argument('--wait-seconds', type=float, default=50)
    args = parser.parse_args()
    if args.resume:
        gate = json.loads((args.output / 'gate.json').read_text())
        if gate.get('status') != 'paused' or gate.get('token') != args.resume:
            raise ValueError('Refusing stale checkpoint acknowledgement')
        pending = args.output / 'resume.json.tmp'
        pending.write_text(json.dumps({'token': args.resume, 'acknowledged_at': time.time(),
                                      'controller': 'Codex; untimed checkpoint; then waiting only'}))
        pending.replace(args.output / 'resume.json')
    deadline = time.monotonic() + args.wait_seconds
    while time.monotonic() < deadline:
        path = args.output / 'gate.json'
        if path.exists():
            try:
                gate = json.loads(path.read_text())
            except json.JSONDecodeError:
                time.sleep(.5)
                continue
            if gate.get('status') != 'running' and (gate.get('status') != 'paused' or gate.get('token') != args.resume):
                print(json.dumps(gate), flush=True)
                return
        time.sleep(.5)
    print(json.dumps({'status': 'waiting_for_worker'}), flush=True)


if __name__ == '__main__':
    main()
