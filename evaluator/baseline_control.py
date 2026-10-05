"""Coordinate the agent only at evaluator-owned, untimed checkpoints."""
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


def resume_checkpoint(root, token):
    """Atomically acknowledge only the current evaluator checkpoint."""
    output, gate = current_gate(root)
    if gate.get('status') != 'paused' or gate.get('token') != token:
        raise ValueError('Refusing stale checkpoint acknowledgement')
    pending = output / 'resume.json.tmp'
    pending.write_text(json.dumps({'token': token, 'acknowledged_at': time.time(),
                                  'controller': 'local; untimed checkpoint'}))
    pending.replace(output / 'resume.json')
