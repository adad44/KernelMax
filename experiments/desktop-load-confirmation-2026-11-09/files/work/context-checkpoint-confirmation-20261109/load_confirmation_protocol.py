"""Derive the reviewed 4096-only run from the immutable base protocol."""
import json
from pathlib import Path


def load_protocol(base, work):
    protocol = json.loads(Path(base).read_text())
    scope_path = Path(work) / 'confirmation-scope.json'
    if not scope_path.exists():
        return protocol
    scope = json.loads(scope_path.read_text())
    if scope != {'contexts': [4096], 'max_starts': 240, 'max_active_seconds': 21600}:
        raise ValueError('Unreviewed confirmation scope or bounds')
    protocol['workload']['contexts'] = scope['contexts']
    protocol['workload']['comparison_blocks'] = {'4096': 1}
    protocol['bounds'].update(
        max_starts_global=scope['max_starts'],
        max_epochs=1,
        max_user_resumes=0,
        max_active_wall_seconds_global=scope['max_active_seconds'],
        max_wall_seconds_per_worker=scope['max_active_seconds'],
        base_expected_starts=206,
    )
    return protocol
