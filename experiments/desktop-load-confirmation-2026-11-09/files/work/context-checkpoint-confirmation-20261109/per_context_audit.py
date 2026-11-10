"""Audit independently verified context receipts without pooling timing rows."""
import argparse
import json
from pathlib import Path

from certificates import verify_context
from evidence import EvidenceError, GUARDS, Store, encode, publish, sha_bytes, sha_file, state_of
from load_confirmation_protocol import load_protocol
from runtime_integration import PREPARED

WORK = Path(__file__).resolve().parent
BASE_PROTOCOL = PREPARED / 'protocol.json'
PINS = {
    512: ('resumable-boot-uuid-20261107',
          'fc3c5baad2ec86b59d61171bd69126dd9913edd7a50becd432b8b0c26cc18d3e',
          '7dfbc76c12dbe808d9d0ae18ac0dad9e80d51234d9de03b26800cb33bfb9dfc9'),
    2048: ('resumable-confirmation-20261105',
           '1e59c0cf426261f94df9d083bfee47b2b496b0ee5270374bab33c5a33b2d3cb7',
           '1ae63bf76f661c21c59467a16e81d9ee0775595e5669791533abe4c51590f677'),
}
HISTORY = WORK.parent / 'resumable-boot-uuid-20261107/validation/e002-terminal-audit-20261109.json'
HISTORY_SHA = '8bdc50d99b3b9641c0005e8d6a1eb5188234c240de2305443e879b90aa610292'
COMMON = {
    'candidate_sha256': '709f9a33eb632748e472f213157405dca706f42fd547d87494f950919065ef18',
    'original_reference_sha256': '87230ee192aea2c097a063a7f062f7ad4b7fe1f98182ef118b80f0f04ccbb1a2',
    'checkpoint_receipt_sha256': '3372a4fbf63555ab17cacfab007a1ba0dae27306654a9169e725dc28247193d5',
}
NATIVE_SHA = '47228969d7db3b1a353e33ab777ab5679332d8b1292512b75372b9783f479990'


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def checked_context(root, context, *, receipt_sha=None, exit_sha=None):
    protocol = load_protocol(BASE_PROTOCOL, root)
    base = json.loads(BASE_PROTOCOL.read_text())
    require(sha_bytes(encode(base)) == 'a361ed7fcf02e1e9a824ff549b22faf778b19c09900325d5bd615497a3a3ed5e',
            'Base statistical protocol changed')
    expected_scope = [4096] if context == 4096 else [512, 2048, 4096]
    require(protocol['workload']['contexts'] == expected_scope, 'Unexpected receipt protocol scope')
    if context != 4096:
        require(protocol == base, 'Historical protocol changed')
    evidence = root / 'evidence'
    require((evidence / 'journal.jsonl').is_file(), 'Evidence journal missing')
    store = Store(evidence, protocol)
    events, _ = store.read()
    state = state_of(events, protocol)
    require(state['current_epoch'] is None, 'Worker epoch still open')
    seal = state['seals'][str(context)]
    receipt_path = evidence / seal['receipt']['path']
    post_path = root / 'post-exit/e001.json'
    if receipt_sha is not None:
        require(sha_file(receipt_path) == receipt_sha, 'Pinned context receipt changed')
    if exit_sha is not None:
        require(sha_file(post_path) == exit_sha, 'Pinned post-exit report changed')
    post = json.loads(post_path.read_text())
    require(post['epoch'] == 'e001' and post['worker_process_verified_gone'] is True
            and post['context_seals'][str(context)] == seal,
            'Context seal missing from verified post-exit report')
    result = verify_context(store, context)
    for epoch in result['contributing_epochs']:
        closure = state['epochs'][epoch]
        require(closure['closed'] and closure['exit_verified'], 'Contributing worker exit unverified')
        guards = json.loads((evidence / closure['receipt']['path']).read_text())
        require(all(guards.get(key) is True for key in GUARDS), 'Final contributing guard failed')
    identities = result['frozen_identities']
    require(all(identities[key] == value for key, value in COMMON.items()), 'Incompatible model or reference')
    sources = identities['source_sha256']
    require(sources[str(root / 'native_runner.py')] == NATIVE_SHA, 'Incompatible native timing runner')
    require(sources[str(root / 'contract.yaml')] == sha_file(PREPARED / 'base-contract.yaml'),
            'Incompatible benchmark contract')
    if context == 4096:
        require(len(state['epochs']) == 1 and state['epochs']['e001']['reason'] == 'complete'
                and post.get('completed_confirmation_scope') == [4096]
                and state['starts_reserved'] == post['reserved_start_slots'] <= 240
                and state['active_seconds_closed'] == post['active_seconds_upper_total'] <= 21600,
                'Fresh 4096 scope or budget is incomplete')
    else:
        require(context in post['verified_positive_contexts'] and result['repeatable_gain_verified'],
                'Historical checkpoint lacks a verified positive result')
    return {
        'pass': str(root), 'receipt_sha256': sha_file(receipt_path),
        'post_exit_sha256': sha_file(post_path), 'contexts': protocol['workload']['contexts'],
        'all_controls_passed': result['all_controls_passed'], 'aa_noise': result['aa_noise'],
        'blocks': {number: {key: block[key] for key in ('pairs', 'median_ratio', 'bootstrap_95pct_ratio',
                  'median_saved_ms_per_token', 'repeatable_gain_verified')}
                   for number, block in result['blocks'].items()},
    }


def clears_threshold(contexts, ratio):
    if set(contexts) != {512, 2048, 4096}:
        return False
    for context, row in contexts.items():
        if not row['all_controls_passed'] or len(row['blocks']) != (2 if context == 512 else 1):
            return False
        for block in row['blocks'].values():
            if (block['pairs'] != 20 or not block['repeatable_gain_verified']
                    or block['median_ratio'] < ratio
                    or block['bootstrap_95pct_ratio'][0] <= max(ratio, 1 + row['aa_noise'])):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', action='store_true', help='Audit all three completed scopes and publish once')
    args = parser.parse_args()
    require(sha_file(HISTORY) == HISTORY_SHA, 'Failed-attempt audit changed')
    contexts = {context: checked_context(WORK.parent / run, context, receipt_sha=receipt_sha, exit_sha=exit_sha)
                for context, (run, receipt_sha, exit_sha) in PINS.items()}
    if args.final:
        contexts[4096] = checked_context(WORK, 4096)
    report = {
        'schema': 1, 'scope': 'Full-model decode at separate recorded desktop sessions for 512, 2048 and 4096',
        **COMMON, 'contexts': contexts, 'missing_contexts': sorted({512, 2048, 4096} - set(contexts)),
        'one_percent_verified': clears_threshold(contexts, 1.01),
        'five_percent_verified': clears_threshold(contexts, 1.05),
        'failed_attempt_audit': {'path': str(HISTORY), 'sha256': HISTORY_SHA},
    }
    if args.final:
        publish(WORK / 'per-context-verdict.json', report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
