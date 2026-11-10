"""Read-only process check, durable exit bound, then post-exit evidence audit."""
import argparse
import json
from pathlib import Path
from evidence import Controller,Store,EvidenceError,publish
from certificates import verify_context
from runtime_integration import boot_identity,process_identity,PREPARED
from load_confirmation_protocol import load_protocol


def finalize_exit(work,epoch):
    work=Path(work)
    protocol=load_protocol(PREPARED/'protocol.json',work)
    store=Store(work/'evidence',protocol);controller=Controller(store)
    state=controller.state();previous=state['epochs'][epoch]
    if not previous['closed']: raise EvidenceError('Worker has not closed its epoch')
    guard=json.loads((store.root/previous['receipt']['path']).read_text())
    old=guard['worker_process'];current=process_identity(old['pid'])
    if current==old: raise EvidenceError('Same worker is still live; cannot exclude cleanup time')
    if not previous['exit_verified']:
        controller.verify_exit(epoch,clock_domain=boot_identity(),process_verified_gone=True,
            observation={'worker_process':old,'observed_process':current})
    state=controller.state();store.verify_artifacts()
    contexts={context:verify_context(store,int(context)) for context in state['seals']}
    all_exits=all(not item.get('requires_external_exit') or item.get('exit_verified') for item in state['epochs'].values())
    last=list(state['epochs'].values())[-1]
    calendar=last['exit_observed_utc']-state['freeze']['utc']
    scope={str(context) for context in protocol['workload']['contexts']}
    scoped_complete=(last['reason']=='complete' and all_exits
        and state['active_seconds_closed']<=protocol['bounds']['max_active_wall_seconds_global']
        and 0<=calendar<=protocol['bounds']['max_calendar_seconds']
        and set(contexts)==scope)
    complete=scoped_complete and scope=={'512','2048','4096'}
    goal=complete and all(value['five_percent_verified'] for value in contexts.values())
    report={'schema':1,'epoch':epoch,'worker_process_verified_gone':True,
        'active_seconds_upper_total':state['active_seconds_closed'],'calendar_seconds':calendar,
        'reserved_start_slots':state['starts_reserved'],
        'completed_confirmation_scope':sorted(map(int,scope)) if scoped_complete else [],
        'complete_locked_workload':complete,'five_percent_improvement_verified':goal,
        'verified_positive_contexts':[int(key) for key,value in contexts.items() if value['repeatable_gain_verified']],
        'context_seals':state['seals'],'acceptance_enabled':False,
        'claim_scope':protocol['environment']['claim_scope']}
    publish(work/f'post-exit/{epoch}.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--work',required=True,type=Path);parser.add_argument('--epoch',required=True)
    args=parser.parse_args()
    print(json.dumps(finalize_exit(args.work,args.epoch),indent=2))
