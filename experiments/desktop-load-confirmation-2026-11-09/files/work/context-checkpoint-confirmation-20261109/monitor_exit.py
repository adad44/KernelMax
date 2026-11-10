"""One bounded CPU monitor for an explicitly launched worker; no model execution."""
import argparse
import json
from pathlib import Path
import time
from runtime_integration import process_identity
from verify_exit import finalize_exit


def monitor(work,epoch,expected,*,timeout_seconds=72120):
    if type(expected['pid']) is not int or expected['pid']<=0 or not expected['birth']:
        raise ValueError('A verified positive PID and exact birth are required')
    if not 0<timeout_seconds<=72120: raise ValueError('Bounded monitor only')
    started=time.monotonic()
    while process_identity(expected['pid'])==expected:
        if time.monotonic()-started>=timeout_seconds:
            raise TimeoutError('Monitor bound; never restart or signal a worker on observation timeout')
        time.sleep(1)
    return finalize_exit(work,epoch)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',required=True,type=Path);parser.add_argument('--epoch',required=True)
    parser.add_argument('--pid',required=True,type=int);parser.add_argument('--birth',required=True)
    args=parser.parse_args()
    print(json.dumps(monitor(args.work,args.epoch,{'pid':args.pid,'birth':args.birth}),indent=2))
