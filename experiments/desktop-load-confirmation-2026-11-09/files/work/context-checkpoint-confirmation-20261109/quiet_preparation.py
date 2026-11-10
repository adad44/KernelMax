"""Bounded nominal-start preparation; no inference or timed sample retry."""
import time


class QuietPreparationFailure(RuntimeError):
    pass


def wait_nominal(events, *, context, stable_seconds, max_seconds, observe,
                 thermal, check, persist, poll_seconds=1, sleep=time.sleep,
                 clock=time.monotonic):
    if (not 0 < stable_seconds < max_seconds <= 1200 or not 0 < poll_seconds <= 2):
        raise ValueError('Quiet preparation bounds invalid')
    entry={'context':dict(context),'stable_seconds':stable_seconds,'max_seconds':max_seconds,
           'outside_request_timing':True,'inference_requests':0,'timed_retries':0,
           'status':'waiting','samples':[],'environment_boundaries':[]}
    events.append(entry)
    started=clock()
    nominal_since=None
    next_check=started
    try:
        entry['environment_before']=observe()
        persist()
        if "'AC Power'" not in entry['environment_before']['power']:
            raise QuietPreparationFailure('AC power unavailable before quiet preparation')
        while True:
            now=clock()
            if now-started>=max_seconds:
                entry['status']='timed_out'
                raise QuietPreparationFailure('Nominal thermal window not reached within preparation bound')
            if now>=next_check:
                check()
                boundary=observe()
                entry['environment_boundaries'].append(boundary)
                if "'AC Power'" not in boundary['power']:
                    raise QuietPreparationFailure('AC power lost during quiet preparation')
                if boundary['thermal_state']!=0:
                    nominal_since=None
                next_check=now+10
            level=thermal()
            entry['samples'].append({'elapsed_seconds':clock()-started,'thermal_state':level})
            if level!=0:
                nominal_since=None
            elif nominal_since is None:
                nominal_since=clock()
            if nominal_since is not None and clock()-nominal_since>=stable_seconds:
                final=observe()
                entry['environment_after']=final
                if "'AC Power'" not in final['power']:
                    raise QuietPreparationFailure('AC power lost at quiet-preparation completion')
                if final['thermal_state']==0 and clock()-started<max_seconds:
                    entry['status']='completed'
                    entry['continuous_nominal_seconds']=clock()-nominal_since
                    return entry
                nominal_since=None
            remaining=max_seconds-(clock()-started)
            if remaining>0:
                sleep(min(poll_seconds,remaining))
    except BaseException as error:
        if entry['status']=='waiting': entry['status']='failed'
        entry['error']=str(error)
        raise
    finally:
        entry['elapsed_seconds']=clock()-started
        persist()
