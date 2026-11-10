"""Bridge for the frozen native runner. Importing this module does not load MLX."""
import json
import atexit
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
from uuid import UUID
from evidence import Controller,Store,EvidenceError,GUARDS,encode,publish,sha_bytes,sha_file,sync_directory
from verification import ContextVerifier
from certificates import verify_context
from load_confirmation_protocol import load_protocol
from evaluator.native_baseline import environment
from evaluator.measurement import assess_environment

PREPARED=Path('/Users/alandiaz/Downloads/KernelMaxxxing/work/resumable-confirmation-contract-20261104')


class UserPause(Exception): pass


def process_identity(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','lstart='],capture_output=True,text=True,timeout=5)
    return {'pid':pid,'birth':result.stdout.strip()} if result.returncode==0 and result.stdout.strip() else None


def boot_identity():
    """Identify the boot session independently of wall-clock adjustments."""
    try:
        result = subprocess.run(
            ['/usr/sbin/sysctl', '-n', 'kern.bootsessionuuid'],
            capture_output=True, text=True, timeout=5, check=True,
        )
        session = UUID(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise EvidenceError('Cannot bind monotonic clock domain to a boot session') from error
    if session.int == 0:
        raise EvidenceError('Boot session UUID is empty')
    return f'boot-session:{session}'


def validate_activation(work,protocol):
    """Activation is external to the prepared contract and absent by default."""
    work=Path(work)
    activation=json.loads((work/'timing-authorization.json').read_text())
    required=('user_timing_authorization','operator','measurement_basis','statistical_failure_handling')
    if (activation.get('authorized') is not True or activation.get('protocol_sha256')!=sha_bytes(encode(protocol))
            or activation.get('prepared_bundle_sha256')!=sha_file(PREPARED/'contract-bundle.json')
            or any(not isinstance(activation.get(key),str) or not activation[key].strip() for key in required)
            or activation.get('unchanged_statistical_reroll') is not False):
        raise EvidenceError('Specific timing activation is absent, stale or outside the prepared contract')
    identities=activation['frozen_identities']
    build=json.loads((work/'runtime-build.json').read_text())
    if (activation.get('runtime_build_sha256')!=sha_file(work/'runtime-build.json')
            or identities['source_sha256']!=build['source_sha256']):
        raise EvidenceError('Executable integration was not frozen into this activation')
    for path,expected in identities['source_sha256'].items():
        if sha_file(path)!=expected: raise EvidenceError('Executable/input identity changed before activation: '+path)
    if (identities['original_reference_sha256']!=protocol['original_reference_sha256']
            or identities['candidate_sha256']!=protocol['candidate_sha256']):
        raise EvidenceError('Activation changed the original reference or candidate')
    return activation


class RuntimeSession:
    def __init__(self,work,epoch):
        boot_monotonic=time.monotonic();boot_utc=time.time()
        self.work=Path(work);self.protocol=load_protocol(PREPARED/'protocol.json',self.work)
        bundle=json.loads((PREPARED/'contract-bundle.json').read_text())
        for path,expected in bundle['source_sha256'].items():
            if sha_file(path)!=expected: raise EvidenceError('Reviewed contract bundle changed')
        self.activation=validate_activation(self.work,self.protocol)
        observed=environment()
        ready=assess_environment(observed,observed,mode='exploration',confirmation_policy='retain_and_flag_desktop_load')
        if not ready['continue_pass']: raise EvidenceError('Native preflight failed: '+str(ready['stop_reasons']))
        self.gpu_lease=Path('/Users/alandiaz/Downloads/KernelMaxxxing/work/native-gpu-worker.lock').open('a+b')
        fcntl.flock(self.gpu_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atexit.register(self.gpu_lease.close)
        if not epoch or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in epoch):
            raise EvidenceError('Invalid epoch identifier')
        self.epoch=epoch;self.epoch_work=self.work/'epochs'/epoch
        self.epoch_work.mkdir(parents=True,exist_ok=False)
        self.store=Store(self.work/'evidence',self.protocol)
        self.controller=Controller(self.store)
        self.clock_domain=boot_identity()
        identities=self.activation['frozen_identities']
        if not self.store.read()[0]: self.controller.freeze(identities)
        previous=self.controller.state()['epochs']
        verified_previous_process_exit=False
        if previous:
            last=list(previous.values())[-1]
            if not last.get('closed'): raise EvidenceError('Previous worker did not close its epoch')
            receipt=json.loads((self.store.root/last['receipt']['path']).read_text())
            old=receipt['worker_process'];current=process_identity(old['pid'])
            verified_previous_process_exit=current is None or current!=old
            if verified_previous_process_exit and not last.get('exit_verified'):
                # This is an upper bound through observation, not an assumed
                # exit timestamp. A late observation may exhaust the budget.
                self.controller.verify_exit(last['epoch'],clock_domain=self.clock_domain,
                    process_verified_gone=True,observation={'worker_process':old,'observed_process':current})
        self.process=process_identity(os.getpid())
        if self.process is None: raise EvidenceError('Could not identify controller process birth')
        self.controller.open_epoch(epoch,identities=identities,
            previous_process_verified_gone=verified_previous_process_exit,
            started_monotonic=boot_monotonic,started_utc=boot_utc,clock_domain=self.clock_domain)
        self.verifier=ContextVerifier(self.store)
        self.reason=None;self.finished=False
        self.pending_cs=None;self.pending_state_path=None
        self.old_sigint=signal.signal(signal.SIGINT,self.controller.request_pause)
        self.controller.start_watchdog()

    @property
    def pause_requested(self): return self.controller.pause_requested

    def check(self,*,honor_pause=True):
        self.controller.check()
        if honor_pause and self.pause_requested: raise UserPause('Explicit user pause at drained CPU boundary')

    def _collect(self,collect,cs,context,lane,phase,pair,block=None):
        self.check()
        label=f'{self.epoch}-{phase}-{context}-{block}-{pair}-{lane}'
        coordinates={'context':context,'lane':lane,'phase':phase,'pair':pair,'block':block}
        self.controller.reserve(label,'collection',coordinates,['primer','measured'])
        primers=len(cs['resident_primers'])
        started_before=self.request_counter()
        target=cs['phase_memory_requests'] if phase=='diagnostic' else cs['requests']
        measured=len(target)
        native_phase={'diagnostic':'phase_memory','comparison':f'confirm-{block}',
            'warmup':f'warmup-{block}','aa':'aa','aa_warmup':'aa_warmup'}[phase]
        error=None
        try: collect(context,lane,native_phase,pair,phase=='diagnostic')
        except BaseException as caught: error=caught
        new_primers=cs['resident_primers'][primers:];new_measured=target[measured:]
        if len(new_primers)>1 or len(new_measured)>1: raise EvidenceError('Native collector generated extra requests')
        started=self.request_counter()-started_before
        if started not in (0,1,2): raise EvidenceError('Native collector started extra model calls')
        values=[]
        for index,(rows,is_primer) in enumerate(((new_primers,True),(new_measured,False))):
            raw=rows[0] if rows else {'missing_reason':repr(error)}
            flags=raw.get('environment_assessment',{})
            values.append({'executed':index<started,'result_complete':bool(rows),'eligible':bool(rows) and flags.get(
                'continue_pass' if is_primer else 'confirmation_sample_eligible') is True,'raw':raw})
        self.controller.commit(label,values)
        if error is not None: raise error
        self.check()

    def _qualify(self,correctness,cs,request_counter):
        self.check();name=self.epoch+'-qualification'
        self.controller.reserve(name,'qualification',{},['reference_trace','candidate_trace'])
        before=request_counter();error=None;handoff=None
        try: handoff=correctness(512,8)
        except BaseException as caught: error=caught
        started=request_counter()-before
        if started not in (0,1,2): raise EvidenceError('Extra correctness request started')
        proof={'full':[row for row in cs['correctness'] if row['outputs']==128],
            'reuse':cs['correctness_reuse'],'handoff':handoff}
        if handoff is not None:
            arrays=Path(handoff['arrays'])
            with arrays.open('r+b') as stream: os.fsync(stream.fileno())
            sync_directory(arrays.parent)
        rows=[{'executed':index<started,'eligible':error is None and started==2,
            'raw':proof if error is None else {'failure':repr(error)}} for index in range(2)]
        self.controller.commit(name,rows)
        if error is not None: raise error
        self.verifier._references(self.controller.state(),self.epoch)
        self.check()

    def _warm(self,collect,cs,context,block=None):
        phase='aa_warmup' if block is None else 'warmup'
        lanes=('A','B') if block is None else ('baseline','candidate')
        for pair in range(1,6):
            for lane in lanes: self._collect(collect,cs,context,lane,phase,pair,block)

    def execute(self,collect,correctness,cs,guards,request_counter):
        self.request_counter=request_counter
        self._qualify(correctness,cs,request_counter)
        diagnostics={row['coordinates']['lane'] for row in self.controller.state()['collections'].values()
            if row['coordinates'].get('phase')=='diagnostic' and row['committed']
            and all(item['executed'] and item['eligible'] for item in row['artifacts'])}
        for lane in ('baseline','candidate'):
            if lane not in diagnostics: self._collect(collect,cs,512,lane,'diagnostic',0)
        for context in self.protocol['workload']['contexts']:
            state=self.controller.state()
            if str(context) in state['seals']: continue
            previous=[row for row in state['controls'].values() if row['context']==context]
            if any(not row['passed'] for row in previous):
                # The prepared policy does not allow reopening a completed
                # failed context merely to obtain a different A/A result.
                raise EvidenceError('Paused statistically failed context cannot obtain a replacement control')
            self._warm(collect,cs,context)
            for pair in range(1,21):
                for lane in ('A','B') if pair%2 else ('B','A'):
                    self._collect(collect,cs,context,lane,'aa',pair)
            control=self.verifier.record_control(self.controller,context)
            cs['calibrations'][str(context)]=control
            for block in range(1,self.protocol['workload']['comparison_blocks'][str(context)]+1):
                from evidence import pair_rows
                pairs,_=pair_rows(self.store,context,block)
                if len(pairs)>20: raise EvidenceError('Extra paired observations')
                if len(pairs)==20: continue
                self._warm(collect,cs,context,block)
                while len(pairs)<20:
                    state=self.controller.state()
                    prior=[row['coordinates']['pair'] for row in state['collections'].values()
                        if row['coordinates'].get('phase')=='comparison'
                        and row['coordinates'].get('context')==context and row['coordinates'].get('block')==block]
                    pair=max(prior,default=0)+1
                    for lane in ('baseline','candidate') if pair%2 else ('candidate','baseline'):
                        self._collect(collect,cs,context,lane,'comparison',pair,block)
                    pairs,_=pair_rows(self.store,context,block)
            self.check()
            value=self.verifier.seal(self.controller,context,guards())
            for block,summary in value['blocks'].items(): cs['summaries'][f'{context}-block-{block}']=summary
            self._retain(value)
        self.reason='complete'
        cs['status']='completed_resumable_confirmation'

    def _retain(self,value):
        if not value['repeatable_gain_verified']: return
        checked=verify_context(self.store,value['context'])
        if checked!=value: raise EvidenceError('Live context conclusion does not match its immutable seal')
        catalog=self.work/'resumable-checkpoint-catalog.json'
        revisions=sorted((self.work/'catalog-revisions').glob('*.json')) if (self.work/'catalog-revisions').exists() else []
        previous_hash=None;data=None;previous_entries=[]
        for number,path in enumerate(revisions,1):
            next_data=json.loads(path.read_text())
            if (path.name!=f'{number:03d}.json' or len(next_data['checkpoints'])!=number
                    or next_data.get('previous_revision_sha256')!=previous_hash
                    or next_data['checkpoints'][:-1]!=previous_entries):
                raise EvidenceError('Append-only checkpoint catalog changed')
            data=next_data;previous_hash=sha_file(path);previous_entries=data['checkpoints']
        if data is None: data={'schema':1,'original_reference_sha256':self.protocol['original_reference_sha256'],
            'protocol_sha256':sha_bytes(encode(self.protocol)),'checkpoints':[]}
        if data['original_reference_sha256']!=self.protocol['original_reference_sha256']:
            raise EvidenceError('Checkpoint reference changed')
        context=value['context'];receipt=self.controller.state()['seals'][str(context)]['receipt']
        if any(row['context']==context for row in data['checkpoints']): raise EvidenceError('Context checkpoint already retained')
        data['checkpoints'].append({'context':context,'candidate_sha256':self.protocol['candidate_sha256'],
            'evidence':receipt,'five_percent_verified':value['five_percent_verified'],
            'acceptance_enabled':False,'claim_scope':value['claim_scope']})
        data['previous_revision_sha256']=previous_hash
        # Each catalog revision is immutable; a mutable pointer is unnecessary.
        publish(self.work/f'catalog-revisions/{len(data["checkpoints"]):03d}.json',data)
        if not catalog.exists(): publish(catalog,{'schema':1,'original_reference_sha256':self.protocol['original_reference_sha256'],
            'protocol_sha256':sha_bytes(encode(self.protocol)),'revisions_directory':'catalog-revisions',
            'selection':'highest contiguous validated hash-linked revision; no mutable pointer'})

    def finalize(self,cs):
        if self.finished: return
        reason='user_pause' if cs.get('status')=='paused_by_user' else self.reason or 'hard_stop'
        guards={key:cs.get(key) is True for key in GUARDS}
        guards['worker_process']=self.process
        self.controller.close(reason,guards)
        self.finished=True
        signal.signal(signal.SIGINT,self.old_sigint)
        state=self.controller.state()
        reason=state['epochs'][self.epoch]['reason']
        sealed={int(context):json.loads((self.store.root/value['receipt']['path']).read_text())
            for context,value in state['seals'].items()}
        full=(reason=='complete' and all(guards.values()) and set(sealed)=={512,2048,4096}
            and all(value['five_percent_verified'] for value in sealed.values())
            and state['active_seconds_closed']<=self.protocol['bounds']['max_active_wall_seconds_global'])
        publish(self.epoch_work/'epoch-conclusion.json',{'reason':reason,'global_starts_reserved':state['starts_reserved'],
            'active_seconds_closed':state['active_seconds_closed'],'sealed_contexts':sorted(sealed),
            'evidence_goal_requirements_met_pending_exit':full,
            'five_percent_improvement_verified':False,'requires_exit_verification':True,
            'guards':guards,'acceptance_enabled':False})
        cs['five_percent_improvement_verified']=False
        cs['repeatable_improvement_verified']=any(value['repeatable_gain_verified'] for value in sealed.values())

    def stage_finalize(self,cs,path):
        self.pending_cs=cs;self.pending_state_path=Path(path)

    def finish_native(self,state):
        if self.pending_cs is None:
            self.abort();return
        cs=self.pending_cs
        cs['protected_unchanged']=cs['protected_unchanged'] and state['protected_unchanged']
        cs['checkpoint_and_snapshot_unchanged']=cs['checkpoint_and_snapshot_unchanged'] and state['checkpoint_unchanged']
        cs['vocabulary_storage_unchanged']=cs['vocabulary_storage_unchanged'] and state['vocabulary_storage_unchanged']
        self.finalize(cs)
        # Legacy mutable state remains an operational report; immutable rows,
        # sealed prefixes and the verified post-exit report are authoritative.
        temporary=self.pending_state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(cs,indent=2,sort_keys=True,allow_nan=False)+'\n')
        temporary.replace(self.pending_state_path)
        state['status']=cs['status']
        state['five_percent_improvement_verified']=False
        if state['commands']:
            state['commands'][-1]['repeatable_improvement_verified']=cs['repeatable_improvement_verified']

    def abort(self):
        if self.finished: return
        self.controller.close('hard_stop',{'worker_process':self.process,
            **{key:False for key in GUARDS},'reason':'Runtime exited before complete final guard receipts'})
        self.finished=True
        signal.signal(signal.SIGINT,self.old_sigint)
