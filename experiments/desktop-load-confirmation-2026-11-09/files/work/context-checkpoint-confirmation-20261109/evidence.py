"""Durable, conservative reservations and immutable result publication.

One controller process owns a pass. Native timing is supplied by an adapter;
this module never imports MLX or authorizes a model run.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid


class EvidenceError(RuntimeError): pass
class BoundExceeded(EvidenceError): pass

GUARDS=('protected_unchanged','checkpoint_and_snapshot_unchanged',
    'vocabulary_storage_unchanged','original_methods_restored','reused_correctness_sources_unchanged')


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha_bytes(value): return hashlib.sha256(value).hexdigest()


def raw_json(value):
    """Keep nonfinite failed measurements explicitly, never as a valid number."""
    if isinstance(value,float) and not math.isfinite(value): return {'__nonfinite_float__':repr(value)}
    if isinstance(value,dict): return {key:raw_json(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)): return [raw_json(item) for item in value]
    return value


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def publish(path,value):
    """Publish once without replacing a target, including after a crash."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.parent/('.'+path.name+'.pending-'+uuid.uuid4().hex)
    data=encode(value)+b'\n'
    with temporary.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    try:
        os.link(temporary,path)  # Atomic no-replace publication on the same FS.
        sync_directory(path.parent)
    finally:
        # An existing target is never overwritten or treated as the new row.
        temporary.unlink()
        sync_directory(path.parent)
    return {'path':str(path),'sha256':sha_bytes(data)}


def state_of(events,protocol):
    if not events or events[0].get('kind')!='freeze': raise EvidenceError('Missing frozen identities')
    if events[0]['protocol_sha256']!=sha_bytes(encode(protocol)):
        raise EvidenceError('Frozen protocol changed')
    state={'epochs':{},'current_epoch':None,'collections':{},'controls':{},'seals':{},
        'starts_reserved':0,'active_seconds_closed':0.0,'terminal':False,'freeze':events[0]}
    for event in events[1:]:
        kind=event['kind'];current=state['current_epoch']
        if kind=='open_epoch':
            if state['terminal'] or current is not None or event['epoch'] in state['epochs']:
                raise EvidenceError('Epoch cannot reopen or bypass a terminal pass')
            if len(state['epochs'])>=protocol['bounds']['max_epochs']: raise BoundExceeded('Epoch limit')
            if state['epochs']:
                last=list(state['epochs'].values())[-1]
                if last.get('reason')!='user_pause' or event.get('previous_process_verified_gone') is not True:
                    raise EvidenceError('Only verified closed user-pause epochs can resume')
                if last.get('requires_external_exit') and not last.get('exit_verified'):
                    raise EvidenceError('Worker exit and its final active-wall bound are unverified')
                if last.get('clock_domain') is not None and event.get('clock_domain')!=last['clock_domain']:
                    raise EvidenceError('Monotonic clock domain changed across resume')
            state['epochs'][event['epoch']]={**event,'closed':False}
            state['current_epoch']=event['epoch']
        elif kind=='reserve':
            if state['terminal'] or current is None or event['epoch']!=current:
                raise EvidenceError('Reservation outside active epoch')
            if event['collection_id'] in state['collections']: raise EvidenceError('Duplicate collection')
            if any(row['epoch']==current and not row.get('committed') for row in state['collections'].values()):
                raise EvidenceError('Prior collection is still executing or orphaned')
            calls=event['calls']
            if len(calls) not in (1,2) or len({call['id'] for call in calls})!=len(calls):
                raise EvidenceError('Invalid atomic call reservation')
            old_ids={call['id'] for row in state['collections'].values() for call in row['calls']}
            if any(call['id'] in old_ids for call in calls): raise EvidenceError('Duplicate call slot')
            if len(calls)==2 and event['role']!='qualification' and [call['role'] for call in calls]!=['primer','measured']:
                raise EvidenceError('Reserve primer and measurement together before either runs')
            state['starts_reserved']+=len(calls)
            if state['starts_reserved']>protocol['bounds']['max_starts_global']: raise BoundExceeded('Global start limit')
            state['collections'][event['collection_id']]={**event,'committed':False}
        elif kind=='commit':
            collection=state['collections'].get(event['collection_id'])
            if (collection is None or current is None or collection['epoch']!=current
                    or collection['committed'] or len(event['artifacts'])!=len(collection['calls'])):
                raise EvidenceError('Result cannot be replaced or moved between epochs')
            if [row['id'] for row in event['artifacts']]!=[call['id'] for call in collection['calls']]:
                raise EvidenceError('Result does not account for every reserved slot')
            for artifact in event['artifacts']:
                if type(artifact['executed']) is not bool or type(artifact['eligible']) is not bool:
                    raise EvidenceError('Execution and eligibility must be explicit')
                if not artifact['executed'] and artifact['eligible']:
                    raise EvidenceError('An unexecuted call cannot be eligible')
            collection.update(committed=True,artifacts=event['artifacts'])
        elif kind=='control':
            key=f"{event['epoch']}:{event['context']}"
            if current is None or event['epoch']!=current or key in state['controls']:
                raise EvidenceError('Control group cannot be replaced')
            if any(row['context']==event['context'] and not row['passed'] for row in state['controls'].values()):
                raise EvidenceError('Completed failed controls cannot be retried')
            state['controls'][key]=event
        elif kind=='seal_context':
            context=str(event['context'])
            if state['terminal'] or current is None or context in state['seals']:
                raise EvidenceError('Context cannot be resealed')
            state['seals'][context]=event
        elif kind=='close_epoch':
            if current is None or event['epoch']!=current or event['reason'] not in ('user_pause','complete','hard_stop'):
                raise EvidenceError('Invalid epoch closure')
            if not isinstance(event['active_seconds'],(int,float)) or event['active_seconds']<0:
                raise EvidenceError('Invalid active-wall receipt')
            state['epochs'][current].update(event,closed=True,exit_verified=False)
            state['active_seconds_closed']+=event['active_seconds']
            state['current_epoch']=None
            if event['reason'] in ('hard_stop','complete'): state['terminal']=True
        elif kind=='verify_exit':
            epoch=state['epochs'].get(event['epoch'])
            if (current is not None or epoch is None or not epoch['closed'] or epoch['exit_verified']
                    or event['process_verified_gone'] is not True or event['clock_domain']!=epoch.get('clock_domain')
                    or not math.isfinite(event['active_seconds_upper'])
                    or event['active_seconds_upper']<epoch['active_seconds']):
                raise EvidenceError('Invalid or duplicate process-exit bound')
            state['active_seconds_closed']+=event['active_seconds_upper']-epoch['active_seconds']
            epoch.update(exit_verified=True,exit_receipt=event['receipt'],
                active_seconds_upper=event['active_seconds_upper'],exit_observed_utc=event['utc'])
        else: raise EvidenceError('Unknown journal event')
    return state


class Store:
    def __init__(self,root,protocol):
        self.root=Path(root).resolve();self.protocol=protocol
        self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'journal.jsonl'
        self.audit_writes=[]

    def _read(self,stream):
        stream.seek(0);data=stream.read()
        if data and not data.endswith(b'\n'): raise EvidenceError('Torn journal; no automatic truncation')
        events=[];head='0'*64
        for sequence,line in enumerate(data.splitlines()):
            record=json.loads(line);claimed=record.pop('sha256',None)
            if record.get('sequence')!=sequence or record.get('previous_sha256')!=head or sha_bytes(encode(record))!=claimed:
                raise EvidenceError('Altered journal chain')
            head=claimed;events.append(record['payload'])
        if events: state_of(events,self.protocol)
        return events,head

    def read(self):
        if not self.path.exists(): return [],'0'*64
        with self.path.open('rb') as stream:
            fcntl.flock(stream,fcntl.LOCK_SH)
            return self._read(stream)

    def append(self,payload,expected_sequence):
        with self.path.open('a+b') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            events,head=self._read(stream)
            if len(events)!=expected_sequence: raise EvidenceError('Stale controller')
            state_of(events+[payload],self.protocol)
            record={'sequence':len(events),'previous_sha256':head,'payload':payload}
            record['sha256']=sha_bytes(encode(record))
            stream.seek(0,os.SEEK_END);stream.write(encode(record)+b'\n');stream.flush();os.fsync(stream.fileno())
            sync_directory(self.root)
        self.audit_writes.append(payload['kind'])
        return record['sha256']

    def verify_artifacts(self,events=None):
        events=self.read()[0] if events is None else events
        for event in events:
            receipts=[]
            if event['kind']=='commit': receipts=event['artifacts']
            if event['kind'] in ('control','seal_context','close_epoch','verify_exit'): receipts=[event['receipt']]
            for receipt in receipts:
                path=self.root/receipt['path']
                if path.is_symlink() or not path.resolve().is_relative_to(self.root) or sha_file(path)!=receipt['sha256']:
                    raise EvidenceError('Artifact identity changed: '+str(path))
        return True


class Controller:
    def __init__(self,store,*,monotonic=time.monotonic,utc=time.time):
        self.store=store;self.protocol=store.protocol;self.monotonic=monotonic;self.utc=utc
        self.epoch=None;self.started=None;self.utc_started=None
        self.pause_requested=False;self.violation=None;self.stop_watchdog=threading.Event();self.watchdog=None
        self.lease=None

    def state(self): return state_of(self.store.read()[0],self.protocol)

    def append(self,event):
        events,_=self.store.read()
        return self.store.append(event,len(events))

    def freeze(self,identities):
        if self.store.read()[0]: raise EvidenceError('Pass already frozen')
        self.append({'kind':'freeze','protocol_sha256':sha_bytes(encode(self.protocol)),
            'utc':self.utc(),'identities':identities})

    def open_epoch(self,epoch,*,identities,previous_process_verified_gone=False,started_monotonic=None,started_utc=None,clock_domain=None):
        state=self.state()
        if state['current_epoch'] is not None: raise EvidenceError('Previous epoch remains live or unresolved')
        if identities!=state['freeze']['identities']: raise EvidenceError('Resume identities changed')
        self.store.verify_artifacts()
        if state['epochs']:
            last=list(state['epochs'].values())[-1]
            receipt=json.loads((self.store.root/last['receipt']['path']).read_text())
            if not all(receipt.get(key) is True for key in GUARDS):
                raise EvidenceError('Prior epoch final provenance failed')
        self.lease=(self.store.root/'worker.lock').open('a+b')
        try:
            fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.started=self.monotonic() if started_monotonic is None else started_monotonic
            self.utc_started=self.utc() if started_utc is None else started_utc
            if self.started>self.monotonic() or self.utc_started>self.utc(): raise EvidenceError('Future worker start receipt')
            self.epoch=epoch
            self.check()
            self.append({'kind':'open_epoch','epoch':epoch,'utc':self.utc_started,
                'started_monotonic':self.started,'clock_domain':clock_domain,
                'previous_process_verified_gone':previous_process_verified_gone})
            self.check()
        except BaseException:
            self.lease.close();self.lease=None
            raise

    def check(self):
        if self.violation: raise BoundExceeded(self.violation)
        state=self.state()
        elapsed=self.monotonic()-self.started if self.started is not None else 0
        if elapsed<0 or (self.utc_started is not None and self.utc()<self.utc_started):
            raise BoundExceeded('Clock regression')
        bounds=self.protocol['bounds']
        if self.utc()-state['freeze']['utc']>=bounds['max_calendar_seconds']: raise BoundExceeded('Calendar limit')
        if elapsed>=bounds['max_wall_seconds_per_worker']: raise BoundExceeded('Worker wall limit')
        if state['active_seconds_closed']+elapsed>=bounds['max_active_wall_seconds_global']:
            raise BoundExceeded('Global active-wall limit')
        return elapsed

    def start_watchdog(self):
        # No evidence serialization or fsync is performed by the watchdog.
        # It records an in-memory violation; the active request drains and its
        # raw result is committed before the controlling thread raises.
        closed=self.state()['active_seconds_closed'];first=self.state()['freeze']['utc'];bounds=self.protocol['bounds']
        def watch():
            while not self.stop_watchdog.wait(bounds['watchdog_seconds']):
                elapsed=self.monotonic()-self.started
                if elapsed<0 or self.utc()<self.utc_started: self.violation='Clock regression'
                elif self.utc()-first>=bounds['max_calendar_seconds']: self.violation='Calendar limit'
                elif elapsed>=bounds['max_wall_seconds_per_worker']: self.violation='Worker wall limit'
                elif closed+elapsed>=bounds['max_active_wall_seconds_global']: self.violation='Global active-wall limit'
        self.watchdog=threading.Thread(target=watch,daemon=True);self.watchdog.start()

    def request_pause(self,*_): self.pause_requested=True

    def reserve(self,collection_id,role,coordinates,call_roles):
        self.check()
        if self.pause_requested: raise EvidenceError('Pause requested before collection')
        calls=[{'id':collection_id+f'-{i}','role':call_role} for i,call_role in enumerate(call_roles)]
        self.append({'kind':'reserve','collection_id':collection_id,'epoch':self.epoch,
            'role':role,'coordinates':coordinates,'calls':calls})
        return calls

    def commit(self,collection_id,results):
        collection=self.state()['collections'][collection_id]
        if len(results)!=len(collection['calls']): raise EvidenceError('Results must account for reserved slots')
        artifacts=[]
        for call,result in zip(collection['calls'],results):
            relative='rows/'+call['id']+'.json'
            receipt=publish(self.store.root/relative,raw_json(result))
            artifacts.append({'id':call['id'],'path':relative,'sha256':receipt['sha256'],
                'executed':result['executed'],'eligible':result['eligible']})
        self.append({'kind':'commit','collection_id':collection_id,'artifacts':artifacts})
        # Persist the completed evidence even if its wall bound was exceeded.
        self.check()

    def collect(self,collection_id,coordinates,run_collection):
        self.reserve(collection_id,'collection',coordinates,['primer','measured'])
        # run_collection must perform both calls serially with no artifact/
        # journal operation between them. It supplies drained raw observations.
        results=run_collection()
        self.commit(collection_id,results)
        return results

    def close(self,reason,guard_receipt):
        self.stop_watchdog.set()
        if self.watchdog is not None: self.watchdog.join()
        try: self.check()
        except BoundExceeded as error:
            reason='hard_stop';guard_receipt={**guard_receipt,'bound_stop':str(error)}
        unresolved=any(row['epoch']==self.epoch and not row['committed'] for row in self.state()['collections'].values())
        if unresolved or not all(guard_receipt.get(key) is True for key in GUARDS):
            reason='hard_stop'
        receipt=publish(self.store.root/f'epochs/{self.epoch}-close.json',guard_receipt)
        self.append({'kind':'close_epoch','epoch':self.epoch,'reason':reason,
            'utc':self.utc(),'active_seconds':max(0,self.monotonic()-self.started),
            'requires_external_exit':bool(guard_receipt.get('worker_process')),
            'receipt':{'path':str(Path(receipt['path']).relative_to(self.store.root)),'sha256':receipt['sha256']}})
        self.epoch=None
        if self.lease is not None: self.lease.close();self.lease=None

    def verify_exit(self,epoch,*,clock_domain,process_verified_gone,observation):
        previous=self.state()['epochs'][epoch]
        if clock_domain!=previous.get('clock_domain') or not process_verified_gone:
            raise EvidenceError('Process exit or boot identity is not verified')
        upper=self.monotonic()-previous['started_monotonic']
        if upper<previous['active_seconds']: raise EvidenceError('Process-exit clock regressed')
        receipt=publish(self.store.root/f'epochs/{epoch}-exit.json',{
            **observation,'active_seconds_upper':upper,'clock_domain':clock_domain,'utc':self.utc()})
        self.append({'kind':'verify_exit','epoch':epoch,'utc':self.utc(),'clock_domain':clock_domain,
            'process_verified_gone':True,'active_seconds_upper':upper,
            'receipt':{'path':str(Path(receipt['path']).relative_to(self.store.root)),'sha256':receipt['sha256']}})


def pair_rows(store,context,block):
    """Return every complete same-epoch pair and every structurally unmatched row."""
    store.verify_artifacts();state=state_of(store.read()[0],store.protocol)
    groups={};unmatched=[];pair_epochs={};last_pair=0
    for collection in state['collections'].values():
        c=collection['coordinates']
        if c.get('phase')!='comparison' or c.get('context')!=context or c.get('block')!=block: continue
        key=(collection['epoch'],c['pair'])
        if c['pair'] not in pair_epochs:
            if c['pair']!=last_pair+1: raise EvidenceError('Global pair order was reset or skipped')
            last_pair=c['pair'];pair_epochs[c['pair']]=collection['epoch']
        elif pair_epochs[c['pair']]!=collection['epoch']:
            raise EvidenceError('An abandoned pair ID cannot be reused across epochs')
        group=groups.setdefault(key,{})
        lane=c['lane']
        if lane not in ('baseline','candidate') or lane in group: raise EvidenceError('Duplicate or unknown paired lane')
        order=('baseline','candidate') if c['pair']%2 else ('candidate','baseline')
        if lane!=order[len(group)]: raise EvidenceError('Predeclared alternating order changed')
        if collection['committed']:
            artifact=collection['artifacts'][-1]
            group[lane]={'collection':collection,'row':json.loads((store.root/artifact['path']).read_text())}
        else:
            group[lane]={'collection':collection,'row':None}
    pairs=[]
    for key,group in groups.items():
        if set(group)=={'baseline','candidate'} and all(value['row'] is not None and value['row']['executed'] for value in group.values()):
            pairs.append({'epoch':key[0],'pair':key[1],'lanes':group})
        else: unmatched.extend(value for value in group.values())
    return pairs,unmatched
