#!/usr/bin/python3
"""Root-owned GPU occupancy policy. Failed or stale telemetry never causes kills."""
import argparse
import csv
from dataclasses import dataclass, field
import fnmatch
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import pwd
import re
import signal
import subprocess
import time

ENV = {'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LC_ALL': 'C'}
LOG = logging.getLogger('gpu-guard')


def emit(event, **data):
    LOG.warning(json.dumps({'time': time.time(), 'event': event, **data}, sort_keys=True))


@dataclass(frozen=True)
class Proc:
    pid: int
    start: int
    uid: int
    pgid: int
    cgroup: str
    comm: str
    unit: str = ''
    user_unit: bool = False

    @property
    def identity(self):
        return (self.pid, self.start)

    @property
    def job(self):
        # Docker's systemd cgroup driver uses docker-<container-id>.scope.
        container_scope = re.search(r'(?:^|/)docker-[0-9a-f]{64}\.scope(?:/|$)', self.cgroup)
        if self.unit or container_scope or ('/docker/' in self.cgroup) or ('kubepods' in self.cgroup):
            return f'{self.uid}:cgroup:{self.cgroup}'
        return f'{self.uid}:pgid:{self.pgid}:{self.cgroup}'


def proc(pid):
    p = Path('/proc') / str(pid)
    try:
        stat = (p/'stat').read_text().rsplit(') ', 1)[1].split()
        if stat[0] == 'Z':
            return None
        status = (p/'status').read_text()
        uid = int(re.search(r'^Uid:\s+(\d+)', status, re.M)[1])
        cg = next(x.split('::', 1)[1] for x in (p/'cgroup').read_text().splitlines() if x.startswith('0::'))
        leaf = cg.rsplit('/', 1)[-1]
        unit = leaf if leaf.endswith('.service') and not leaf.startswith('user@') else ''
        return Proc(pid, int(stat[19]), uid, int(stat[2]), cg, (p/'comm').read_text().strip(), unit, '/user@' in cg)
    except (OSError, ValueError, IndexError, StopIteration, TypeError):
        return None


def safe_signal(p, sig, dry_run=False):
    """A pidfd plus a fresh start-time check prevents PID reuse races."""
    try:
        fd = os.pidfd_open(p.pid)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        current = proc(p.pid)
        if not current or current.identity != p.identity or current.uid != p.uid:
            return False
        if not dry_run:
            signal.pidfd_send_signal(fd, sig)
        emit('signal' if not dry_run else 'would_signal', pid=p.pid, uid=p.uid,
             signal=signal.Signals(sig).name, comm=p.comm, unit=p.unit)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def stop_user_unit(p, dry_run):
    if not p.user_unit or not re.fullmatch(r'[A-Za-z0-9_.@:\\-]+\.service', p.unit):
        return
    # Only the user's leaf job unit, never their user manager or session.
    if dry_run:
        emit('would_stop_unit', uid=p.uid, unit=p.unit)
        return
    try:
        account = pwd.getpwuid(p.uid)
        result = subprocess.run([
            '/usr/bin/systemctl', '--user', f'--machine={account.pw_name}@.host',
            '--no-block', 'stop', p.unit
        ], env=ENV,
           capture_output=True, timeout=10)
        emit('stop_unit', uid=p.uid, unit=p.unit, returncode=result.returncode,
             detail=result.stderr.decode(errors='replace')[:300] if result.returncode else '')
    except (KeyError, OSError, subprocess.TimeoutExpired) as error:
        emit('stop_unit_error', uid=p.uid, unit=p.unit, error=type(error).__name__, detail=str(error)[:300])


def query_csv(query):
    result = subprocess.run(['/usr/bin/nvidia-smi', query, '--format=csv,noheader,nounits'],
                            env=ENV, capture_output=True, text=True, timeout=4)
    if result.returncode:
        raise RuntimeError('nvidia-smi failed')
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def telemetry():
    gpus = {}
    for row in query_csv('--query-gpu=uuid,utilization.gpu,utilization.memory'):
        if len(row) != 3 or not row[0].startswith('GPU-'):
            raise ValueError('Invalid GPU telemetry')
        try:
            values = (float(row[1]), float(row[2]))
            if not all(0 <= x <= 100 for x in values):
                raise ValueError('Invalid utilization')
            gpus[row[0]] = max(values)
        except ValueError:
            gpus[row[0]] = None
    if not gpus:
        raise ValueError('No GPU telemetry')
    apps = []
    for row in query_csv('--query-compute-apps=gpu_uuid,pid,used_gpu_memory'):
        if not row:
            continue
        if len(row) != 3 or row[0] not in gpus:
            raise ValueError('Invalid process telemetry')
        apps.append((row[0], int(row[1]), float(row[2])))
    return gpus, apps


@dataclass
class Job:
    procs: dict = field(default_factory=dict)
    gpus: set = field(default_factory=set)
    memory: float = 0


def jobs_from(gpus, apps):
    jobs = {}
    for gpu, pid, memory in apps:
        p = proc(pid)
        if p is None or pid < 2 or pid == os.getpid():
            continue
        j = jobs.setdefault(p.job, Job())
        j.procs[p.identity] = p
        j.gpus.add(gpu)
        j.memory += memory
    return jobs


class IdlePolicy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.states = {}
        self.last = None

    def failure(self):
        self.states.clear()
        self.last = None

    def step(self, now, gpus, jobs):
        actions = []
        if self.last is not None and now - self.last > self.cfg['max_sample_gap_seconds']:
            self.states.clear()
        self.last = now
        self.states = {k:v for k,v in self.states.items() if k in jobs}
        for key, job in jobs.items():
            values = [gpus.get(g) for g in job.gpus]
            idle = (job.memory > self.cfg['memory_threshold_mib'] and values and
                    all(x is not None and x < self.cfg['utilization_threshold'] for x in values))
            old = self.states.get(key)
            members = set(job.procs)
            # Recycled PGIDs/cgroups and genuinely new jobs never inherit an old timer.
            if old and not members.intersection(old['members']):
                old = None
            if old and old['term'] is not None:
                old['members'] = members
                if not old['killed'] and now-old['term'] >= self.cfg['term_grace_seconds']:
                    actions.append(('kill', key, job))
                    old['killed'] = True
                continue
            if not idle:
                self.states.pop(key, None)
                continue
            if old is None:
                old = {'since':now, 'warned':False, 'term':None, 'killed':False, 'members':members}
                self.states[key] = old
            old['members'] = members
            duration = now - old['since']
            if duration >= self.cfg['idle_seconds']:
                actions.append(('term', key, job))
                old['term'] = now
            elif not old['warned'] and duration >= self.cfg['idle_seconds']-self.cfg['warning_seconds']:
                actions.append(('warn', key, job))
                old['warned'] = True
        return actions


def deny_autoclaim(cfg, seen, dry_run):
    current = set()
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        p = proc(int(item.name))
        if p is None or p.pid < 2 or p.pid == os.getpid():
            continue
        banned = any(fnmatch.fnmatchcase(p.unit, pattern) for pattern in cfg['blocked_units'])
        if not banned:
            try:
                # Compare complete arguments/basenames, never arbitrary command substrings.
                args = [(x.decode(errors='replace')) for x in (item/'cmdline').read_bytes().split(b'\0') if x]
                tokens = set(args) | {os.path.basename(a) for a in args}
                banned = any(all(x in tokens for x in rule) for rule in cfg['blocked_commands'])
            except OSError:
                pass
        if not banned:
            continue
        current.add(p.identity)
        if p.identity not in seen:
            emit('blocked_autoclaim', pid=p.pid, uid=p.uid, comm=p.comm, unit=p.unit)
            stop_user_unit(p, dry_run)
            safe_signal(p, signal.SIGTERM, dry_run)
            seen[p.identity] = time.monotonic()
        elif time.monotonic()-seen[p.identity] >= cfg['term_grace_seconds']:
            safe_signal(p, signal.SIGKILL, dry_run)
            seen[p.identity] = time.monotonic()
    return {k:v for k,v in seen.items() if k in current}


def configure_logging(log_dir):
    LOG.setLevel(logging.INFO)
    LOG.addHandler(logging.StreamHandler())
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(Path(log_dir)/'events.jsonl', maxBytes=10*1024*1024, backupCount=5)
        LOG.addHandler(handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/etc/gpu-guard/config.json')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--log-dir', default='/var/log/gpu-guard')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    assert cfg['idle_seconds'] >= 60 and cfg['term_grace_seconds'] >= 1
    assert 0 < cfg['utilization_threshold'] <= 100 and cfg['memory_threshold_mib'] >= 0
    configure_logging(args.log_dir)
    dry_run = args.dry_run or not cfg['enforce']
    policy = IdlePolicy(cfg)
    seen = {}
    last_error = 0
    last_status = 0
    emit('started', enforce=not dry_run, idle_seconds=cfg['idle_seconds'],
         threshold=cfg['utilization_threshold'], memory_mib=cfg['memory_threshold_mib'])
    while True:
        start = time.monotonic()
        if start-last_status >= 5:
            seen = deny_autoclaim(cfg, seen, dry_run)
        try:
            gpus, apps = telemetry()
            jobs = jobs_from(gpus, apps)
            for action,key,job in policy.step(time.monotonic(), gpus, jobs):
                emit('idle_'+action, job=key, pids=[p.pid for p in job.procs.values()],
                     gpus=sorted(job.gpus), memory_mib=job.memory)
                if action == 'warn':
                    continue
                if action == 'term':
                    units = set()
                    for p in job.procs.values():
                        if p.user_unit and (p.uid,p.unit) not in units:
                            stop_user_unit(p, dry_run)
                            units.add((p.uid,p.unit))
                for p in job.procs.values():
                    safe_signal(p, signal.SIGTERM if action=='term' else signal.SIGKILL, dry_run)
            status = {'time':time.time(), 'enforce':not dry_run, 'telemetry':'ok',
                      'gpu_count':len(gpus), 'gpu_processes':len(apps), 'tracked_jobs':len(jobs),
                      'idle_jobs':{k:round(time.monotonic()-v['since']) for k,v in policy.states.items()}}
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            policy.failure()
            status = {'time':time.time(), 'enforce':not dry_run, 'telemetry':'error', 'error':str(error)}
            if start-last_error >= 60:
                emit('telemetry_error_no_kills', error=type(error).__name__)
                last_error = start
        if start-last_status >= 5 or args.once:
            status_dir = Path('/run/gpu-guard')
            if status_dir.exists():
                temp = status_dir/'status.tmp'
                temp.write_text(json.dumps(status, indent=2)+'\n')
                os.chmod(temp, 0o644)
                temp.replace(status_dir/'status.json')
            last_status = start
        if args.once:
            print(json.dumps(status))
            return
        time.sleep(max(0.05, cfg['sample_seconds']-(time.monotonic()-start)))


if __name__ == '__main__':
    main()
