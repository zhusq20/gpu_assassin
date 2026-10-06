import signal
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import gpu_guard as g

CFG = dict(idle_seconds=30, max_sample_gap_seconds=5, utilization_threshold=5,
           memory_threshold_mib=1024, term_grace_seconds=6, warning_seconds=5)

def process(pid=100, start=1, pgid=100):
    return g.Proc(pid, start, 1034, pgid, '/user.slice/session-1.scope', 'test')

def job(p=None, gpus=None, memory=2048):
    p=p or process()
    return g.Job({p.identity:p}, set(gpus or ['GPU-A']), memory)

class PolicyTests(unittest.TestCase):
    def run_until(self, policy, end, workload=None, gpu=None, start=0):
        actions=[]
        for now in range(start,end+1):
            actions += policy.step(now,gpu or {'GPU-A':0},{'job':workload or job()})
        return actions

    def test_exact_threshold_and_grace(self):
        policy=g.IdlePolicy(CFG)
        self.assertEqual([a[0] for a in self.run_until(policy,36)], ['warn','term','kill'])
        self.assertEqual(policy.states['job']['term'],30)

    def test_short_idle_never_killed(self):
        self.assertEqual(self.run_until(g.IdlePolicy(CFG),24),[])

    def test_busy_gpu_protects_multigpu_job(self):
        j=job(gpus=['GPU-A','GPU-B'])
        self.assertEqual(self.run_until(g.IdlePolicy(CFG),100,j,{'GPU-A':0,'GPU-B':5}),[])

    def test_unknown_gpu_protects_multigpu_job(self):
        j=job(gpus=['GPU-A','GPU-B'])
        self.assertEqual(self.run_until(g.IdlePolicy(CFG),100,j,{'GPU-A':0,'GPU-B':None}),[])

    def test_small_allocation_is_preserved(self):
        self.assertEqual(self.run_until(g.IdlePolicy(CFG),100,job(memory=1024)),[])

    def test_activity_restarts_idle_window(self):
        policy=g.IdlePolicy(CFG)
        self.run_until(policy,20)
        policy.step(21,{'GPU-A':80},{'job':job()})
        self.assertNotIn('job',policy.states)
        self.assertNotIn('term',[x[0] for x in self.run_until(policy,45,start=22)])

    def test_telemetry_failure_clears_all_timers(self):
        policy=g.IdlePolicy(CFG)
        self.run_until(policy,29)
        policy.failure()
        self.assertEqual(policy.step(30,{'GPU-A':0},{'job':job()}),[])
        self.assertEqual(policy.states['job']['since'],30)

    def test_sampling_gap_clears_timer(self):
        policy=g.IdlePolicy(CFG)
        self.run_until(policy,29)
        self.assertEqual(policy.step(40,{'GPU-A':0},{'job':job()}),[])
        self.assertEqual(policy.states['job']['since'],40)

    def test_pid_reuse_does_not_inherit_timer(self):
        policy=g.IdlePolicy(CFG)
        self.run_until(policy,29)
        self.assertEqual(policy.step(30,{'GPU-A':0},{'job':job(process(start=999))}),[])
        self.assertEqual(policy.states['job']['since'],30)

    def test_exited_job_is_forgotten(self):
        policy=g.IdlePolicy(CFG)
        self.run_until(policy,29)
        policy.step(30,{'GPU-A':0},{})
        self.assertEqual(policy.states,{})

    def test_pidfd_reuse_guard(self):
        with patch.object(g.os,'pidfd_open',return_value=55), patch.object(g.os,'close'), \
             patch.object(g,'proc',return_value=process(start=2)), \
             patch.object(g.signal,'pidfd_send_signal') as send:
            self.assertFalse(g.safe_signal(process(start=1),signal.SIGTERM))
            send.assert_not_called()

    def test_dry_run_never_signals(self):
        with patch.object(g.os,'pidfd_open',return_value=55), patch.object(g.os,'close'), \
             patch.object(g,'proc',return_value=process()), \
             patch.object(g.signal,'pidfd_send_signal') as send:
            self.assertTrue(g.safe_signal(process(),signal.SIGKILL,dry_run=True))
            send.assert_not_called()

    def test_worker_pids_share_job_group(self):
        p1,p2=process(100,1,90),process(101,2,90)
        self.assertEqual(p1.job,p2.job)
        self.assertNotEqual(p1.job,process(102,3,91).job)

    def test_docker_systemd_workers_share_container_job(self):
        cg='/system.slice/docker-'+('a'*64)+'.scope'
        p1=g.Proc(100,1,0,100,cg,'worker')
        p2=g.Proc(101,2,0,101,cg,'worker')
        self.assertEqual(p1.job,p2.job)
        p3=g.Proc(102,3,0,102,'/system.slice/docker-'+('b'*64)+'.scope','worker')
        self.assertNotEqual(p1.job,p3.job)

    def test_busy_gpu_protects_entire_docker_container(self):
        cg='/system.slice/docker-'+('a'*64)+'.scope'
        workers={100:g.Proc(100,1,0,100,cg,'worker'),101:g.Proc(101,2,0,101,cg,'worker')}
        with patch.object(g,'proc',side_effect=workers.get):
            jobs=g.jobs_from({'GPU-A':0,'GPU-B':90},[('GPU-A',100,2048),('GPU-B',101,2048)])
        self.assertEqual(len(jobs),1)
        policy=g.IdlePolicy(CFG)
        for now in range(100):
            self.assertEqual(policy.step(now,{'GPU-A':0,'GPU-B':90},jobs),[])

    def test_term_grace_is_not_evaded_by_late_activity(self):
        policy=g.IdlePolicy(CFG)
        self.run_until(policy,30)
        actions=[]
        for now in range(31,37):
            actions+=policy.step(now,{'GPU-A':80},{'job':job()})
        self.assertEqual([a[0] for a in actions],['kill'])

    def test_pidfd_signals_only_our_disposable_child(self):
        child=subprocess.Popen(['/usr/bin/sleep','30'])
        try:
            p=g.proc(child.pid)
            self.assertIsNotNone(p)
            self.assertTrue(g.safe_signal(p,signal.SIGTERM))
            self.assertEqual(child.wait(timeout=3),-signal.SIGTERM)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_user_service_stop_uses_target_user_bus(self):
        p=g.Proc(100,1,1034,100,'/user.slice/user@1034.service/app.slice/job.service','test','job.service',True)
        with patch.object(g.pwd,'getpwuid',return_value=SimpleNamespace(pw_gid=1035,pw_name='testuser')), \
             patch.object(g.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
            g.stop_user_unit(p,False)
            self.assertNotIn('user',run.call_args.kwargs)
            self.assertEqual(run.call_args.args[0],['/usr/bin/systemctl','--user','--machine=testuser@.host','--no-block','stop','job.service'])

if __name__=='__main__':
    unittest.main()
