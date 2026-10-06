#!/usr/bin/env python3
"""Build a self-contained root installer for the guard only, without GPU maintenance."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


ROOT_INSTALLER = r'''
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

if os.geteuid() != 0:
    raise SystemExit('The installer must run as root')
if socket.gethostname() != payload['host']:
    raise SystemExit('Host mismatch: refusing to install a policy prepared for another machine')

def run(*args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=45,
                            env={**os.environ, 'SYSTEMD_IGNORE_CHROOT': '1',
                                 'SYSTEMD_LOG_TARGET': 'console'})
    if result.returncode:
        print(result.stdout + result.stderr, flush=True)
        result.check_returncode()
    return result

stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
backup = Path('/var/lib/gpu-guard/backups') / ('deploy-' + stamp)
for path, mode in [('/usr/local/lib/gpu-guard', 0o755), ('/etc/gpu-guard', 0o755),
                   ('/var/log/gpu-guard', 0o750), ('/var/lib/gpu-guard', 0o700),
                   ('/var/lib/gpu-guard/backups', 0o700), (str(backup), 0o700)]:
    target = Path(path)
    if target.is_symlink():
        raise SystemExit('Refusing symlink directory: ' + path)
    target.mkdir(parents=True, exist_ok=True)
    os.chown(target, 0, 0)
    os.chmod(target, mode)

manifest = {'host': payload['host'], 'time_utc': stamp, 'backup': str(backup), 'files': {}}
for filename, encoded in payload['files'].items():
    target = Path(filename)
    if target.is_symlink():
        raise SystemExit('Refusing symlink target: ' + filename)
    saved = backup / filename.lstrip('/')
    if target.exists():
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, saved)
    data = base64.b64decode(encoded, validate=True)
    temporary = target.with_name(target.name + '.gpu-guard-install')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
    os.chown(temporary, 0, 0)
    os.chmod(temporary, 0o644)
    temporary.replace(target)
    manifest['files'][filename] = hashlib.sha256(data).hexdigest()

run('/usr/bin/systemd-analyze', 'verify', '/etc/systemd/system/gpu-guard.service')
run('/usr/bin/systemctl', 'daemon-reload')
run('/usr/bin/systemctl', 'enable', 'gpu-guard.service')
run('/usr/bin/systemctl', 'restart', 'gpu-guard.service')
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    status_file = Path('/run/gpu-guard/status.json')
    if status_file.exists():
        status = json.loads(status_file.read_text())
        if status.get('time', 0) >= time.time() - 10 and status.get('telemetry') == 'ok' and status.get('enforce'):
            if status.get('gpu_count') != payload['expected_gpu_count']:
                raise SystemExit('Guard is running but GPU count differs from the preflight inventory')
            manifest['initial_status'] = status
            break
    time.sleep(1)
else:
    print(run('/usr/bin/journalctl', '-u', 'gpu-guard.service', '-n', '30', '--no-pager').stdout)
    raise SystemExit('Guard did not produce fresh, enforcing GPU telemetry')
manifest['service'] = run('/usr/bin/systemctl', 'show', 'gpu-guard.service',
                          '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'NRestarts').stdout
manifest['enabled'] = run('/usr/bin/systemctl', 'is-enabled', 'gpu-guard.service').stdout.strip()
manifest_file = Path('/var/lib/gpu-guard/deployment.json')
manifest_file.write_text(json.dumps(manifest, indent=2) + '\n')
os.chown(manifest_file, 0, 0)
os.chmod(manifest_file, 0o600)
print(json.dumps(manifest, indent=2))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--gpu-count', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-polkit', action='store_true',
                        help='omit the JS polkit rule on hosts running polkit 0.105')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    config = json.loads((source / 'config.json').read_text())
    assert config['enforce'] is True
    service = (source / 'gpu-guard.service').read_text().replace(' gpu-quarantine.service', '')
    policy = (source / 'POLICY.zh-CN.txt').read_text().split('\n已隔离硬件：', 1)[0]
    policy += '\n部署主机：' + args.host + '。本次仅安装占用管理服务，硬件隔离须依据本机诊断单独实施。\n'
    content = {
        '/usr/local/lib/gpu-guard/gpu_guard.py': (source / 'gpu_guard.py').read_bytes(),
        '/etc/gpu-guard/config.json': (json.dumps(config, indent=2) + '\n').encode(),
        '/etc/gpu-guard/POLICY.zh-CN.txt': policy.encode(),
        '/etc/systemd/system/gpu-guard.service': service.encode(),
    }
    # polkit 0.105 (Debian/Ubuntu legacy) ignores JS rules; its manage-units default is auth_admin.
    if not args.skip_polkit:
        content['/etc/polkit-1/rules.d/49-gpu-guard.rules'] = (source / '49-gpu-guard.rules').read_bytes()
    payload = {'host': args.host, 'expected_gpu_count': args.gpu_count,
               'files': {name: base64.b64encode(data).decode() for name, data in content.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('payload = ' + repr(payload) + '\n' + ROOT_INSTALLER)
    args.output.chmod(0o600)
    print(json.dumps({'installer': str(args.output.resolve()), 'host': args.host,
                      'files': {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}}, indent=2))


if __name__ == '__main__':
    main()
