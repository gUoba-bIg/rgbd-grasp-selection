"""Run isolated synthetic suites and retain one complete pre-freeze test log."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pipeline import project_environment
from scripts import multiview_common_v14 as common


def main():
    protocol = common.read(common.CONFIG)
    common.require('logs/P3/multiview_tests_v14.log' not in protocol['source_hashes'],
                   'Test log is frozen; run individual unittest suites without overwriting the sealed log')
    chunks, failed = [], False
    for suite, runtime in [('generation', 'vision'), ('geometry', 'grasp_eval'),
                           ('assets', 'vision'), ('wrapper', 'vision'), ('acceptance', 'vision')]:
        command = [str(ROOT / ('runtime/envs/' + runtime + '/Scripts/python.exe')), '-B', '-m', 'unittest',
                   'discover', '-s', 'tests', '-p', 'test_multiview_' + suite + '_v14.py', '-v']
        result = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
        chunks.append(('SUITE ' + suite + ' RUNTIME ' + runtime + '\n').encode('utf-8') + result.stdout)
        print(result.stdout.decode('utf-8', errors='replace'), flush=True)
        failed = failed or result.returncode != 0
    chunks.append(('ALL_PASSED=' + str(not failed) + '\n').encode('utf-8'))
    common.atomic_bytes(ROOT / 'logs/P3/multiview_tests_v14.log', b'\n'.join(chunks))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
