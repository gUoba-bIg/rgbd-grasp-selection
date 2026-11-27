"""Retain the complete synthetic test run before source freezing."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import local_risk_common_v15 as common
from scripts.pipeline import project_environment


def main():
    protocol = common.read(common.CONFIG)
    name = 'logs/P3/local_risk_tests_v15.log'
    common.require(name not in protocol['source_hashes'], 'Test log is frozen; run suites without overwriting it')
    chunks, failed = [], False
    for suite in ('common', 'features', 'evaluation', 'acceptance', 'wrapper'):
        command = [str(ROOT / protocol['interpreter']), '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                   '-p', 'test_local_risk_' + suite + '_v15.py', '-v']
        result = subprocess.run(command, cwd=ROOT, env=project_environment(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=180)
        chunks.append(('SUITE ' + suite + '\n').encode() + result.stdout)
        print(result.stdout.decode('utf-8', errors='replace'), flush=True)
        failed = failed or result.returncode != 0 or b'Ran 0 tests' in result.stdout
    chunks.append(('ALL_PASSED=' + str(not failed) + '\n').encode())
    common.atomic_bytes(ROOT / name, b'\n'.join(chunks))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
