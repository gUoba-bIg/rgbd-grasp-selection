"""Save the complete paired-baseline component tests before source freezing."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pipeline import project_environment
from scripts import multiview_common_v14r1 as storage


def main():
    protocol=storage.read(ROOT/'configs/segmentation_baseline_v16.json')
    name='logs/P4/segmentation_baseline_v16_tests.log'
    storage.require(name not in protocol['source_hashes'],'Final test log is already frozen')
    suites=('test_segmentation_baseline_v16.py','test_baseline_resume_v16.py',
            'test_baseline_evaluation_v16.py','test_baseline_wrapper_v16.py')
    chunks=[]
    failed=False
    for suite in suites:
        command=[str(ROOT/protocol['interpreter']),'-B','-m','unittest','discover','-s','tests','-p',suite,'-v']
        result=subprocess.run(command,cwd=ROOT,env=project_environment(ROOT),stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,timeout=240)
        chunks.append(('SUITE '+suite+'\n').encode()+result.stdout)
        print(result.stdout.decode('utf-8',errors='replace')[-1800:],flush=True)
        failed=failed or result.returncode != 0 or b'Ran 0 tests' in result.stdout
    chunks.append(('ALL_PASSED='+str(not failed)+'\n').encode())
    storage.atomic_bytes(ROOT/name,b'\n'.join(chunks))
    if failed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
