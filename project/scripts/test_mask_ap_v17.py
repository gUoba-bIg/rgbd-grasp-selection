"""Record fixed synthetic suites before environment or main evaluation freezing."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from scripts import mask_ap_common_v17 as common
from scripts.pipeline import project_environment


def main(phase):
    env_phase=phase=='environment'
    protocol=common.read(ROOT/('configs/coco_eval_environment_v17.json' if env_phase else 'configs/mask_ap_v17.json'))
    name='logs/P4/mask_ap_v17/'+phase+'_tests.log'
    common.require(name not in protocol['source_hashes'],'Test log is already frozen')
    if env_phase:
        suites=[('runtime/envs/grasp_eval/Scripts/python.exe','test_coco_environment_v17.py')]
    else:
        suites=[(protocol['interpreter'],'test_mask_scores_v17.py'),
            (protocol['evaluation_interpreter'],'test_mask_ap_evaluation_v17.py'),
            (protocol['evaluation_interpreter'],'test_mask_ap_acceptance_v17.py'),
            (protocol['evaluation_interpreter'],'test_mask_ap_wrapper_v17.py')]
    chunks=[];failed=False
    for interpreter,suite in suites:
        result=subprocess.run([str(ROOT/interpreter),'-B','-m','unittest','discover','-s','tests','-p',suite,'-v'],
            cwd=ROOT,env=project_environment(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300)
        chunks.append(('SUITE '+suite+'\n').encode()+result.stdout)
        print(result.stdout.decode('utf-8',errors='replace')[-2000:],flush=True)
        failed=failed or result.returncode!=0 or b'Ran 0 tests' in result.stdout or b'skipped=' in result.stdout
    chunks.append(('ALL_PASSED='+str(not failed)+'\n').encode())
    common.atomic_bytes(ROOT/name,b'\n'.join(chunks))
    if failed: raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('environment','main'),required=True)
    main(parser.parse_args().phase)
