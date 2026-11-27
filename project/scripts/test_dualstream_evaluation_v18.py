"""Run isolated evaluation component suites with CUDA disabled."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts.pipeline import project_environment


def main():
    environment = project_environment(ROOT)
    environment['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '4'
    suites = [
        ('runtime/envs/vision/Scripts/python.exe', ['tests.test_dualstream_scores_v18']),
        ('runtime/envs/coco_eval_v17/Scripts/python.exe', ['tests.test_dualstream_ap_v18']),
        ('runtime/envs/grasp_eval/Scripts/python.exe', ['tests.test_dualstream_evaluation_wrapper_v18']),
        ('runtime/envs/grasp_eval/Scripts/python.exe', ['tests.test_dualstream_acceptance_v18.'+name for name in ('NumericalTests', 'InputAndSummaryTests', 'RecoveryTests', 'BlockTests', 'EntryTests')]),
        ('runtime/envs/vision/Scripts/python.exe', ['tests.test_dualstream_acceptance_v18.VisionReferenceTests']),
    ]
    for interpreter, modules in suites:
        print('CPU evaluation tests: '+', '.join(modules), flush=True)
        result = subprocess.run([str(ROOT/interpreter), '-B', '-m', 'unittest', '-v']+modules,
                                cwd=ROOT, env=environment, timeout=300)
        if result.returncode: return result.returncode
    print('All CPU evaluation tests passed; no real evaluation performed.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
