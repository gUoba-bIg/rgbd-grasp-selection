"""CPU component and simulated-checkpoint tests before research source freeze."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts.pipeline import project_environment


if __name__ == '__main__':
    environment = project_environment(ROOT)
    environment['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '4'
    modules = ['tests.test_segmentation_quality_v18', 'tests.test_dualstream_training_v18',
        'tests.test_dualstream_training_resume_v18', 'tests.test_dualstream_training_wrapper_v18']
    print('CPU synthetic/component tests; no research frames or GPU: ' + ' '.join(modules), flush=True)
    result = subprocess.run([str(ROOT/'runtime/envs/vision/Scripts/python.exe'), '-B', '-m', 'unittest', '-v']
        + modules, cwd=ROOT, env=environment, timeout=300)
    sys.exit(result.returncode)
