"""Run CPU-only model, storage and orchestration checks before source freeze."""
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
    for interpreter, modules in (
        ('runtime/envs/vision/Scripts/python.exe', ['tests.test_dualstream_model_v18']),
        ('runtime/envs/grasp_eval/Scripts/python.exe',
         ['tests.test_dualstream_resources_v18', 'tests.test_dualstream_resources_wrapper_v18']),
    ):
        command = [str(ROOT/interpreter), '-B', '-m', 'unittest', '-v'] + modules
        print('CPU TESTS: ' + ' '.join(modules), flush=True)
        result = subprocess.run(command, cwd=ROOT, env=environment, timeout=180)
        if result.returncode:
            return result.returncode
    print('All CPU-only dual-stream resource tests passed.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
