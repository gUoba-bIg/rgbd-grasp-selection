"""Run synthetic geometry component suites without production sampling."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from scripts.pipeline import project_environment


def main():
    environment = project_environment(ROOT)
    environment['CUDA_VISIBLE_DEVICES'] = ''
    environment['MPLCONFIGDIR'] = str(ROOT/'logs/P6/geometry_propagation_v19/test_plot_cache')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '4'
    modules = ['tests.test_geometry_propagation_v19', 'tests.test_geometry_propagation_storage_v19',
               'tests.test_geometry_propagation_wrapper_v19', 'tests.test_geometry_propagation_acceptance_v19']
    return subprocess.run([str(ROOT/'runtime/envs/grasp_eval/Scripts/python.exe'), '-B', '-m', 'unittest', '-v']+modules,
        cwd=ROOT, env=environment, timeout=300).returncode


if __name__ == '__main__': sys.exit(main())
