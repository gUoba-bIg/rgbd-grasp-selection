"""Run every primary-evaluation synthetic test in its required isolated runtime."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from scripts import mask_ap_common_v17 as common
from scripts.pipeline import project_environment


def main():
    protocol=common.read(ROOT/'configs/mask_ap_v17.json')
    output='logs/P4/mask_ap_v17/main_tests.log'
    common.require(output not in protocol['source_hashes'],'Main test log is already frozen')
    vision=protocol['interpreter']; coco=protocol['evaluation_interpreter']
    suites=[(vision,['test_mask_scores_v17']),
            (coco,['test_mask_ap_evaluation_v17']),
            (vision,['test_mask_ap_acceptance_v17.'+name for name in
                     ('DecoderTests','RLETests','PrimaryAPTests','StorageAuditTests')]),
            (coco,['test_mask_ap_acceptance_v17.OfficialReferenceTests']),
            (coco,['test_mask_ap_wrapper_v17'])]
    chunks=[]; failed=False
    for interpreter,names in suites:
        source=('import sys,unittest; sys.path.insert(0,"tests"); '
                'suite=unittest.defaultTestLoader.loadTestsFromNames('+repr(names)+'); '
                'result=unittest.TextTestRunner(verbosity=2).run(suite); '
                'sys.exit(0 if result.wasSuccessful() and result.testsRun>0 and not result.skipped else 1)')
        command=[str(ROOT/interpreter),'-B','-c',source]
        result=subprocess.run(command,cwd=ROOT,env=project_environment(ROOT),stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,timeout=300)
        chunks.append(('RUNTIME '+interpreter+'\nSUITES '+json.dumps(names)+'\n').encode()+result.stdout)
        print(result.stdout.decode('utf-8',errors='replace')[-2200:],flush=True)
        failed=failed or result.returncode!=0
    chunks.append(('ALL_PASSED='+str(not failed)+'\n').encode())
    common.atomic_bytes(ROOT/output,b'\n'.join(chunks))
    if failed: raise SystemExit(1)


if __name__=='__main__':
    main()
