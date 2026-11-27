"""Wait for actual returned receipts and publish a verified, lightweight local volume."""
import time
import sys
from common_v1 import *
from after_network import call

def wait_for(path):
    study=OUT/"study"
    while not path.is_file():
        if (study/"STOP.json").is_file() or (study/"method_STOP.json").is_file():
            raise RuntimeError("shared stop recorded before "+str(path))
        time.sleep(45)

if __name__=="__main__":
    wait_for(OUT/"study/run/rank_completed.json");disk()
    call("per_unit_source_index",sys.executable,"observation_index.py")
    wait_for(OUT/"admission/completed_stages.json");disk()
    call("Chinese_report",C/"env/rank/Scripts/python.exe","reporting.py")
    call("review_package",sys.executable,"deliver.py")
