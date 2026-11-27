"""Resolve the authoritative final ledger while keeping the original paused snapshot."""
from common_v1 import OUT,need

def method_ledger():
    directory=OUT/"study/run"
    if (OUT/"study/method_STOP.json").exists():
        final=directory/"method_ledger_continued.json"
        need(final.is_file(),"a retained pause exists; no final continued ledger")
        return final
    return directory/"method_ledger.json"
