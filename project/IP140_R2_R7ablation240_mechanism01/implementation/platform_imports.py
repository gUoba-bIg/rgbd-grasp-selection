"""Load frozen engineering libraries for standalone R4 processes, with no default trial."""
from pathlib import Path
import sys
from types import ModuleType

VERSION=Path(__file__).resolve().parents[1]
ROOT=VERSION.parent
PLATFORM=ROOT/"IP140_R2_smoke22_miss_negative_controls_full_execution01"
def bind_frozen_libraries():
    if 'batch_context' not in sys.modules:
        context=ModuleType('batch_context')
        context.VERSION=PLATFORM
        context.ROOT=ROOT
        context.SLOT=-1
        context.SLOT_DIR=VERSION/'unbound_native_worker'
        context.OBJECT_ID=None
        context.KIND='r4_algorithm'
        sys.modules['batch_context']=context
    else:
        context=sys.modules['batch_context']
        if context.KIND!='r4_algorithm' or context.SLOT!=-1: raise RuntimeError('mixed engineering and R4 contexts')
    if str(PLATFORM) not in sys.path:
        sys.path.append(str(PLATFORM))
