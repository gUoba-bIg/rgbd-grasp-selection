"""Standard NPZ with lossless LZMA; preserves every original array bit."""
import io,zipfile
from contextlib import contextmanager
import numpy as np

def save_cache(path,**arrays):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_LZMA) as archive:
        for key,value in arrays.items():
            buffer=io.BytesIO();np.save(buffer,value,allow_pickle=False);archive.writestr(key+'.npy',buffer.getvalue())

@contextmanager
def load_cache(path):
    with np.load(path,allow_pickle=False) as z:yield z
