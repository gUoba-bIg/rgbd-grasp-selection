"""Lossless standard Zstandard frames for formal raw static/native evidence. The decoder accepts streamed frames."""
import builtins, ctypes as C, io, os, gzip
from pathlib import Path
DLL=Path(os.environ.get("ProgramFiles","C:/Program Files"))/"Git/mingw64/bin/libzstd.dll"
z=C.CDLL(str(DLL))
S=C.c_size_t;P=C.c_void_p
class Buf(C.Structure):
    _fields_=[("data",P),("size",S),("pos",S)]
def fun(name,restype,*params):
    f=getattr(z,name);f.argtypes=list(params);f.restype=restype;return f
error=fun("ZSTD_isError",C.c_uint,S);error_name=fun("ZSTD_getErrorName",C.c_char_p,S)
newc=fun("ZSTD_createCCtx",P);freec=fun("ZSTD_freeCCtx",S,P)
setparam=fun("ZSTD_CCtx_setParameter",S,P,C.c_int,C.c_int)
compress=fun("ZSTD_compressStream2",S,P,C.POINTER(Buf),C.POINTER(Buf),C.c_int)
newd=fun("ZSTD_createDCtx",P);freed=fun("ZSTD_freeDCtx",S,P)
decompress=fun("ZSTD_decompressStream",S,P,C.POINTER(Buf),C.POINTER(Buf))
def checked(code):
    if error(code):raise IOError(error_name(code).decode())
    return code
class Writer(io.BufferedIOBase):
    def __init__(self,p,mode,level):
        super().__init__();self.fileobj=builtins.open(p,mode);self.ctx=newc()
        if not self.ctx:raise MemoryError("ZSTD_createCCtx")
        checked(setparam(self.ctx,100,level));checked(setparam(self.ctx,101,24));checked(setparam(self.ctx,201,1))
        self.out=C.create_string_buffer(131072);self.pending=bytearray()
    def writable(self):return True
    def pump(self,op):
        raw=bytes(self.pending);self.pending.clear()
        b=Buf(C.cast(C.c_char_p(raw),P),len(raw),0)
        while True:
            ob=Buf(C.cast(self.out,P),len(self.out),0)
            rem=checked(compress(self.ctx,C.byref(ob),C.byref(b),op))
            if ob.pos:self.fileobj.write(self.out.raw[:ob.pos])
            if b.pos==b.size and (op==0 or rem==0):break
    def write(self,data):
        self.pending.extend(data)
        if len(self.pending)>=262144:self.pump(0)
        return len(data)
    def flush(self):
        if self.ctx:
            self.pump(1);self.fileobj.flush()
    def close(self):
        if self.closed:return
        try:
            self.pump(2);self.fileobj.flush()
        finally:
            freec(self.ctx);self.ctx=None;self.fileobj.close();super().close()
class Reader(io.RawIOBase):
    def __init__(self,p):
        super().__init__();self.file=builtins.open(p,"rb");self.ctx=newd();self.input=b"";self.pos=0;self.last=1
        if not self.ctx:raise MemoryError("ZSTD_createDCtx")
    def readable(self):return True
    def readinto(self,b):
        target=(C.c_char*len(b)).from_buffer(b);ob=Buf(C.cast(target,P),len(b),0)
        while not ob.pos:
            if self.pos==len(self.input):
                self.input=self.file.read(131072);self.pos=0
                if not self.input:
                    if self.last:raise EOFError("incomplete Zstandard stream")
                    return 0
            ib=Buf(C.cast(C.c_char_p(self.input),P),len(self.input),self.pos)
            self.last=checked(decompress(self.ctx,C.byref(ob),C.byref(ib)));self.pos=ib.pos
        return ob.pos
    def close(self):
        if not self.closed:
            self.file.close();freed(self.ctx);super().close()
def open(path,mode="rb",compresslevel=3,encoding=None,**kwargs):
    if not str(path).endswith(".zst"):return gzip.open(path,mode,compresslevel=compresslevel,encoding=encoding,**kwargs)
    if "r" in mode:stream=io.BufferedReader(Reader(path),262144)
    else:stream=Writer(path,"xb" if "x" in mode else "wb",3)
    if "t" in mode:return io.TextIOWrapper(stream,encoding=encoding or "utf8",newline=kwargs.get("newline"))
    return stream
