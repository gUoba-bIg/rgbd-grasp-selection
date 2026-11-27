"""Byte-copy independent storage objects, re-read each, then apply Windows read-only attribute."""
from seal_io import *
import shutil
def copy_checked(source,destination,expected):
    if destination.exists():
        check(not os.path.samefile(source,destination) and sha(source)==expected==sha(destination),destination)
    else:
        with source.open('rb') as a,destination.open('xb') as b:shutil.copyfileobj(a,b,8*1024**2)
        check(not os.path.samefile(source,destination) and sha(destination)==expected,(source,destination))
    os.chmod(destination,stat.S_IREAD)
    return dict(file=source.name,sha256=expected,bytes=source.stat().st_size,different_file_id=source.stat().st_ino!=destination.stat().st_ino,readonly=bool(destination.stat().st_file_attributes&stat.FILE_ATTRIBUTE_READONLY))
if __name__=='__main__':
    import sys
    dst=V/'v1_archive_readonly'; records=[]
    for p in sorted((REPO/'locks').glob('*.receipt.json')):
        r=read(p);records.append(copy_checked(ARCH/r['name'],dst/r['name'],r['sha256']))
    put(V/'external_checksums/payload_copy_verification.json',{'files':records,'verified_utc':utc(),'scope':'independent files on same volume; read-only attribute and hashes, no physical WORM claim'})
