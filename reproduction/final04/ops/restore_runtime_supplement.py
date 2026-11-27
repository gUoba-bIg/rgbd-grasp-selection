"""Restore only the four frozen execution originals on another clean deployment. An occupied target fails."""
import argparse,hashlib,json,stat,zipfile
from pathlib import Path,PurePosixPath
PIN='8b8d8e721bfaf00c0d83f719053bbb951af1eacdfea512ccacd39621ba5e04bf'
def main():
    a=argparse.ArgumentParser();a.add_argument('target',type=Path);a.add_argument('volume',type=Path);a.add_argument('manifest',type=Path);args=a.parse_args()
    base=(args.target.resolve(strict=True)/'p').resolve(strict=True)
    assert (args.target/'setup_complete.json').is_file()
    with args.volume.open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==PIN
    m=json.loads(args.manifest.read_bytes());assert m['sha256']==PIN and len(m['files'])==4
    checked=[]
    with zipfile.ZipFile(args.volume)as z:
        for r in m['files']:
            rel=PurePosixPath(r['registered_path'])
            assert not rel.is_absolute() and '..'not in rel.parts and ':'not in str(rel)
            dst=(base/str(rel)).resolve();assert dst.is_relative_to(base) and not dst.exists()
            b=z.read(str(rel));assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
            checked.append((dst,b,r['sha256']))
    for dst,b,h in checked:
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb')as f:f.write(b)
        assert hashlib.sha256(dst.read_bytes()).hexdigest()==h
        dst.chmod(stat.S_IREAD)
    print('Verified four original runtime members at their registered locations.')
if __name__=='__main__':main()
