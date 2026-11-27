"""Seal the R8 derived manuscript and verify each member of the transferable archive."""
import csv,json,hashlib,zipfile
from pathlib import Path,PurePosixPath
V=Path(__file__).resolve().parents[1]
folders=('manuscript','tables','figures','supplement','evidence','scripts','source')
def sd(data):return hashlib.sha256(data).hexdigest()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def kind(rel):
    return {'manuscript':'论文','tables':'统计和试次证据','figures':'图稿和图注','supplement':'补充材料','evidence':'封存和核验','scripts':'离线分析与排版源码','source':'稿件结构及引文'}.get(rel.parts[0],'阅读入口')
items=[Path('README.md')]+[p.relative_to(V) for folder in folders for p in (V/folder).rglob('*') if p.is_file() and not (p.name.startswith(('~$','.','~')) or '__pycache__' in p.parts or p.suffix=='.pyc')]
items=sorted(items,key=lambda p:p.as_posix());manifest=V/'SHA256SUMS.txt';dest=V/'delivery'/'R8_论文与统计轻量审查包.zip'
if manifest.exists() or dest.exists():raise FileExistsError('R8 delivery already sealed')
with (V/'目录清单.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['包内路径','字节数','类别','来源'])
    for item in items:w.writerow([item.as_posix(),(V/item).stat().st_size,kind(item),'R8 派生'])
items=sorted([*items,Path('目录清单.csv')],key=lambda p:p.as_posix())
checks={p.as_posix():sha(V/p) for p in items}
with manifest.open('x',encoding='utf8',newline='\n') as f:
    for rel,h in checks.items():f.write(h+'  '+rel+'\n')
with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in [*items,Path(manifest.name)]:
        name=p.as_posix();z.write(V/p,name,compress_type=zipfile.ZIP_STORED if p.suffix in ('.gz','.docx','.png') else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(dest) as z:
    names=z.namelist()
    assert len(names)==len(set(s.casefold() for s in names)) and len(names)==len(checks)+1
    assert z.testzip() is None
    assert set(names)==set(checks)|{manifest.name}
    for s in names:
        assert not PurePosixPath(s).is_absolute() and '..' not in PurePosixPath(s).parts and ':' not in s and '\\' not in s,s
        assert sd(z.read(s))==sha(V/s)==checks.get(s,sha(manifest)),s
digest=sha(dest)
with dest.with_name(dest.name+'.sha256').open('x',encoding='utf8') as f:f.write(digest+'  '+dest.name+'\n')
result=dict(zip=dest.name,bytes=dest.stat().st_size,members=len(names),manifest_members=len(checks),
            crc_verified=True,all_content_hashes_verified=True,zip_sha256=digest,manifest_sha256=sha(manifest),
            large_original_files_included=0)
with (dest.parent/'交付核验.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False))
