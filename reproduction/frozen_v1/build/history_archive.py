from seal_io import *
anchors=read(ROOT/R8/'evidence/history_seal.json')['source_versions']
for phase,name in [('source_R6_initial',P0),('source_R6_final',R6),('source_R7_final',R7),('source_R8_paper',R8)]:
    d=ROOT/name;mf=d/'SHA256SUMS.txt';m=manifest(mf)
    fn={name+'/'+k:(d/k,h) for k,h in m.items()}
    fn[name+'/SHA256SUMS.txt']=(mf,None)
    for p in walk(d/'delivery'):fn[p.relative_to(ROOT).as_posix()]=(p,None)
    if name in [R6,R7]:
        item=next(x for x in anchors if x['directory']==name)
        check(sha(mf)==item['manifest_sha256'],name)
        fn[name+'/'+item['archived_core']]=(d/item['archived_core'],item['core_sha256'])
    if name==R8:
        ref=read(d/'delivery/交付核验.json')
        check(sha(mf)==ref['manifest_sha256'],name);fn[name+'/delivery/'+ref['zip']]=(d/'delivery'/ref['zip'],ref['zip_sha256'])
    pack(phase,fn)
pack('source_bilingual_final',{p.relative_to(ROOT).as_posix():(p,None) for p in walk(ROOT/'成文')})
