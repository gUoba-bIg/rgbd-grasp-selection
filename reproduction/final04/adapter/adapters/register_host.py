"""Establish one immutable host binding before preflight. Does not load the engine or either network."""
import json,sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from amsi_origin import observe,match
target=Path(sys.argv[1]).resolve()
if Path(sys.executable).resolve()!=target/'bin/python.exe':raise RuntimeError('not a fresh bootstrap')
root=target/'setup_logs'
d=observe();f=Path(d['registry']['canonical_path'])
def out(name,obj):
    with (root/name).open('x',encoding='utf8') as fp:json.dump(obj,fp,ensure_ascii=False,indent=2)
out('registered_host_provider.json',d)
rows=[]
def test(k,actual,want):
    rows.append({'fixture':k,'accepted':actual,'expected':want,'pass':actual==want})
test('registered64_signed_server',match(d,f,d),True)
for key,v in [('signature_status','UnknownError'),('signer_simple','Microsoft Corporation'),('bytes',0),('sha256','1'*64),('version','test-changed'),('thumbprint','0'*40)]:
    q=deepcopy(d);q['signature'][key]=v;test(key,match(q,f,d),False)
for k,v in [('view',32),('provider','{00000000-0000-0000-0000-000000000000}'),('value','unexpected changed registration')]:
    q=deepcopy(d);q['registry'][k]=v;test(k,match(q,f,d),False)
test('same_vendor_unregistered_file',match(d,Path(sys.executable),d),False)
test('sibling_same_directory_not_whitelisted',match(d,f.with_name('MsMpEng.exe'),d),False)
out('provider_boundary_tests.json',rows)
if not all(x['pass'] for x in rows):raise RuntimeError('registration regression')
